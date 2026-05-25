from __future__ import annotations

import json
import math
import os
import threading
import time
from pathlib import Path
from typing import Any

import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import safe_open
from scipy.signal import resample_poly


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
    ) -> None:
        super().__init__()
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=n_mels,
            norm="slaney",
            mel_scale="slaney",
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        mel = self.mel_transform(waveform)
        log_mel = torch.clamp(mel, min=1e-10).log10()
        return (log_mel + 4.0) / 4.0


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        weights = self.query(x).squeeze(-1)
        if mask is not None:
            weights = weights.masked_fill(~mask, float("-inf"))
        weights = F.softmax(weights, dim=-1)
        return torch.bmm(weights.unsqueeze(1), x).squeeze(1)


class ConvFrontend(nn.Module):
    def __init__(self, n_mels: int, d_model: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, d_model // 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model // 2, d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)


class AudioQualityClassifier(nn.Module):
    def __init__(
        self,
        n_mels: int = 80,
        d_model: int = 192,
        nhead: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 3000,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.downsample_rate = 4
        self.frontend = ConvFrontend(n_mels, d_model, dropout)
        self.pos_encoder = PositionalEncoding(d_model, max_len // 4 + 100, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=1,
            norm=nn.LayerNorm(d_model),
        )
        self.pooling = AttentionPooling(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, mels: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.frontend(mels)
        time_steps = x.shape[1]
        if mask is not None:
            mask = mask[:, :: self.downsample_rate]
            if mask.shape[1] > time_steps:
                mask = mask[:, :time_steps]
            elif mask.shape[1] < time_steps:
                pad = torch.ones(
                    mask.shape[0],
                    time_steps - mask.shape[1],
                    device=mask.device,
                    dtype=mask.dtype,
                )
                mask = torch.cat([mask, pad], dim=1)
        x = self.pos_encoder(x)
        src_key_padding_mask = ~mask if mask is not None else None
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.pooling(x, mask)
        return self.classifier(x)


class AudioQualityRouter:
    def __init__(
        self,
        checkpoint_path: str | os.PathLike[str],
        *,
        device: str | None = None,
        threshold: float = 0.5,
        sample_rate: int = 16000,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(threshold)
        self.sample_rate = int(sample_rate)
        self.model, self.mel_extractor = self._load_model()

    def _load_model(self) -> tuple[nn.Module, nn.Module]:
        checkpoint_path = Path(self.checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            checkpoint_config = json.loads(metadata.get("config", "{}")) if metadata else {}
            config = checkpoint_config.get("model", {})
            state_dict = safe_load_file(str(checkpoint_path), device=self.device)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            config = checkpoint.get("config", {}).get("model", {})
            state_dict = checkpoint["model_state_dict"]

        model = AudioQualityClassifier(
            n_mels=config.get("n_mels", 80),
            d_model=config.get("d_model", 192),
            nhead=config.get("nhead", 4),
            dim_feedforward=config.get("dim_feedforward", 512),
            dropout=config.get("dropout", 0.1),
            max_len=config.get("max_len", 3000),
            num_classes=config.get("num_classes", 2),
        )
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        feature_config = checkpoint_config.get("feature", {}) if checkpoint_path.suffix == ".safetensors" else {}
        mel_extractor = LogMelSpectrogram(
            sample_rate=self.sample_rate,
            n_mels=config.get("n_mels", 80),
            n_fft=feature_config.get("n_fft", 400),
            hop_length=feature_config.get("hop_length", 160),
            win_length=feature_config.get("win_length", 400),
        ).to(self.device)
        mel_extractor.eval()
        return model, mel_extractor

    def _load_audio(self, audio_path: str | os.PathLike[str]) -> torch.Tensor:
        audio_np, sr = sf.read(str(audio_path), always_2d=True)
        audio_np = audio_np.mean(axis=1)
        if sr != self.sample_rate:
            gcd = math.gcd(int(sr), self.sample_rate)
            audio_np = resample_poly(audio_np, self.sample_rate // gcd, int(sr) // gcd)
        waveform = torch.from_numpy(audio_np).float().unsqueeze(0)
        return waveform.to(self.device)

    @torch.no_grad()
    def infer(self, audio_path: str | os.PathLike[str]) -> dict[str, Any]:
        waveform = self._load_audio(audio_path)
        mel = self.mel_extractor(waveform)
        mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)
        logits = self.model(mel, mask=None)
        probs = torch.softmax(logits, dim=-1)
        degraded_prob = float(probs[0, 1].item())
        is_degraded = degraded_prob >= self.threshold
        return {
            "is_degraded": is_degraded,
            "degraded_prob": degraded_prob,
            "label": int(is_degraded),
        }

    def predict(self, audio_path: str | os.PathLike[str]) -> tuple[bool, float]:
        result = self.infer(audio_path)
        return bool(result["is_degraded"]), float(result["degraded_prob"])



class LoRADeltaSwitch:
    def __init__(self, keep_delta_on_gpu: bool = False) -> None:
        self.keep_delta_on_gpu = bool(keep_delta_on_gpu)
        self.active = False
        self.entries: list[dict[str, Any]] = []
        self.skipped_modules: list[str] = []
        self._delta_cache: dict[tuple[str, torch.device, torch.dtype], torch.Tensor] = {}

    def load_adapter(self, parent_module: nn.Module, adapter_dir: str | os.PathLike[str]) -> None:
        adapter_path = Path(adapter_dir)
        config = json.loads((adapter_path / "adapter_config.json").read_text())
        alpha_pattern = config.get("alpha_pattern", {}) or {}
        lora_alpha = float(config.get("lora_alpha", 1.0))
        state = safe_load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")

        entries: list[dict[str, Any]] = []
        skipped: list[str] = []
        suffix = ".lora_A.weight"
        for key in sorted(k for k in state if k.endswith(suffix)):
            module_path = key[len("base_model.model.") : -len(suffix)]
            b_key = f"base_model.model.{module_path}.lora_B.weight"
            if b_key not in state:
                raise KeyError(f"Missing LoRA B weight for {module_path}: {b_key}")
            try:
                module = parent_module.get_submodule(module_path)
            except AttributeError:
                skipped.append(module_path)
                continue
            if not hasattr(module, "weight"):
                raise TypeError(f"LoRA target has no weight parameter: {module_path}")
            if module.weight.ndim != 2:
                raise TypeError(
                    f"Unsupported LoRA target weight shape for {module_path}: "
                    f"{tuple(module.weight.shape)}"
                )

            a = state[key].contiguous()
            b = state[b_key].contiguous()
            rank = int(a.shape[0])
            alpha = float(alpha_pattern.get(module_path, lora_alpha))
            scale = alpha / float(rank)
            if tuple(module.weight.shape) != (int(b.shape[0]), int(a.shape[1])):
                raise ValueError(
                    f"LoRA shape mismatch for {module_path}: weight={tuple(module.weight.shape)}, "
                    f"A={tuple(a.shape)}, B={tuple(b.shape)}"
                )
            entries.append({"path": module_path, "module": module, "a": a, "b": b, "scale": scale})

        if not entries:
            raise ValueError(f"No matching LoRA target modules found in {adapter_path / 'adapter_model.safetensors'}")
        self.entries = entries
        self.skipped_modules = skipped

    def _delta_for(self, entry: dict[str, Any]) -> torch.Tensor:
        module = entry["module"]
        weight = module.weight
        cache_key = (entry["path"], weight.device, weight.dtype)
        if self.keep_delta_on_gpu and cache_key in self._delta_cache:
            return self._delta_cache[cache_key]

        a = entry["a"].to(device=weight.device, dtype=torch.float32)
        b = entry["b"].to(device=weight.device, dtype=torch.float32)
        delta = torch.matmul(b, a).mul_(float(entry["scale"])).to(dtype=weight.dtype)
        if self.keep_delta_on_gpu:
            self._delta_cache[cache_key] = delta
        return delta

    @torch.no_grad()
    def set_active(self, active: bool) -> float:
        active = bool(active)
        if active == self.active:
            return 0.0
        start = time.perf_counter()
        sign = 1.0 if active else -1.0
        for entry in self.entries:
            module = entry["module"]
            module.weight.data.add_(self._delta_for(entry), alpha=sign)
        self.active = active
        return time.perf_counter() - start

class Qwen3ASR:
    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        device_map: str | None = None,
        dtype: Any | None = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        **model_kwargs: Any,
    ) -> None:
        from qwen_asr import Qwen3ASRModel

        self.model_path = str(Path(model_path).expanduser())
        if device_map is None:
            device_map = "cuda:0" if torch.cuda.is_available() else "cpu"
        if dtype is None:
            dtype = torch.bfloat16 if device_map != "cpu" else torch.float32
        self.model = Qwen3ASRModel.from_pretrained(
            self.model_path,
            dtype=dtype,
            device_map=device_map,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )

    def infer(
        self,
        audio: Any,
        *,
        language: str | None = None,
        return_objects: bool = False,
        **transcribe_kwargs: Any,
    ) -> str | list[str] | Any:
        if isinstance(audio, os.PathLike):
            audio = str(audio)
        elif isinstance(audio, (list, tuple)):
            audio = [str(item) if isinstance(item, os.PathLike) else item for item in audio]
        results = self.model.transcribe(audio=audio, language=language, **transcribe_kwargs)
        if return_objects:
            return results
        if isinstance(results, list):
            return [str(getattr(result, "text", result)).strip() for result in results]
        return str(getattr(results, "text", results)).strip()


class MegaASR:
    NAME = "Mega-ASR"

    def __init__(
        self,
        model_path: str | os.PathLike[str],
        *,
        lora_dir: str | os.PathLike[str],
        router_checkpoint: str | os.PathLike[str],
        routing_enabled: bool = True,
        quality_threshold: float = 0.5,
        device_map: str | None = None,
        quality_device: str | None = None,
        max_inference_batch_size: int = 32,
        max_new_tokens: int = 256,
        keep_delta_on_gpu: bool = False,
        **model_kwargs: Any,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.lora_dir = str(Path(lora_dir).expanduser())
        self.router_checkpoint = str(Path(router_checkpoint).expanduser())
        self.routing_enabled = bool(routing_enabled)
        self.keep_delta_on_gpu = bool(keep_delta_on_gpu)
        self.stats = {"total": 0, "use_base": 0, "use_lora": 0}
        self.switch_times: list[dict[str, float | str]] = []
        self.router = None
        if self.routing_enabled:
            self.router = AudioQualityRouter(
                checkpoint_path=self.router_checkpoint,
                device=quality_device,
                threshold=quality_threshold,
            )
        self.asr = Qwen3ASR(
            model_path=self.model_path,
            device_map=device_map,
            max_inference_batch_size=max_inference_batch_size,
            max_new_tokens=max_new_tokens,
            **model_kwargs,
        )
        self._adapter_lock = threading.RLock()
        self._active_lora = False
        self.lora_switch = LoRADeltaSwitch(keep_delta_on_gpu=self.keep_delta_on_gpu)
        self._load_lora()
        self._set_lora(True)

    @staticmethod
    def _unwrap_audio(audio: Any) -> Any:
        if isinstance(audio, (list, tuple)) and len(audio) == 1:
            return audio[0]
        return audio

    def _get_inner_model(self) -> Any:
        return getattr(self.asr.model, "model", self.asr.model)

    def _load_lora(self) -> None:
        self.lora_switch.load_adapter(self._get_inner_model(), self.lora_dir)

    def _set_lora(self, active: bool) -> None:
        elapsed = self.lora_switch.set_active(active)
        if active != self._active_lora:
            direction = "base_to_lora" if active else "lora_to_base"
            self.switch_times.append({"direction": direction, "time": elapsed})
        self._active_lora = active

    def _route(self, audio: Any) -> tuple[bool, float | None, str]:
        if self.routing_enabled and self.router is not None:
            is_degraded, degraded_prob = self.router.predict(audio)
            return is_degraded, degraded_prob, "router"
        return True, None, "default"

    def _infer_with_adapter_state(self, use_lora: bool, audio: Any, **kwargs: Any) -> Any:
        with self._adapter_lock:
            self._set_lora(use_lora)
            return self.asr.infer(audio, **kwargs)

    def infer(
        self,
        audio: Any,
        *,
        language: str | None = None,
        return_objects: bool = False,
        return_route: bool = False,
        **transcribe_kwargs: Any,
    ) -> Any:
        audio = self._unwrap_audio(audio)
        use_lora, degraded_prob, route_source = self._route(audio)
        result = self._infer_with_adapter_state(
            use_lora,
            audio,
            language=language,
            return_objects=return_objects,
            **transcribe_kwargs,
        )
        self.stats["total"] += 1
        if use_lora:
            self.stats["use_lora"] += 1
        else:
            self.stats["use_base"] += 1
        if return_route:
            return {
                "text": result,
                "use_lora": use_lora,
                "degraded_prob": degraded_prob,
                "route_source": route_source,
            }
        return result

    def infer_with_lora(self, audio: Any, **kwargs: Any) -> Any:
        return self._infer_with_adapter_state(True, self._unwrap_audio(audio), **kwargs)

    def infer_without_lora(self, audio: Any, **kwargs: Any) -> Any:
        return self._infer_with_adapter_state(False, self._unwrap_audio(audio), **kwargs)

    @torch.no_grad()
    def batch_infer(self, audios: list[Any], **kwargs: Any) -> list[Any]:
        return [self.infer(audio, **kwargs) for audio in audios]
