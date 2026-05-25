from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AudioSegment:
    path: str
    index: int
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


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
        max_duration: float = 30.0,
        max_router_chunks: int = 3,
    ) -> None:
        self.checkpoint_path = str(Path(checkpoint_path).expanduser())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = float(threshold)
        self.sample_rate = int(sample_rate)
        self.max_duration = float(max_duration)
        self.max_router_chunks = max(1, int(max_router_chunks))
        self.max_samples = max(1, int(self.sample_rate * self.max_duration))
        self.model, self.mel_extractor = self._load_model()

    def _load_model(self) -> tuple[nn.Module, nn.Module]:
        checkpoint_path = Path(self.checkpoint_path)
        if checkpoint_path.suffix == ".safetensors":
            with safe_open(str(checkpoint_path), framework="pt", device="cpu") as f:
                metadata = f.metadata()
            checkpoint_config = json.loads(metadata.get("config", "{}")) if metadata else {}
            config = checkpoint_config.get("model", {})
            data_config = checkpoint_config.get("data", {})
            state_dict = safe_load_file(str(checkpoint_path), device=self.device)
        else:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            checkpoint_config = checkpoint.get("config", {})
            config = checkpoint_config.get("model", {})
            data_config = checkpoint_config.get("data", {})
            state_dict = checkpoint["model_state_dict"]

        if "sample_rate" in data_config:
            self.sample_rate = int(data_config["sample_rate"])
        if "max_duration" in data_config:
            self.max_duration = float(data_config["max_duration"])
        self.max_samples = max(1, int(self.sample_rate * self.max_duration))

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

    def _router_chunks(self, waveform: torch.Tensor) -> list[torch.Tensor]:
        total_samples = int(waveform.shape[-1])
        if total_samples <= self.max_samples:
            return [waveform]

        chunk_count = min(self.max_router_chunks, math.ceil(total_samples / self.max_samples))
        if chunk_count <= 1:
            starts = [0]
        else:
            max_start = total_samples - self.max_samples
            starts = [round(i * max_start / (chunk_count - 1)) for i in range(chunk_count)]
        return [waveform[:, start : start + self.max_samples] for start in starts]

    @torch.no_grad()
    def infer(self, audio_path: str | os.PathLike[str]) -> dict[str, Any]:
        waveform = self._load_audio(audio_path)
        degraded_probs: list[float] = []
        for chunk in self._router_chunks(waveform):
            mel = self.mel_extractor(chunk)
            mel = mel.squeeze(0).transpose(0, 1).unsqueeze(0)
            if mel.shape[1] > self.model.pos_encoder.pe.shape[1]:
                mel = mel[:, : self.model.pos_encoder.pe.shape[1], :]
            logits = self.model(mel, mask=None)
            probs = torch.softmax(logits, dim=-1)
            degraded_probs.append(float(probs[0, 1].item()))

        degraded_prob = max(degraded_probs) if degraded_probs else 0.0
        is_degraded = degraded_prob >= self.threshold
        return {
            "is_degraded": is_degraded,
            "degraded_prob": degraded_prob,
            "label": int(is_degraded),
            "router_chunks": len(degraded_probs),
            "router_max_duration": self.max_duration,
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
        max_new_tokens: int = 1024,
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



def _iter_result_objects(result: Any):
    if isinstance(result, dict) and "text" in result:
        yield from _iter_result_objects(result["text"])
    elif isinstance(result, (list, tuple)):
        yield from result
    else:
        yield result


def _extract_result_text(result: Any) -> str:
    texts: list[str] = []
    for item in _iter_result_objects(result):
        if item is None:
            continue
        if isinstance(item, str):
            text = item
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            text = item["text"]
        elif hasattr(item, "text"):
            text = getattr(item, "text")
        else:
            text = str(item)
        text = str(text).strip()
        if text:
            texts.append(text)
    return "\n".join(texts).strip()


def _extract_result_language(result: Any) -> str:
    if isinstance(result, dict):
        lang = result.get("language") or result.get("detected_language") or result.get("lang")
        if lang:
            return str(lang)
    for item in _iter_result_objects(result):
        if isinstance(item, dict):
            lang = item.get("language") or item.get("detected_language") or item.get("lang")
        else:
            lang = getattr(item, "language", None) or getattr(item, "lang", None)
        if lang:
            return str(lang)
    return ""


def _merge_segment_texts(texts: list[str]) -> str:
    return "\n".join(text.strip() for text in texts if text and text.strip()).strip()


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
        max_new_tokens: int = 1024,
        keep_delta_on_gpu: bool = False,
        split_long_audio: bool = True,
        long_audio_segment_seconds: float = 30.0,
        long_audio_overlap_seconds: float = 0.0,
        long_audio_min_tail_seconds: float = 1.0,
        **model_kwargs: Any,
    ) -> None:
        self.model_path = str(Path(model_path).expanduser())
        self.lora_dir = str(Path(lora_dir).expanduser())
        self.router_checkpoint = str(Path(router_checkpoint).expanduser())
        self.routing_enabled = bool(routing_enabled)
        self.keep_delta_on_gpu = bool(keep_delta_on_gpu)
        self.split_long_audio = bool(split_long_audio)
        self.long_audio_segment_seconds = max(1.0, float(long_audio_segment_seconds))
        self.long_audio_overlap_seconds = max(0.0, float(long_audio_overlap_seconds))
        if self.long_audio_overlap_seconds >= self.long_audio_segment_seconds:
            self.long_audio_overlap_seconds = 0.0
        self.long_audio_min_tail_seconds = max(0.0, float(long_audio_min_tail_seconds))
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

    @staticmethod
    def _local_audio_path(audio: Any) -> Path | None:
        if not isinstance(audio, (str, os.PathLike)):
            return None
        path = Path(audio).expanduser()
        if not path.is_file():
            return None
        return path

    def _segment_windows(self, duration: float) -> list[tuple[float, float]]:
        if duration <= self.long_audio_segment_seconds:
            return []

        windows: list[tuple[float, float]] = []
        step = max(0.1, self.long_audio_segment_seconds - self.long_audio_overlap_seconds)
        start = 0.0
        while start < duration:
            remaining = duration - start
            if windows and remaining <= self.long_audio_min_tail_seconds:
                previous_start, _ = windows[-1]
                windows[-1] = (previous_start, duration)
                break

            end = min(duration, start + self.long_audio_segment_seconds)
            windows.append((start, end))
            if end >= duration:
                break
            start += step

        return windows if len(windows) > 1 else []

    def _split_audio_with_soundfile(self, audio_path: Path, temp_dir: Path) -> list[AudioSegment]:
        segments: list[AudioSegment] = []
        with sf.SoundFile(str(audio_path)) as source:
            sample_rate = int(source.samplerate)
            total_frames = int(len(source))
            if sample_rate <= 0 or total_frames <= 0:
                return []

            windows = self._segment_windows(total_frames / float(sample_rate))
            for index, (start, end) in enumerate(windows):
                start_frame = max(0, min(total_frames, round(start * sample_rate)))
                end_frame = max(start_frame, min(total_frames, round(end * sample_rate)))
                frame_count = end_frame - start_frame
                if frame_count <= 0:
                    continue

                source.seek(start_frame)
                data = source.read(frame_count, dtype="float32", always_2d=True)
                if data.size == 0:
                    continue

                chunk_path = temp_dir / f"chunk_{index:04d}_{start_frame}_{end_frame}.wav"
                sf.write(str(chunk_path), data, sample_rate)
                segments.append(
                    AudioSegment(
                        path=str(chunk_path),
                        index=index,
                        start=start_frame / float(sample_rate),
                        end=end_frame / float(sample_rate),
                    )
                )
        return segments

    def _split_audio_with_torchaudio(self, audio_path: Path, temp_dir: Path) -> list[AudioSegment]:
        waveform, sample_rate = torchaudio.load(str(audio_path))
        sample_rate = int(sample_rate)
        total_frames = int(waveform.shape[-1])
        if sample_rate <= 0 or total_frames <= 0:
            return []

        segments: list[AudioSegment] = []
        windows = self._segment_windows(total_frames / float(sample_rate))
        for index, (start, end) in enumerate(windows):
            start_frame = max(0, min(total_frames, round(start * sample_rate)))
            end_frame = max(start_frame, min(total_frames, round(end * sample_rate)))
            if end_frame <= start_frame:
                continue

            chunk = waveform[:, start_frame:end_frame].contiguous()
            chunk_path = temp_dir / f"chunk_{index:04d}_{start_frame}_{end_frame}.wav"
            torchaudio.save(str(chunk_path), chunk, sample_rate)
            segments.append(
                AudioSegment(
                    path=str(chunk_path),
                    index=index,
                    start=start_frame / float(sample_rate),
                    end=end_frame / float(sample_rate),
                )
            )
        return segments

    def _split_audio_if_needed(self, audio: Any) -> tuple[list[AudioSegment] | None, Path | None]:
        if not self.split_long_audio:
            return None, None

        audio_path = self._local_audio_path(audio)
        if audio_path is None:
            return None, None

        temp_dir = Path(tempfile.gettempdir()) / "comfyui_mega_asr_segments" / uuid.uuid4().hex
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            try:
                segments = self._split_audio_with_soundfile(audio_path, temp_dir)
            except Exception:
                segments = self._split_audio_with_torchaudio(audio_path, temp_dir)

            if len(segments) <= 1:
                shutil.rmtree(temp_dir, ignore_errors=True)
                return None, None
            return segments, temp_dir
        except Exception:
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None, None

    def _record_route(self, use_lora: bool) -> None:
        self.stats["total"] += 1
        if use_lora:
            self.stats["use_lora"] += 1
        else:
            self.stats["use_base"] += 1

    def _route(self, audio: Any) -> tuple[bool, float | None, str]:
        if self.routing_enabled and self.router is not None:
            is_degraded, degraded_prob = self.router.predict(audio)
            return is_degraded, degraded_prob, "router"
        return True, None, "default"

    def _infer_with_adapter_state(self, use_lora: bool, audio: Any, **kwargs: Any) -> Any:
        with self._adapter_lock:
            self._set_lora(use_lora)
            return self.asr.infer(audio, **kwargs)

    @staticmethod
    def _route_label(use_lora: bool | None) -> str:
        if use_lora is True:
            return "mega_lora"
        if use_lora is False:
            return "base"
        return "mixed"

    def _summarize_segment_routes(self, segments: list[dict[str, Any]]) -> dict[str, Any]:
        routes = [segment["route"] for segment in segments]
        lora_values = [route.get("use_lora") for route in routes]
        if lora_values and all(value is True for value in lora_values):
            use_lora: bool | None = True
        elif lora_values and all(value is False for value in lora_values):
            use_lora = False
        else:
            use_lora = None

        probs = [route.get("degraded_prob") for route in routes if route.get("degraded_prob") is not None]
        route_sources = sorted({str(route.get("route_source")) for route in routes if route.get("route_source")})
        if route_sources == ["router"]:
            route_source = "segmented_router"
        elif route_sources:
            route_source = "segmented_" + "+".join(route_sources)
        else:
            route_source = "segmented"

        return {
            "use_lora": use_lora,
            "degraded_prob": max(probs) if probs else None,
            "route_source": route_source,
            "route_label": self._route_label(use_lora),
            "segmented": True,
            "segments": [
                {
                    "index": segment["index"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "duration": segment["duration"],
                    "text": segment["text"],
                    "language": segment["language"],
                    "route": segment["route"],
                }
                for segment in segments
            ],
        }

    def _infer_segmented(
        self,
        original_audio: Any,
        segments: list[AudioSegment],
        *,
        language: str | None,
        return_objects: bool,
        fixed_use_lora: bool | None,
        transcribe_kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        segment_payloads: list[dict[str, Any]] = []
        texts: list[str] = []
        languages: list[str] = []

        for segment in segments:
            if fixed_use_lora is None:
                use_lora, degraded_prob, route_source = self._route(segment.path)
            else:
                use_lora = fixed_use_lora
                degraded_prob = None
                route_source = "forced_lora" if fixed_use_lora else "forced_base"

            raw_result = self._infer_with_adapter_state(
                use_lora,
                segment.path,
                language=language,
                return_objects=return_objects,
                **transcribe_kwargs,
            )
            self._record_route(use_lora)

            text = _extract_result_text(raw_result)
            detected_language = _extract_result_language(raw_result)
            if text:
                texts.append(text)
            if detected_language and detected_language not in languages:
                languages.append(detected_language)

            segment_payloads.append(
                {
                    "index": segment.index,
                    "start": segment.start,
                    "end": segment.end,
                    "duration": segment.duration,
                    "text": text,
                    "language": detected_language,
                    "route": {
                        "use_lora": use_lora,
                        "degraded_prob": degraded_prob,
                        "route_source": route_source,
                    },
                    "raw_result": raw_result,
                }
            )

        joined_text = _merge_segment_texts(texts)
        detected_language = languages[0] if languages else ""
        result_payload = {
            "text": joined_text,
            "language": detected_language,
            "segmented": True,
            "audio_path": str(original_audio),
            "segment_seconds": self.long_audio_segment_seconds,
            "overlap_seconds": self.long_audio_overlap_seconds,
            "segments": segment_payloads,
        }
        result = result_payload if return_objects else joined_text
        return result, self._summarize_segment_routes(segment_payloads)

    def _infer_maybe_segmented(
        self,
        audio: Any,
        *,
        language: str | None,
        return_objects: bool,
        fixed_use_lora: bool | None,
        transcribe_kwargs: dict[str, Any],
    ) -> tuple[Any, dict[str, Any]]:
        segments, temp_dir = self._split_audio_if_needed(audio)
        if segments is not None and temp_dir is not None:
            try:
                return self._infer_segmented(
                    audio,
                    segments,
                    language=language,
                    return_objects=return_objects,
                    fixed_use_lora=fixed_use_lora,
                    transcribe_kwargs=transcribe_kwargs,
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        if fixed_use_lora is None:
            use_lora, degraded_prob, route_source = self._route(audio)
        else:
            use_lora = fixed_use_lora
            degraded_prob = None
            route_source = "forced_lora" if fixed_use_lora else "forced_base"

        result = self._infer_with_adapter_state(
            use_lora,
            audio,
            language=language,
            return_objects=return_objects,
            **transcribe_kwargs,
        )
        self._record_route(use_lora)
        return result, {
            "use_lora": use_lora,
            "degraded_prob": degraded_prob,
            "route_source": route_source,
            "route_label": self._route_label(use_lora),
            "segmented": False,
        }

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
        result, route_payload = self._infer_maybe_segmented(
            audio,
            language=language,
            return_objects=return_objects,
            fixed_use_lora=None,
            transcribe_kwargs=transcribe_kwargs,
        )
        if return_route:
            return {"text": result, **route_payload}
        return result

    def infer_with_lora(
        self,
        audio: Any,
        *,
        language: str | None = None,
        return_objects: bool = False,
        **transcribe_kwargs: Any,
    ) -> Any:
        result, _ = self._infer_maybe_segmented(
            self._unwrap_audio(audio),
            language=language,
            return_objects=return_objects,
            fixed_use_lora=True,
            transcribe_kwargs=transcribe_kwargs,
        )
        return result

    def infer_without_lora(
        self,
        audio: Any,
        *,
        language: str | None = None,
        return_objects: bool = False,
        **transcribe_kwargs: Any,
    ) -> Any:
        result, _ = self._infer_maybe_segmented(
            self._unwrap_audio(audio),
            language=language,
            return_objects=return_objects,
            fixed_use_lora=False,
            transcribe_kwargs=transcribe_kwargs,
        )
        return result

    @torch.no_grad()
    def batch_infer(self, audios: list[Any], **kwargs: Any) -> list[Any]:
        return [self.infer(audio, **kwargs) for audio in audios]
