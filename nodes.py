from __future__ import annotations

import importlib.util
import json
import sys
import threading
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import folder_paths

PLUGIN_ROOT = Path(__file__).resolve().parent
if str(PLUGIN_ROOT) not in sys.path:
    sys.path.insert(0, str(PLUGIN_ROOT))

try:
    from .mega_asr_impl import MegaASR
except ImportError:
    from mega_asr_impl import MegaASR

MEGA_ASR_SUBDIR = "Mega-ASR"
QWEN_MODEL_SUBDIR = "Qwen3-ASR-1.7B"
LORA_SUBDIR = "mega-asr-merged"
ROUTER_SUBDIR = "audio_quality_router"

ROUTING_MODES = ["auto_router", "force_mega_lora", "force_base"]
DTYPE_CHOICES = ["auto", "bfloat16", "float16", "float32"]
ATTN_CHOICES = ["default", "flash_attention_2", "sdpa", "eager"]
LANGUAGE_CHOICES = [
    "auto",
    "Chinese",
    "English",
    "Cantonese",
    "Japanese",
    "Korean",
    "German",
    "French",
    "Russian",
    "Spanish",
    "Portuguese",
    "Italian",
    "Vietnamese",
    "Arabic",
    "Hindi",
]

_MODEL_CACHE: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


def _model_root() -> Path:
    return Path(folder_paths.models_dir) / MEGA_ASR_SUBDIR


def _model_paths() -> Dict[str, Path]:
    root = _model_root()
    return {
        "root": root,
        "qwen": root / QWEN_MODEL_SUBDIR,
        "lora": root / LORA_SUBDIR,
        "router": root / ROUTER_SUBDIR,
        "router_weights": root / ROUTER_SUBDIR / "best_acc_model.safetensors",
    }


def _missing_model_items(paths: Dict[str, Path]) -> list[str]:
    checks = [
        ("model root", paths["root"]),
        ("Qwen3-ASR base model", paths["qwen"]),
        ("Qwen3-ASR config", paths["qwen"] / "config.json"),
        ("Qwen3-ASR safetensors index", paths["qwen"] / "model.safetensors.index.json"),
        ("Mega-ASR LoRA", paths["lora"]),
        ("Mega-ASR LoRA config", paths["lora"] / "adapter_config.json"),
        ("Mega-ASR LoRA weights", paths["lora"] / "adapter_model.safetensors"),
        ("audio quality router", paths["router"]),
        ("audio quality router weights", paths["router_weights"]),
    ]
    missing = [f"{label}: {path}" for label, path in checks if not path.exists()]

    index_path = paths["qwen"] / "model.safetensors.index.json"
    if index_path.exists():
        try:
            index_data = json.loads(index_path.read_text())
            shard_names = sorted(set(index_data.get("weight_map", {}).values()))
            for shard_name in shard_names:
                shard_path = paths["qwen"] / shard_name
                if not shard_path.exists():
                    missing.append(f"Qwen3-ASR weight shard: {shard_path}")
        except Exception as exc:
            missing.append(f"Qwen3-ASR safetensors index is not readable: {index_path} ({exc})")

    return missing


def _validate_model_layout() -> Dict[str, Path]:
    paths = _model_paths()
    missing = _missing_model_items(paths)
    if missing:
        expected = (
            f"{paths['root']}/\n"
            f"  {QWEN_MODEL_SUBDIR}/\n"
            f"  {LORA_SUBDIR}/\n"
            f"  {ROUTER_SUBDIR}/best_acc_model.safetensors"
        )
        raise FileNotFoundError(
            "Mega-ASR model files were not found under ComfyUI models/Mega-ASR.\n"
            f"Missing:\n- " + "\n- ".join(missing) + "\n\n"
            f"Expected layout:\n{expected}"
        )
    return paths


def _dtype_from_name(name: str):
    if name == "auto":
        return None
    import torch

    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        try:
            return _jsonable(value.model_dump())
        except Exception:
            pass
    if hasattr(value, "dict"):
        try:
            return _jsonable(value.dict())
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return _jsonable(
            {
                k: v
                for k, v in vars(value).items()
                if not k.startswith("_") and not callable(v)
            }
        )
    return str(value)


def _iter_result_objects(result: Any) -> Iterable[Any]:
    if isinstance(result, dict) and "text" in result:
        yield from _iter_result_objects(result["text"])
    elif isinstance(result, (list, tuple)):
        yield from result
    else:
        yield result


def _extract_text(result: Any) -> str:
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


def _extract_language(result: Any) -> str:
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


def _parse_extra_kwargs(raw_json: str) -> Dict[str, Any]:
    raw_json = (raw_json or "").strip()
    if not raw_json:
        return {}
    value = json.loads(raw_json)
    if not isinstance(value, dict):
        raise ValueError("transcribe_kwargs_json must be a JSON object, for example: {}")
    return value


def _audio_to_wav(audio: Dict[str, Any], force_mono: bool) -> str:
    import torch
    import torchaudio

    if "waveform" not in audio or "sample_rate" not in audio:
        raise ValueError("Invalid ComfyUI AUDIO input. Expected keys: waveform, sample_rate.")

    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if waveform.ndim == 3:
        waveform = waveform[0]
    elif waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if force_mono and waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    waveform = waveform.detach().cpu().float()
    temp_dir = Path(folder_paths.get_temp_directory()) / "mega_asr"
    temp_dir.mkdir(parents=True, exist_ok=True)
    wav_path = temp_dir / f"mega_asr_{uuid.uuid4().hex}.wav"
    torchaudio.save(str(wav_path), waveform, sample_rate)
    return str(wav_path)


def _load_model(
    routing_mode: str,
    device_map: str,
    dtype: str,
    attn_implementation: str,
    quality_threshold: float,
    max_new_tokens: int,
    max_inference_batch_size: int,
    low_cpu_mem_usage: bool,
    keep_delta_on_gpu: bool,
    reload_model: bool,
) -> Dict[str, Any]:
    paths = _validate_model_layout()
    routing_enabled = routing_mode == "auto_router"
    cache_key = (
        str(paths["root"].resolve()),
        routing_enabled,
        device_map.strip() or "auto",
        dtype,
        attn_implementation,
        float(quality_threshold),
        int(max_new_tokens),
        int(max_inference_batch_size),
        bool(low_cpu_mem_usage),
        bool(keep_delta_on_gpu),
    )

    with _CACHE_LOCK:
        if reload_model:
            _MODEL_CACHE.pop(cache_key, None)
        if cache_key in _MODEL_CACHE:
            model_info = dict(_MODEL_CACHE[cache_key])
            model_info["routing_mode"] = routing_mode
            return model_info

    model_kwargs: Dict[str, Any] = {"low_cpu_mem_usage": bool(low_cpu_mem_usage)}
    actual_device = device_map.strip() or "auto"
    device_arg = None if actual_device == "auto" else actual_device
    torch_dtype = _dtype_from_name(dtype)
    if torch_dtype is not None:
        model_kwargs["dtype"] = torch_dtype
    if attn_implementation != "default":
        model_kwargs["attn_implementation"] = attn_implementation

    try:
        model = MegaASR(
            model_path=str(paths["qwen"]),
            lora_dir=str(paths["lora"]),
            router_checkpoint=str(paths["router_weights"]),
            routing_enabled=routing_enabled,
            quality_threshold=float(quality_threshold),
            device_map=device_arg,
            max_inference_batch_size=int(max_inference_batch_size),
            max_new_tokens=int(max_new_tokens),
            keep_delta_on_gpu=bool(keep_delta_on_gpu),
            **model_kwargs,
        )
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Missing Python dependency for Mega-ASR: {exc.name}. "
            "Install this node's requirements in the same Python environment used by ComfyUI."
        ) from exc

    model_info = {
        "model": model,
        "routing_mode": routing_mode,
        "routing_enabled": routing_enabled,
        "model_root": str(paths["root"]),
        "base_model": str(paths["qwen"]),
        "lora_model": str(paths["lora"]),
        "router_model": str(paths["router_weights"]),
        "quality_threshold": float(quality_threshold),
        "device_map": actual_device,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "max_new_tokens": int(max_new_tokens),
        "max_inference_batch_size": int(max_inference_batch_size),
        "keep_delta_on_gpu": bool(keep_delta_on_gpu),
        "implementation": "built_in",
    }
    with _CACHE_LOCK:
        _MODEL_CACHE[cache_key] = dict(model_info)
    return model_info


def _run_transcription(
    model_info: Dict[str, Any],
    audio_path: str,
    language: str,
    transcribe_kwargs_json: str,
) -> Tuple[str, str, str, float, str]:
    path = Path(audio_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Audio path not found: {path}")

    mega_asr = model_info["model"]
    language_value: Optional[str] = None if language == "auto" else language
    kwargs = _parse_extra_kwargs(transcribe_kwargs_json)
    if language_value is not None:
        kwargs["language"] = language_value
    kwargs["return_objects"] = True

    routing_mode = model_info.get("routing_mode", "auto_router")
    route_payload: Dict[str, Any]

    if routing_mode == "force_base":
        raw_result = mega_asr.infer_without_lora(str(path), **kwargs)
        route_payload = {"mode": routing_mode, "use_lora": False, "quality_prob": None}
    elif routing_mode == "force_mega_lora":
        raw_result = mega_asr.infer_with_lora(str(path), **kwargs)
        route_payload = {"mode": routing_mode, "use_lora": True, "quality_prob": None}
    else:
        routed_result = mega_asr.infer(str(path), return_route=True, **kwargs)
        raw_result = routed_result.get("text") if isinstance(routed_result, dict) else routed_result
        route_payload = {
            "mode": routing_mode,
            "use_lora": routed_result.get("use_lora") if isinstance(routed_result, dict) else None,
            "quality_prob": (
                routed_result.get("degraded_prob") if isinstance(routed_result, dict) else None
            ),
            "route_source": (
                routed_result.get("route_source") if isinstance(routed_result, dict) else None
            ),
        }
        if isinstance(routed_result, dict):
            for key in (
                "route_label",
                "segmented",
                "segments",
                "segment_seconds",
                "overlap_seconds",
            ):
                if key in routed_result:
                    route_payload[key] = routed_result[key]

    text = _extract_text(raw_result)
    detected_language = _extract_language(raw_result)
    quality_prob = route_payload.get("quality_prob")
    payload = {
        "text": text,
        "detected_language": detected_language,
        "audio_path": str(path),
        "route": route_payload,
        "raw_result": _jsonable(raw_result),
    }
    raw_json = json.dumps(payload, ensure_ascii=False, indent=2)
    route_label = str(route_payload.get("route_label") or routing_mode)
    if "route_label" not in route_payload:
        if route_payload.get("use_lora") is True:
            route_label = "mega_lora"
        elif route_payload.get("use_lora") is False:
            route_label = "base"
    return (
        text,
        raw_json,
        route_label,
        float(quality_prob) if quality_prob is not None else -1.0,
        detected_language,
    )


def _dependency_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _status_report() -> str:
    paths = _model_paths()
    missing = _missing_model_items(paths)
    report = {
        "model_root": str(paths["root"]),
        "model_root_rule": "ComfyUI/models/Mega-ASR",
        "model_files_ok": not missing,
        "missing_model_items": missing,
        "implementation": "built_in",
        "requires_official_source_clone": False,
        "qwen_asr_installed": _dependency_available("qwen_asr"),
        "safetensors_installed": _dependency_available("safetensors"),
        "soundfile_installed": _dependency_available("soundfile"),
        "scipy_installed": _dependency_available("scipy"),
        "torchaudio_installed": _dependency_available("torchaudio"),
    }
    return json.dumps(report, ensure_ascii=False, indent=2)


class MegaASRLoader:
    RETURN_TYPES = ("MEGA_ASR_MODEL", "STRING")
    RETURN_NAMES = ("model", "model_info")
    FUNCTION = "load"
    CATEGORY = "Audio/Mega-ASR"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "routing_mode": (ROUTING_MODES, {"default": "auto_router"}),
                "device_map": (
                    "STRING",
                    {
                        "default": "auto",
                        "multiline": False,
                        "placeholder": "auto | cuda:0 | cuda:1 | cpu",
                    },
                ),
                "dtype": (DTYPE_CHOICES, {"default": "auto"}),
                "attn_implementation": (ATTN_CHOICES, {"default": "default"}),
                "quality_threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 1024, "min": 32, "max": 8192, "step": 32},
                ),
                "max_inference_batch_size": (
                    "INT",
                    {"default": 32, "min": 1, "max": 256, "step": 1},
                ),
            },
            "optional": {
                "low_cpu_mem_usage": ("BOOLEAN", {"default": True}),
                "keep_delta_on_gpu": ("BOOLEAN", {"default": False}),
                "reload_model": ("BOOLEAN", {"default": False}),
            },
        }

    def load(
        self,
        routing_mode: str,
        device_map: str,
        dtype: str,
        attn_implementation: str,
        quality_threshold: float,
        max_new_tokens: int,
        max_inference_batch_size: int,
        low_cpu_mem_usage: bool = True,
        keep_delta_on_gpu: bool = True,
        reload_model: bool = False,
    ):
        model_info = _load_model(
            routing_mode=routing_mode,
            device_map=device_map,
            dtype=dtype,
            attn_implementation=attn_implementation,
            quality_threshold=quality_threshold,
            max_new_tokens=max_new_tokens,
            max_inference_batch_size=max_inference_batch_size,
            low_cpu_mem_usage=low_cpu_mem_usage,
            keep_delta_on_gpu=keep_delta_on_gpu,
            reload_model=reload_model,
        )
        visible_info = {k: v for k, v in model_info.items() if k != "model"}
        return (model_info, json.dumps(visible_info, ensure_ascii=False, indent=2))


class MegaASRTranscribeAudio:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "text",
        "raw_response",
        "route",
        "quality_prob",
        "detected_language",
        "temp_audio_path",
    )
    FUNCTION = "transcribe"
    CATEGORY = "Audio/Mega-ASR"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MEGA_ASR_MODEL",),
                "audio": ("AUDIO",),
                "language": (LANGUAGE_CHOICES, {"default": "auto"}),
            },
            "optional": {
                "force_mono": ("BOOLEAN", {"default": True}),
                "transcribe_kwargs_json": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
            },
        }

    def transcribe(
        self,
        model: Dict[str, Any],
        audio: Dict[str, Any],
        language: str,
        force_mono: bool = True,
        transcribe_kwargs_json: str = "{}",
    ):
        audio_path = _audio_to_wav(audio, force_mono=force_mono)
        text, raw_json, route, prob, detected_language = _run_transcription(
            model, audio_path, language, transcribe_kwargs_json
        )
        return (text, raw_json, route, prob, detected_language, audio_path)


class MegaASRTranscribeFile:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "FLOAT", "STRING")
    RETURN_NAMES = ("text", "raw_response", "route", "quality_prob", "detected_language")
    FUNCTION = "transcribe"
    CATEGORY = "Audio/Mega-ASR"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MEGA_ASR_MODEL",),
                "audio_path": (
                    "STRING",
                    {"default": "", "multiline": False, "placeholder": "/path/to/audio.wav"},
                ),
                "language": (LANGUAGE_CHOICES, {"default": "auto"}),
            },
            "optional": {
                "transcribe_kwargs_json": (
                    "STRING",
                    {"default": "{}", "multiline": True},
                ),
            },
        }

    def transcribe(
        self,
        model: Dict[str, Any],
        audio_path: str,
        language: str,
        transcribe_kwargs_json: str = "{}",
    ):
        return _run_transcription(model, audio_path, language, transcribe_kwargs_json)


class MegaASREnvironmentStatus:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_json",)
    FUNCTION = "check"
    CATEGORY = "Audio/Mega-ASR"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    def check(self):
        return (_status_report(),)


NODE_CLASS_MAPPINGS = {
    "MegaASRLoader": MegaASRLoader,
    "MegaASRTranscribeAudio": MegaASRTranscribeAudio,
    "MegaASRTranscribeFile": MegaASRTranscribeFile,
    "MegaASREnvironmentStatus": MegaASREnvironmentStatus,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MegaASRLoader": "Mega-ASR Loader",
    "MegaASRTranscribeAudio": "Mega-ASR Transcribe Audio",
    "MegaASRTranscribeFile": "Mega-ASR Transcribe File",
    "MegaASREnvironmentStatus": "Mega-ASR Environment Status",
}
