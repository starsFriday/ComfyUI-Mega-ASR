# ComfyUI-Mega-ASR

ComfyUI nodes for local [Mega-ASR](https://huggingface.co/zhifeixie/Mega-ASR) speech-to-text transcription.

![ComfyUI-Mega-ASR workflow screenshot](https://github.com/user-attachments/assets/411fbce4-d060-4067-a155-fd64e0e6f040)

Example workflow: [`workflow/Mega-ASR.json`](workflow/Mega-ASR.json)

Mega-ASR is a robust ASR system built on Qwen3-ASR-1.7B. It combines the base Qwen3-ASR model, Mega-ASR LoRA adaptation weights, and an audio-quality router. The router decides whether each input should run through the robust Mega-ASR path or the base recognition path.

This repository includes the ComfyUI node code and a built-in Mega-ASR inference adapter based on the official implementation from [xzf-thu/Mega-ASR.git](https://github.com/xzf-thu/Mega-ASR.git). You do not need to clone the official Mega-ASR source separately.

## Features

- Local transcription from ComfyUI `AUDIO` inputs.
- Local transcription from a local audio file path.
- Uses the standard ComfyUI model folder: `ComfyUI/models/Mega-ASR`.
- Built-in Mega-ASR wrapper, audio-quality router, and LoRA adapter switching.
- Supports router mode, always-on Mega-ASR LoRA mode, and base-only mode.
- Automatically splits long local audio in the backend, transcribes each segment, and merges the text in order.
- Returns transcription text, raw JSON metadata, selected route, router probability, detected language, segment metadata, and temporary audio path.
- Includes an environment-status node for checking model files and dependencies.

## Supported Languages

Mega-ASR uses Qwen3-ASR-1.7B as the base ASR model. According to the official Qwen3-ASR documentation, it supports language identification and speech recognition for **30 languages and 22 Chinese dialects/accents**.

The node's `language` input can be left as `auto` for automatic language detection. You can also set a language hint manually when the input language is known.

Supported languages:

- Chinese `zh`
- English `en`
- Cantonese `yue`
- Arabic `ar`
- German `de`
- French `fr`
- Spanish `es`
- Portuguese `pt`
- Indonesian `id`
- Italian `it`
- Korean `ko`
- Russian `ru`
- Thai `th`
- Vietnamese `vi`
- Japanese `ja`
- Turkish `tr`
- Hindi `hi`
- Malay `ms`
- Dutch `nl`
- Swedish `sv`
- Danish `da`
- Finnish `fi`
- Polish `pl`
- Czech `cs`
- Filipino `fil`
- Persian `fa`
- Greek `el`
- Hungarian `hu`
- Macedonian `mk`
- Romanian `ro`

Supported Chinese dialects/accents:

- Anhui
- Dongbei
- Fujian
- Gansu
- Guizhou
- Hebei
- Henan
- Hubei
- Hunan
- Jiangxi
- Ningxia
- Shandong
- Shaanxi
- Shanxi
- Sichuan
- Tianjin
- Yunnan
- Zhejiang
- Cantonese (Hong Kong accent)
- Cantonese (Guangdong accent)
- Wu language
- Minnan language

Sources: [Qwen/Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B/blob/main/README.md), [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR).

## Model Layout

Download `zhifeixie/Mega-ASR` from Hugging Face and place its contents here:

```text
ComfyUI/models/Mega-ASR/
  Qwen3-ASR-1.7B/
    config.json
    model.safetensors.index.json
    model-00001-of-00002.safetensors
    model-00002-of-00002.safetensors
  mega-asr-merged/
    adapter_config.json
    adapter_model.safetensors
    mega_lora_blocks.json
  audio_quality_router/
    best_acc_model.safetensors
```

The node only uses `models/Mega-ASR` for model weights. It does not search absolute paths such as `/models/Mega-ASR`.

## Quick Start

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/starsFriday/ComfyUI-Mega-ASR.git
cd ..
python -m pip install -r custom_nodes/ComfyUI-Mega-ASR/requirements.txt
python -m pip install -U huggingface_hub
huggingface-cli download zhifeixie/Mega-ASR --local-dir models/Mega-ASR
```

Restart ComfyUI, then load [`workflow/Mega-ASR.json`](workflow/Mega-ASR.json) or create the nodes manually.

## Installation

Clone this repository into `custom_nodes`:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/starsFriday/ComfyUI-Mega-ASR.git
```

Install the minimal runtime dependencies in the same Python environment that runs ComfyUI:

```bash
cd ComfyUI
python -m pip install -r custom_nodes/ComfyUI-Mega-ASR/requirements.txt
```

The dependency list is intentionally small: `qwen-asr`, `safetensors`, `soundfile`, `scipy`, and `torchaudio`.

Download the Mega-ASR model from Hugging Face:

```bash
cd ComfyUI
python -m pip install -U huggingface_hub
huggingface-cli download zhifeixie/Mega-ASR --local-dir models/Mega-ASR
```

If you already downloaded the model manually, copy or move the Hugging Face folder contents into `ComfyUI/models/Mega-ASR` so the layout matches the section above.

Restart ComfyUI after installing dependencies and model files.

## Long Audio

Long audio is handled automatically inside the backend inference adapter. The ComfyUI nodes still make one transcription call; the backend checks the local audio duration, splits files longer than about 30 seconds into temporary wav segments, runs ASR on each segment, and joins the segment text in chronological order.

When `routing_mode` is `auto_router`, each segment is routed independently, so a long file can contain both `base` and `mega_lora` segments. In that case the route output can be `mixed`, and `raw_response` includes per-segment start time, end time, route, detected language, text, and raw model result.

No extra ComfyUI controls are required for this behavior.

## Nodes

### Mega-ASR Loader

Loads the local Mega-ASR stack from `models/Mega-ASR`.

Inputs:

- `routing_mode`
  - `auto_router`: use Mega-ASR's audio-quality router.
  - `force_mega_lora`: always use the Mega-ASR LoRA path.
  - `force_base`: bypass the LoRA and use the Qwen3-ASR base path.
- `device_map`: `auto`, `cuda:0`, `cuda:1`, or `cpu`.
- `dtype`: `auto`, `bfloat16`, `float16`, or `float32`.
- `attn_implementation`: `default`, `flash_attention_2`, `sdpa`, or `eager`.
- `quality_threshold`: router threshold. The official default is `0.5`.
- `max_new_tokens`: decoding limit. This node defaults to `1024` for longer transcripts, such as songs or long-form audio.
- `max_inference_batch_size`: passed to Qwen3-ASR.
- `low_cpu_mem_usage`: passed to model loading.
- `keep_delta_on_gpu`: cache full LoRA deltas on GPU after first switch. Leave this off unless you have enough VRAM.
- `reload_model`: clears the cached loader entry and reloads.

Outputs:

- `model`: model object for the transcription nodes.
- `model_info`: JSON summary of paths and loader settings.

### Mega-ASR Transcribe Audio

Transcribes a ComfyUI `AUDIO` input.

Inputs:

- `model`: output from `Mega-ASR Loader`.
- `audio`: ComfyUI `AUDIO`.
- `language`: `auto` or a language hint.
- `force_mono`: converts multi-channel audio to mono before transcription.
- `transcribe_kwargs_json`: advanced JSON object passed to Qwen3-ASR `transcribe`. Keep this as `{}` for normal use.

Outputs:

- `text`: transcription text.
- `raw_response`: JSON payload with route, segment metadata, and raw model result.
- `route`: `mega_lora`, `base`, `mixed`, or the forced mode label.
- `quality_prob`: router degraded-audio probability, or `-1` when not routed.
- `detected_language`: language field if returned by Qwen3-ASR.
- `temp_audio_path`: temporary wav created from the ComfyUI audio input.

### Mega-ASR Transcribe File

Transcribes a local audio file path.

Inputs and outputs are the same as `Mega-ASR Transcribe Audio`, except the audio input is `audio_path` and no temporary wav is created.

### Mega-ASR Environment Status

Returns JSON showing:

- Expected model root.
- Missing model files or folders.
- Whether this node is using the built-in implementation.
- Whether key dependencies such as `qwen_asr`, `safetensors`, `soundfile`, `scipy`, and `torchaudio` are installed.

Use this node first if the loader does not appear or model loading fails.

## Example Workflow

A ready-to-use ComfyUI workflow is included at [`workflow/Mega-ASR.json`](workflow/Mega-ASR.json). Drag the JSON file into ComfyUI or load it from the workflow menu.

The example connects:

1. `Mega-ASR Loader`
2. `Load Audio`
3. `Mega-ASR Transcribe Audio`
4. Text display/output nodes

The loader is set to `auto_router` by default. For degraded recordings, `auto_router` should route to `mega_lora`; for clean speech, it may route to `base`.

## Recommended Manual Setup

1. Add `Mega-ASR Loader`.
2. Set `routing_mode` to `auto_router`.
3. Connect `Mega-ASR Loader.model` to `Mega-ASR Transcribe Audio.model`.
4. Connect an `AUDIO` source to `Mega-ASR Transcribe Audio.audio`.
5. Run the workflow and read `text`.

## Troubleshooting

### Model files were not found

Check that the directory is exactly:

```text
ComfyUI/models/Mega-ASR/
```

and that it contains `Qwen3-ASR-1.7B`, `mega-asr-merged`, and `audio_quality_router/best_acc_model.safetensors`.

If the loader reports a missing Qwen3-ASR weight shard, re-download the Hugging Face model directory. The safetensors index must match all referenced shard files.

### `qwen_asr` is missing

Install the requirements in the ComfyUI Python environment:

```bash
python -m pip install -r custom_nodes/ComfyUI-Mega-ASR/requirements.txt
```

### CUDA memory issues

Try these loader settings:

- `device_map`: `cpu` for CPU fallback.
- `dtype`: `float16` or `bfloat16` on GPU.
- `max_new_tokens`: lower values for shorter expected transcripts.

### Router or LoRA load errors

Run `Mega-ASR Environment Status` and check that:

- `model_files_ok` is `true`.
- `qwen_asr_installed` is `true`.
- `safetensors_installed` is `true`.

## Credits

- Mega-ASR model and official inference code: [xzf-thu/Mega-ASR.git](https://github.com/xzf-thu/Mega-ASR.git)
- Hugging Face model release: [zhifeixie/Mega-ASR](https://huggingface.co/zhifeixie/Mega-ASR)
- Backbone model: Qwen3-ASR-1.7B

## Citation

If you use Mega-ASR, cite the original project:

```bibtex
@misc{xie2026megaasrinthewild2speechrecognition,
  title={Mega-ASR: Towards In-the-wild^2 Speech Recognition via Scaling up Real-world Acoustic Simulation},
  author={Zhifei Xie and Kaiyu Pang and Haobin Zhang and Deheng Ye and Xiaobin Hu and Shuicheng Yan and Chunyan Miao},
  year={2026},
  eprint={2605.19833},
  archivePrefix={arXiv},
  primaryClass={cs.SD},
  url={https://arxiv.org/abs/2605.19833},
}
```

## License

This ComfyUI node wrapper is released under Apache-2.0. Portions of the built-in inference adapter are adapted from the official Mega-ASR project. Check upstream repositories for the full model and code license terms.
