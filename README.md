# ComfyUI-Anima-Qwen35

ComfyUI nodes for using **Qwen3.5-2B** as the text encoder of [Anima](https://huggingface.co/circlestone-labs/Anima) through a conditioner (adapter). Prompts can be written in English, Chinese or Japanese.

Weights: [SumomoLee/Anima-Qwen3.5-Conditioner](https://huggingface.co/SumomoLee/Anima-Qwen3.5-Conditioner)

> This is a personal project trained to personal preferences. The training data and choices are not neutral, so the results may be biased in style and content.

## Install

Clone into `ComfyUI/custom_nodes` and restart ComfyUI. No extra dependencies; it needs a ComfyUI version that includes Qwen3.5 support (`comfy/text_encoders/qwen35.py`).

## Files

```text
ComfyUI/models/text_encoders/qwen3.5-2b-base/        # Qwen3.5-2B-Base (included in the HF repo under qwen3.5-2b-base/)
ComfyUI/models/llm_adapters/anima-qwen35/
  qwen35-conditioner-v3-distill-ep02-bf16.safetensors
  t5_bundle/{tokenizer.json, tokenizer_config.json, anima_vocab.json}
ComfyUI/models/loras/anima-lora-ep14.safetensors     # optional
```

The tokenizer bundle must stay next to the conditioner file; the node checks that it matches the conditioner's embedding table.

## Usage

1. **Anima Qwen3.5 Adapter Loader** (`loaders/anima`): select the Qwen3.5 checkpoint and the conditioner. Connect its `CLIP` output to the usual CLIP Text Encode nodes.
2. Optional: the conditioner works with the base Anima model alone. To add the LoRA trained with it, use **Anima Qwen3.5 LoRA Loader** instead of the normal LoRA loader (connect Anima `MODEL` and the `CLIP` from step 1).
3. Anima model loader, sampler and VAE are used as normal. See `examples/anima_qwen35.json`.

Set `device` to `cpu` on the adapter loader if VRAM is tight.

The node also loads experimental "rewire" Anima DiT files automatically when present; normal Anima checkpoints are unaffected.

## License

Code: GPL-3.0 (same as ComfyUI). The weights have their own licenses: the conditioner and LoRA follow the CircleStone Labs Non-Commercial License (derivatives of Anima); Qwen3.5 is Apache-2.0.
