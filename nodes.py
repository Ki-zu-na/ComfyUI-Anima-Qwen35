import json
import os

import torch

import comfy.lora
import comfy.sd
import comfy.utils
from comfy.weight_adapter.lora import LoRAAdapter
import folder_paths
from comfy.supported_models_base import ClipTarget

from . import rewire
from .text_encoder import AnimaQwen35TEModel, AnimaQwen35Tokenizer, FUSION_LAYERS


folder_paths.add_model_folder_path("llm_adapters", os.path.join(folder_paths.models_dir, "llm_adapters"))
folder_paths.folder_names_and_paths["llm_adapters"][1].add(".safetensors")

CONCAT_FORMAT = "anima-qwen35-conditioner-packed-v1"
MULTILAYER_FORMAT = "anima-qwen35-multilayer-conditioner"


def find_tokenizer_bundle(adapter_path, bundle):
    """Locate the expanded T5 tokenizer published next to the adapter."""
    directory = os.path.dirname(adapter_path)
    for root in (directory, os.path.dirname(directory)):
        candidate = os.path.join(root, bundle, "tokenizer.json")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Place the expanded Anima tokenizer bundle '{bundle}/' from the adapter repository next to the adapter file.")


def check_tokenizer_identity(tokenizer_path, vocab_size, identity):
    """Reject a tokenizer bundle that does not match the adapter's embedding table."""
    manifest = os.path.join(os.path.dirname(tokenizer_path), "anima_vocab.json")
    if not os.path.isfile(manifest):
        return
    with open(manifest, encoding="utf-8") as f:
        vocab = json.load(f)
    if int(vocab["extended_vocab_size"]) != vocab_size:
        raise ValueError(f"Tokenizer bundle has {vocab['extended_vocab_size']} logical IDs but the adapter embedding table has {vocab_size} rows; use the bundle published with this adapter.")
    if identity is not None and vocab.get("tokenizer_identity") != identity:
        raise ValueError("Tokenizer bundle identity does not match the adapter metadata; use the bundle published with this adapter.")


class AnimaQwen35CLIPLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_name": (folder_paths.get_filename_list("text_encoders"),),
                "adapter_name": (folder_paths.get_filename_list("llm_adapters"),),
            },
            "optional": {"device": (["default", "cpu"], {"default": "default"})},
        }

    RETURN_TYPES = ("CLIP",)
    FUNCTION = "load_clip"
    CATEGORY = "loaders/anima"
    DESCRIPTION = "Load Qwen3.5-2B-Base with a dense, multilayer or concat-fusion Anima adapter. Connect to standard CLIP Text Encode nodes."

    def load_clip(self, qwen_name, adapter_name, device="default"):
        qwen_path = folder_paths.get_full_path_or_raise("text_encoders", qwen_name)
        adapter_path = folder_paths.get_full_path_or_raise("llm_adapters", adapter_name)
        tokenizer_path = os.path.dirname(qwen_path)
        if not os.path.isfile(os.path.join(tokenizer_path, "tokenizer.json")):
            raise FileNotFoundError("Place the Qwen3.5-2B-Base tokenizer files alongside its safetensors checkpoint.")

        sd = comfy.utils.state_dict_prefix_replace(
            comfy.utils.load_torch_file(qwen_path, safe_load=True), {"model.language_model.": "model."}, filter_keys=True,
        )
        if not sd:
            raise ValueError("Select the original Qwen3.5-2B-Base safetensors checkpoint from the adapter repository.")
        model_options = {"dtype": comfy.utils.weight_dtype(sd)}
        adapter_sd, metadata = comfy.utils.load_torch_file(adapter_path, safe_load=True, return_metadata=True)
        metadata = metadata or {}
        adapter_prefix = "adapter." if any(key.startswith("fusion.") for key in adapter_sd) else ""
        bundle, identity = "tokenizer", None
        if metadata.get("format") == CONCAT_FORMAT:
            layers = json.loads(metadata["qwen_layers"])
            width, depth = int(metadata["fusion_width"]), int(metadata["fusion_depth"])
            if metadata.get("fusion_variant") != "concat" or depth not in (1, 2) \
                    or int(metadata["fusion_source_dim"]) != 2048 or width * len(layers) != int(metadata["adapter_source_dim"]):
                raise ValueError("Unsupported conditioner configuration; expected a packed v1 concat fusion over 2048-wide Qwen taps.")
            model_options.update(anima_qwen35_fusion="concat", anima_qwen35_layers=layers,
                                 anima_qwen35_fusion_width=width, anima_qwen35_fusion_depth=depth)
            bundle = metadata.get("tokenizer_bundle", "t5_bundle/").strip("/")
            identity = metadata.get("tokenizer_identity")
        elif adapter_prefix:
            if metadata.get("format") != MULTILAYER_FORMAT:
                raise ValueError("Multilayer adapter requires anima-qwen35-multilayer-conditioner metadata.")
            config = json.loads(metadata["config_json"])
            expected = {"mode": "four_layer", "layers": FUSION_LAYERS, "heads": 8, "norm_first": True,
                        "activation": "relu", "layer_norm_before_projection": True, "dropout": 0.0}
            if config.get("format_version") != 1 or any(config["fusion"].get(k) != v for k, v in expected.items()):
                raise ValueError("Unsupported multilayer fusion configuration; expected the eight-layer four_layer conditioner v1.")
            model_options.update(anima_qwen35_fusion="residual", anima_qwen35_layers=FUSION_LAYERS)
        model_options["anima_qwen35_vocab"] = adapter_sd[adapter_prefix + "embed.weight"].shape[0]
        model_options["anima_qwen35_source_dim"] = adapter_sd[adapter_prefix + "blocks.0.cross_attn.k_proj.weight"].shape[1]
        sd.update(comfy.utils.state_dict_prefix_replace(adapter_sd, {adapter_prefix: "llm_adapter."}))

        anima_tokenizer_path = find_tokenizer_bundle(adapter_path, bundle)
        check_tokenizer_identity(anima_tokenizer_path, model_options["anima_qwen35_vocab"], identity)
        if device == "cpu":
            model_options.update(load_device=torch.device("cpu"), offload_device=torch.device("cpu"))
        clip = comfy.sd.CLIP(
            ClipTarget(AnimaQwen35Tokenizer, AnimaQwen35TEModel),
            embedding_directory=folder_paths.get_folder_paths("embeddings"),
            tokenizer_data={"qwen35_tokenizer_path": tokenizer_path, "anima_tokenizer_path": anima_tokenizer_path},
            parameters=comfy.utils.calculate_parameters(sd), state_dict=[sd], model_options=model_options,
        )
        return (clip,)


def load_lora_for_anima(model, clip, lora, strength_model, strength_clip, lora_metadata=None):
    if strength_model == 0 and strength_clip == 0:
        return model, clip
    adapter_prefixes = ("lora_unet_llm_adapter_", "diffusion_model.llm_adapter.", "text_encoders.llm_adapter.")
    adapter_lora = {k: v for k, v in lora.items() if k.startswith(adapter_prefixes)}
    routed = {k: v for k, v in lora.items() if k not in adapter_lora}
    if adapter_lora and strength_clip != 0:
        if not isinstance(clip.cond_stage_model, AnimaQwen35TEModel):
            raise ValueError("Connect CLIP from Anima Qwen3.5 Adapter Loader.")
        state = clip.cond_stage_model.state_dict()
        key_map = {}
        prefixes = {}
        for key in state:
            if key.startswith("llm_adapter.") and key.endswith(".weight"):
                name = key[:-len(".weight")]
                target = "text_encoders." + name
                for source in ("lora_unet_" + name.replace(".", "_"), "diffusion_model." + name, target):
                    key_map[source] = key
                    prefixes[source + "."] = target + "."
                    prefixes[source + "_lora."] = target + "_lora."
        patches = comfy.lora.load_lora(adapter_lora, key_map, log_missing=False)
        loaded_keys = set()
        for key, patch in patches.items():
            if not isinstance(patch, LoRAAdapter):
                raise ValueError(f"Unsupported adapter patch type for {key}; expected LoRA.")
            up, down, _, mid, _, reshape = patch.weights
            expected = tuple(state[key].shape)
            if up.ndim != 2 or down.ndim != 2 or mid is not None or reshape is not None:
                raise ValueError(f"Unsupported adapter LoRA shape for {key}; expected linear LoRA.")
            if (up.shape[0], down.shape[1]) != expected or up.shape[1] != down.shape[0]:
                raise ValueError(f"Adapter LoRA shape mismatch for {key}: up {tuple(up.shape)}, down {tuple(down.shape)}, expected {expected}. Use a Qwen3.5 adapter LoRA, or set strength_clip to 0 for DiT only.")
            loaded_keys.update(patch.loaded_keys)
        unknown = adapter_lora.keys() - loaded_keys
        if unknown:
            raise ValueError(f"Unsupported adapter LoRA key: {sorted(unknown)[0]}")
        routed.update(comfy.utils.state_dict_prefix_replace(adapter_lora, prefixes))
    return comfy.sd.load_lora_for_models(model, clip, routed, strength_model, strength_clip, lora_metadata=lora_metadata)


class AnimaQwen35LoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "lora_name": (folder_paths.get_filename_list("loras"),),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01,
                                             "tooltip": "Strength of the Qwen3.5 adapter LoRA and other CLIP patches. Use 0 for DiT only."}),
            },
            "optional": {"model": ("MODEL",)},
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_lora"
    CATEGORY = "loaders/anima"
    DESCRIPTION = "Apply DiT LoRA to MODEL and route adapter LoRA to the Qwen3.5 CLIP. Supports dense and multilayer adapters."

    def load_lora(self, clip, lora_name, strength_model, strength_clip, model=None):
        if strength_model == 0 and strength_clip == 0:
            return model, clip
        path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora, metadata = comfy.utils.load_torch_file(path, safe_load=True, return_metadata=True)
        return load_lora_for_anima(model, clip, lora, strength_model, strength_clip, lora_metadata=metadata)


class AnimaRewireMode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",), "mode": (["rewired", "dense", "rewired_no_detail", "rewired_u_balanced", "rewired_no_detail_u_balanced", "rewired_u_smooth",
                 "rewired_no_detail_u_smooth"],
                {"default": "rewired",
                "tooltip": "rewired: the trained residual-to-U-Net schedule. dense: dense-bypass, the same DiT weights with all 28 "
                           "blocks at full resolution and the rewire tensors unused (A/B only; the weights were trained rewired). "
                           "A/B ablations of the rewired path: no_detail = conv3 detail branch switched off; u_balanced = "
                           "up-sampling kernel with its 4 output-phase sums equalised per channel (removes a 2x2-token periodic "
                           "colour offset on flat regions); u_smooth = up-sampling kernel replaced by exact bilinear with each "
                           "channel's learned DC gain kept (removes the nearest-neighbour 2x2 copy pattern)."})},
                "optional": {"extra_skip_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "A/B: multiplies only the 4 extra skips (block 0-3 output -> block 24-27 input) added in the ep3 distillation. "
                           "No effect on files without extra skips."}),
                "skip_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                "tooltip": "A/B: multiplies every long U-ViT skip (block i output -> block j input, incl. the extra skips). 1 = as trained."}),
                "sigma_threshold": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                "tooltip": "sigma hybrid (mode=rewired only): rewired while the current sampling sigma > threshold, dense-bypass "
                           "at sigma <= threshold. 0 = off. Sigma is the model-sampling sigma in [0, 1] (use ModelSamplingAuraFlow)."})}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply"
    CATEGORY = "advanced/model/anima"
    DESCRIPTION = ("Optional. Rewired (residual-to-U-Net) Anima files are detected by Load Diffusion Model automatically; this node "
                   "switches to the dense-bypass block loop for comparisons, or to a sigma hybrid (rewired at high sigma, dense-bypass "
                   "below the threshold). No effect on dense files.")

    def apply(self, model, mode, sigma_threshold=0.0, skip_scale=1.0, extra_skip_scale=1.0):
        if not rewire.is_rewired(model):
            return (model,)
        m = model.clone()
        opts = m.model_options.setdefault("transformer_options", {})
        opts[rewire.OPTION_KEY] = mode
        opts[rewire.SIGMA_KEY] = float(sigma_threshold) if mode != "dense" else 0.0
        opts[rewire.SKIP_KEY] = float(skip_scale)
        opts[rewire.SKIP_X_KEY] = float(extra_skip_scale)
        return (m,)


NODE_CLASS_MAPPINGS = {"AnimaQwen35CLIPLoader": AnimaQwen35CLIPLoader, "AnimaQwen35LoraLoader": AnimaQwen35LoraLoader,
                       "AnimaRewireMode": AnimaRewireMode}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimaQwen35CLIPLoader": "Anima Qwen3.5 Adapter Loader", "AnimaQwen35LoraLoader": "Anima Qwen3.5 LoRA Loader",
                              "AnimaRewireMode": "Anima Rewire Mode (A/B, sigma hybrid)"}
