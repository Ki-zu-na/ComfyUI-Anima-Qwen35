import torch
import torch.nn.functional as F
from transformers import Qwen2TokenizerFast, T5TokenizerFast

import comfy.model_management
import comfy.ops
import comfy.quant_ops
import comfy.sd1_clip
import comfy.utils
from comfy.ldm.anima.model import Attention, LLMAdapter, apply_rotary_pos_emb
from comfy.ldm.depth_anything_3.camera import _Attention as FusionAttention
from comfy.ldm.modules.attention import optimized_attention_for_device
from comfy.text_encoders.anima import T5XXLTokenizer
from comfy.text_encoders.llama import BaseLlama, precompute_freqs_cis
from comfy.text_encoders.qwen35 import Qwen35Config, Qwen35Transformer


FUSION_LAYERS = [3, 7, 8, 11, 12, 16, 20, 24]
CONCAT_FUSION_LAYERS = [2, 3, 6, 9, 12, 21]


class AnimaQwen35Tokenizer:
    def __init__(self, embedding_directory=None, tokenizer_data={}):
        self.qwen35_2b = comfy.sd1_clip.SDTokenizer(
            tokenizer_data["qwen35_tokenizer_path"], tokenizer_class=Qwen2TokenizerFast,
            embedding_directory=embedding_directory, embedding_size=2048, embedding_key="qwen35_2b",
            has_start_token=False, has_end_token=False, pad_with_end=False, pad_token=248044,
            pad_to_max_length=False, max_length=99999999, min_length=1,
            tokenizer_data=tokenizer_data, tokenizer_args={"local_files_only": True},
        )
        self.t5xxl = T5XXLTokenizer(tokenizer_data=tokenizer_data)
        self.t5xxl.tokenizer = T5TokenizerFast(tokenizer_file=tokenizer_data["anima_tokenizer_path"], local_files_only=True)
        self.t5xxl.tokenizer.model_max_length = self.t5xxl.max_length

    def tokenize_with_weights(self, text, return_word_ids=False, **kwargs):
        qwen_tokens = self.qwen35_2b.tokenize_with_weights(text, return_word_ids, **kwargs)
        qwen_tokens = [[(token[0], 1.0, *token[2:]) for token in row] for row in qwen_tokens]
        return {"qwen35_2b": qwen_tokens, "t5xxl": self.t5xxl.tokenize_with_weights(text, return_word_ids, **kwargs)}

    def untokenize(self, token_weight_pair):
        return self.t5xxl.untokenize(token_weight_pair)

    def decode(self, token_ids, **kwargs):
        return self.qwen35_2b.decode(token_ids, **kwargs)

    def state_dict(self):
        return {}


class AdapterRotaryEmbedding(torch.nn.Module):
    def __init__(self, head_dim):
        super().__init__()
        self.head_dim = head_dim

    def forward(self, x, position_ids):
        cos, sin, neg_sin = precompute_freqs_cis(self.head_dim, position_ids, 10000, device=x.device)
        half = self.head_dim // 2
        matrix = torch.stack((cos[..., :half], neg_sin, sin, cos[..., half:]), dim=-1)
        matrix = matrix.unflatten(-1, (2, 2)).transpose(1, 2)
        return cos.squeeze(1), torch.cat((sin, -neg_sin), dim=-1).squeeze(1), matrix


class AdapterAttention(Attention):
    def forward(self, x, mask=None, context=None, position_embeddings=None, position_embeddings_context=None):
        self_attention = context is None
        context = x if self_attention else context
        query = self.q_proj(x).unflatten(-1, (self.n_heads, self.head_dim))
        key = self.k_proj(context).unflatten(-1, (self.n_heads, self.head_dim))
        value = self.v_proj(context).unflatten(-1, (self.n_heads, self.head_dim))

        if position_embeddings is not None and not comfy.model_management.in_training:
            with comfy.ops.CastBiasWeightContext(self.q_norm, query, offloadable=True) as (q_scale, _), comfy.ops.CastBiasWeightContext(self.k_norm, key, offloadable=True) as (k_scale, _):
                if self_attention:
                    query, key = comfy.quant_ops.ck.rms_rope_split_half(query, key, position_embeddings[2], q_scale, k_scale, self.q_norm.eps)
                else:
                    query = comfy.quant_ops.ck.rms_rope_split_half1(query, position_embeddings[2], q_scale, self.q_norm.eps)
                    key = comfy.quant_ops.ck.rms_rope_split_half1(key, position_embeddings_context[2], k_scale, self.k_norm.eps)
        else:
            query = self.q_norm(query)
            key = self.k_norm(key)
            if position_embeddings is not None:
                query = apply_rotary_pos_emb(query.transpose(1, 2), *position_embeddings[:2]).transpose(1, 2)
                key = apply_rotary_pos_emb(key.transpose(1, 2), *position_embeddings_context[:2]).transpose(1, 2)

        attention = optimized_attention_for_device(query.device, mask=mask is not None, small_input=True)
        output = attention(query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), self.n_heads, mask=mask, skip_reshape=True, low_precision_attention=False)
        return self.o_proj(output)


class AnimaQwen35Adapter(LLMAdapter):
    def __init__(self, source_dim=2048, vocab_size=59888, device=None, dtype=None, operations=None):
        super().__init__(source_dim=source_dim, target_dim=1024, model_dim=1024, num_layers=6, num_heads=16, use_self_attn=True, device=device, dtype=dtype, operations=operations)
        self.embed = operations.Embedding(vocab_size, 1024, device=device, dtype=dtype)
        self.rotary_emb = AdapterRotaryEmbedding(64)
        for block in self.blocks:
            block.self_attn = AdapterAttention(1024, 1024, 16, 64, device=device, dtype=dtype, operations=operations)
            block.cross_attn = AdapterAttention(1024, source_dim, 16, 64, device=device, dtype=dtype, operations=operations)


class FusionBlock(torch.nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.self_attn = FusionAttention(256, 8, device=device, dtype=dtype, operations=operations)
        self.linear1 = operations.Linear(256, 1024, device=device, dtype=dtype)
        self.linear2 = operations.Linear(1024, 256, device=device, dtype=dtype)
        self.norm1 = operations.LayerNorm(256, eps=1e-5, device=device, dtype=dtype)
        self.norm2 = operations.LayerNorm(256, eps=1e-5, device=device, dtype=dtype)

    def forward(self, x):
        x = x + self.self_attn(self.norm1(x))
        return x + self.linear2(F.relu(self.linear1(self.norm2(x))))


class QwenResidualBranch(torch.nn.Module):
    def __init__(self, device=None, dtype=None, operations=None):
        super().__init__()
        self.inp = operations.Linear(2048, 256, device=device, dtype=dtype)
        self.layer_embedding = torch.nn.Parameter(torch.empty(8, 256, device=device, dtype=dtype))
        self.blocks = torch.nn.ModuleList([FusionBlock(device=device, dtype=dtype, operations=operations) for _ in range(2)])
        self.layer_reduce = operations.Linear(8, 1, bias=False, device=device, dtype=dtype)
        self.out = operations.Linear(256, 2048, device=device, dtype=dtype)

    def load_sd(self, sd, assign=False):
        for i in range(len(self.blocks)):
            prefix = f"blocks.{i}.self_attn."
            for source, target in (("in_proj_weight", "qkv.weight"), ("in_proj_bias", "qkv.bias"),
                                   ("out_proj.weight", "proj.weight"), ("out_proj.bias", "proj.bias")):
                sd[prefix + target] = sd.pop(prefix + source)
        return self.load_state_dict(sd, strict=True, assign=assign)

    def forward(self, last, taps, mask):
        b, length, layers, dim = taps.shape
        h = self.inp(F.layer_norm(taps, (dim,), eps=1e-5)).reshape(b * length, layers, -1)
        h = h + comfy.ops.cast_to_input(self.layer_embedding, h)
        for block in self.blocks:
            h = block(h)
        h = self.layer_reduce(h.transpose(1, 2)).squeeze(-1).reshape(b, length, -1)
        return last + self.out(h) * mask.unsqueeze(-1)


class QwenConcatFusion(torch.nn.Module):
    """Per-tap RMSNorm and projection into its own subspace, concatenated into the adapter source."""

    def __init__(self, layers=len(CONCAT_FUSION_LAYERS), width=512, depth=2, source_dim=2048, device=None, dtype=None, operations=None):
        super().__init__()
        self.norms = torch.nn.ModuleList([operations.RMSNorm(source_dim, eps=1e-6, device=device, dtype=dtype) for _ in range(layers)])
        self.proj = torch.nn.ModuleList([operations.Linear(source_dim, width, device=device, dtype=dtype) for _ in range(layers)])
        self.hidden = torch.nn.ModuleList([
            torch.nn.Sequential(torch.nn.GELU(), operations.Linear(width, width, device=device, dtype=dtype)) for _ in range(layers)
        ]) if depth == 2 else None
        self.out_norm = operations.RMSNorm(width * layers, eps=1e-6, device=device, dtype=dtype)

    def load_sd(self, sd, assign=False):
        return self.load_state_dict(sd, strict=True, assign=assign)

    def forward(self, taps, mask):
        parts = []
        for index, (norm, proj) in enumerate(zip(self.norms, self.proj)):
            value = proj(norm(taps[:, :, index]))
            if self.hidden is not None:
                value = value + self.hidden[index](value)
            parts.append(value)
        return self.out_norm(torch.cat(parts, dim=-1)) * mask.unsqueeze(-1)


class Qwen35BaseTextModel(BaseLlama, torch.nn.Module):
    def __init__(self, config_dict, dtype, device, operations):
        super().__init__()
        config = Qwen35Config(**config_dict)
        self.num_layers = config.num_hidden_layers
        self.model = Qwen35Transformer(config, device=device, dtype=dtype, ops=operations)
        self.dtype = dtype


def fusion_layers(model_options):
    variant = model_options.get("anima_qwen35_fusion", None)
    if variant is None:
        return None
    return model_options.get("anima_qwen35_layers", CONCAT_FUSION_LAYERS if variant == "concat" else FUSION_LAYERS)


class Qwen35BaseClipModel(comfy.sd1_clip.SDClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        layers = fusion_layers(model_options)
        super().__init__(
            device=device, dtype=dtype, layer=layers if layers is not None else "last", textmodel_json_config={},
            special_tokens={"pad": 248044}, model_class=Qwen35BaseTextModel,
            enable_attention_masks=True, return_attention_masks=True,
            layer_norm_hidden_state=layers is None, model_options=model_options,
        )


class AnimaQwen35TEModel(comfy.sd1_clip.SD1ClipModel):
    def __init__(self, device="cpu", dtype=None, model_options={}):
        super().__init__(device=device, dtype=dtype, name="qwen35_2b", clip_model=Qwen35BaseClipModel, model_options=model_options)
        self.llm_adapter = AnimaQwen35Adapter(
            source_dim=model_options.get("anima_qwen35_source_dim", 2048),
            vocab_size=model_options.get("anima_qwen35_vocab", 59888),
            device=device, dtype=dtype, operations=comfy.ops.manual_cast,
        )
        variant = model_options.get("anima_qwen35_fusion", None)
        if variant == "concat":
            self.fusion = QwenConcatFusion(len(fusion_layers(model_options)), model_options.get("anima_qwen35_fusion_width", 512),
                                           model_options.get("anima_qwen35_fusion_depth", 2),
                                           device=device, dtype=dtype, operations=comfy.ops.manual_cast)
        elif variant == "residual":
            self.fusion = QwenResidualBranch(device=device, dtype=dtype, operations=comfy.ops.manual_cast)
        else:
            self.fusion = None
        self.can_assign_sd = False

    def load_sd(self, sd):
        adapter_sd = comfy.utils.state_dict_prefix_replace(sd, {"llm_adapter.": ""}, filter_keys=True)
        self.llm_adapter.load_state_dict(adapter_sd, strict=True, assign=self.can_assign_sd)
        if self.fusion is not None:
            self.fusion.load_sd(comfy.utils.state_dict_prefix_replace(sd, {"fusion.": ""}, filter_keys=True), assign=self.can_assign_sd)
        return self.qwen35_2b.transformer.load_state_dict(sd, strict=True, assign=self.can_assign_sd)

    def encode_token_weights(self, token_weight_pairs):
        qwen_tokens = [[token[0] for token in row] for row in token_weight_pairs["qwen35_2b"]]
        source, pooled, extra = self.qwen35_2b.encode(qwen_tokens)
        source_mask = extra["attention_mask"]
        if isinstance(self.fusion, QwenConcatFusion):
            source = self.fusion(source.movedim(1, 2), source_mask)
        elif self.fusion is not None:
            source = self.fusion(source[:, -1] * source_mask.unsqueeze(-1), source.movedim(1, 2), source_mask)
        else:
            source = source * source_mask.unsqueeze(-1)
        t5_tokens = [token for row in token_weight_pairs["t5xxl"] for token in row]
        target_ids = torch.tensor([[token[0] for token in t5_tokens]], device=source.device, dtype=torch.long)
        weights = torch.tensor([[token[1] for token in t5_tokens]], device=source.device, dtype=source.dtype).unsqueeze(-1)
        target_mask = target_ids != 0
        output = self.llm_adapter(source, target_ids, target_attention_mask=target_mask, source_attention_mask=source_mask)
        output = output * weights * target_mask.unsqueeze(-1)
        output = F.pad(output, (0, 0, 0, max(0, 512 - output.shape[1])))
        return output.to(comfy.model_management.intermediate_device()), pooled
