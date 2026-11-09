"""Transparent loading of residual-to-U-Net ("rewire") Anima DiT files.

Format ``anima-rewire-unet-v1``: a normal Anima DiT state dict plus ``<prefix>rewire.*`` tensors and the
safetensors metadata ``anima_rewire_format`` / ``anima_rewire`` (JSON config). Port of
``te_experiments/anima_rewire.py`` (after lodestone-rock/residual-to-unet):

    x = merge(cat[U(mid(P h) - P h) + d, h])     (middle span on a 2x2-pooled token grid, "subtract what went in")

Supported layouts (selected from the embedded config, as the reference ``load_rewire`` does):
  * F-symmetric span [F, N-F) (old ``rewire-unet-f7-t2i``: F=7, proj P/U with the pixel-shuffle ``up_detail``
    branch, learned merge, 7 U-ViT skips), optional inner 4x4 level, several disjoint spans;
  * asymmetric single span [a, b) (``rewire2``, e.g. ``rewire-unet-v4r-ep1``: [9, 23), proj P without
    ``up_detail``, ``detail="conv3"`` guided conv decoder or ``detail="resnl"``, learned merge,
    min(a, N-b) skips: output of block a-m+j -> input of block b+m-1-j).
The coarse grid uses re-indexed integer RoPE positions (the model's own ``pos_embedder`` on the coarse shape).

``install()`` wraps ``comfy.sd.load_diffusion_model_state_dict``: rewire keys are popped before
ComfyUI's model detection, the model loads as a plain Anima, then a ``rewire`` submodule is registered on
the diffusion model and its ``_forward`` is replaced on the instance (``forward`` and the
DIFFUSION_MODEL wrapper executor are untouched). Dense files are not affected.

Run-time modes (``transformer_options``, set by the "Anima Rewire Mode" node):
  ``anima_rewire`` = "rewired" (default) | "dense"  -- dense-bypass: same DiT weights, all blocks full-res,
                                                         rewire tensors unused (= reference ``forward_with(model, None)``)
  ``anima_rewire_sigma_threshold`` = tau > 0         -- sigma hybrid: rewired while sigma > tau, dense-bypass at
                                                         sigma <= tau (sigma = transformer_options["sigmas"])
"""
from __future__ import annotations

import json
import logging
import os
import re
import types
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as Fn

import comfy.ldm.common_dit
import comfy.ops

REWIRE_FORMAT = "anima-rewire-unet-v1"
ENV_VAR = "ANIMA_REWIRE"
OPTION_KEY = "anima_rewire"  # transformer_options[OPTION_KEY] = "dense" forces the stock block loop
SKIP_X_KEY = "anima_rewire_extra_skip_scale"  # A/B: multiply only the extra skips (src blocks 0-3, added in the ep3 distillation)
SKIP_KEY = "anima_rewire_skip_scale"  # A/B: multiply every long U-ViT skip (incl. extra skips)
SIGMA_KEY = "anima_rewire_sigma_threshold"  # transformer_options[SIGMA_KEY] = tau: dense-bypass when sigma <= tau
DEFAULT_F = 7
_SUBKEYS = ("level1.", "level2.", "extra_levels.", "extra_merges.", "merge.", "skips.", "extra_skips.", "detail.")
_KEY_RE = re.compile(r"^(.*?)rewire\.(?:" + "|".join(re.escape(s) for s in _SUBKEYS) + r")")
_warned = set()


def _warn_once(tag, msg, level=logging.WARNING):
    if tag not in _warned:
        _warned.add(tag)
        logging.log(level, msg)


# ----------------------------------------------------------------------------------------------- modules

def _to_2d(x):  # (B,T,H,W,D) -> (B*T, D, H, W)
    B, T, H, W, D = x.shape
    return x.permute(0, 1, 4, 2, 3).reshape(B * T, D, H, W), (B, T)


def _from_2d(y, bt):
    B, T = bt
    _, D, H, W = y.shape
    return y.reshape(B, T, D, H, W).permute(0, 1, 3, 4, 2)


def _pad_even(y):
    H, W = y.shape[-2:]
    if H % 2 or W % 2:
        y = Fn.pad(y, (0, W % 2, 0, H % 2), mode="replicate")
    return y


class RewireLevel(nn.Module):
    """One 2x2 pooling level. learnable: P = depthwise 2x2/s2 conv (``p``), U = depthwise 4x4/s2 transposed conv
    on a replicate-padded grid (``u``). proj additionally: ``down_res`` Linear(4D->D) on the pixel-unshuffled
    2x2 neighbourhood added to P and (unless ``up_detail=False``) ``up_detail`` Linear(D->4D) + pixel-shuffle
    added to U."""

    def __init__(self, D, learnable, proj, up_detail=True, device=None, dtype=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.D, self.learnable, self.proj = D, learnable, proj
        self.has_up_detail = bool(proj and up_detail)
        if learnable:
            self.p = operations.Conv2d(D, D, 2, stride=2, groups=D, bias=False, device=device, dtype=dtype)
            self.u = operations.ConvTranspose2d(D, D, 4, stride=2, groups=D, bias=False, device=device, dtype=dtype)
        if proj:
            self.down_res = operations.Linear(4 * D, D, bias=False, device=device, dtype=dtype)
        if self.has_up_detail:
            self.up_detail = operations.Linear(D, 4 * D, bias=False, device=device, dtype=dtype)

    def down(self, x):
        """(B,T,H,W,D) -> (B,T,ceil(H/2),ceil(W/2),D); odd sides are replicate-padded."""
        y, bt = _to_2d(x)
        y = _pad_even(y)
        base = _from_2d(self.p(y) if self.learnable else Fn.avg_pool2d(y, 2), bt)
        if self.proj:
            base = base + self.down_res(_from_2d(Fn.pixel_unshuffle(y, 2), bt))
        return base

    def up(self, x, H, W):
        """(B,T,h,w,D) -> (B,T,H,W,D): bilinear (align_corners=False, edge clamp) at exact init."""
        y, bt = _to_2d(x)
        h, w = y.shape[-2:]
        if self.learnable and getattr(self, "u_smooth", False):
            # A/B: replace U's learned 4x4 shape by the exact bilinear kernel, keeping each channel's DC gain
            # (mean of its 4 phase sums) -> removes the nearest-neighbour-like 2x2 copy pattern
            wt = self.u.weight.to(device=y.device, dtype=torch.float32)
            gain = torch.stack([wt[..., py::2, px::2].sum((-1, -2)) for py in (0, 1) for px in (0, 1)], -1).mean(-1)
            k1 = torch.tensor([0.25, 0.75, 0.75, 0.25], device=y.device)
            wt = gain[..., None, None] * torch.outer(k1, k1)
            y = Fn.conv_transpose2d(Fn.pad(y, (1, 1, 1, 1), mode="replicate"), wt.to(y.dtype), stride=2,
                                    groups=self.D)[..., 3:3 + 2 * h, 3:3 + 2 * w]
        elif self.learnable and getattr(self, "u_balanced", False):
            # A/B ablation: equalise the 4 output-phase sums of each channel's stride-2 4x4 kernel (removes the
            # 2x2-token periodic DC pattern a flat input would get; the AC part of U is unchanged)
            wt = self.u.weight.to(device=y.device, dtype=torch.float32)
            ph = torch.stack([wt[..., py::2, px::2].sum((-1, -2)) for py in (0, 1) for px in (0, 1)], -1)
            mean = ph.mean(-1)
            wt = wt.clone()
            for n, (py, px) in enumerate(((0, 0), (0, 1), (1, 0), (1, 1))):
                wt[..., py::2, px::2] += ((mean - ph[..., n]) / 4)[..., None, None]
            y = Fn.conv_transpose2d(Fn.pad(y, (1, 1, 1, 1), mode="replicate"), wt.to(y.dtype), stride=2,
                                    groups=self.D)[..., 3:3 + 2 * h, 3:3 + 2 * w]
        elif self.learnable:
            y = self.u(Fn.pad(y, (1, 1, 1, 1), mode="replicate"))[..., 3:3 + 2 * h, 3:3 + 2 * w]
        else:
            y = Fn.interpolate(y, size=(2 * h, 2 * w), mode="bilinear", align_corners=False)
        out = _from_2d(y[..., :H, :W], bt)
        if self.has_up_detail:
            d, _ = _to_2d(self.up_detail(x))
            out = out + _from_2d(Fn.pixel_shuffle(d, 2)[..., :H, :W], bt)
        return out

    @torch.no_grad()
    def init_exact(self):
        """Exact-init values (average pool / bilinear, zero residual branches)."""
        if self.learnable:
            self.p.weight = nn.Parameter(torch.full((self.D, 1, 2, 2), 0.25), requires_grad=False)
            k1 = torch.tensor([0.25, 0.75, 0.75, 0.25])
            self.u.weight = nn.Parameter(torch.outer(k1, k1)[None, None].repeat(self.D, 1, 1, 1), requires_grad=False)
        if self.proj:
            self.down_res.weight = nn.Parameter(torch.zeros(self.D, 4 * self.D), requires_grad=False)
        if self.has_up_detail:
            self.up_detail.weight = nn.Parameter(torch.zeros(4 * self.D, self.D), requires_grad=False)


def _ln_pair(u, h, D):
    """cat[LN(u), LN(h)] (LayerNorm without affine, eps 1e-6); half-precision inputs are normalised in fp32."""
    up = (lambda t: t.float()) if u.dtype in (torch.float16, torch.bfloat16) else (lambda t: t)
    z = torch.cat([Fn.layer_norm(up(u), (D,), eps=1e-6), Fn.layer_norm(up(h).to(up(u).dtype), (D,), eps=1e-6)], dim=-1)
    return z.to(u.dtype)


class ConvDetail(nn.Module):
    """rewire2 ``detail="conv3"``: guided conv decoder on the full-res token grid,
    d = out1x1(GELU(conv3x3_2(GELU(conv3x3_1(GELU(conv3x3_0(GELU(inp1x1(cat[LN(U.delta), LN(h)])))))))))."""

    def __init__(self, D, C=1024, layers=3, device=None, dtype=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.D, self.C = D, C
        self.inp = operations.Conv2d(2 * D, C, 1, device=device, dtype=dtype)
        self.convs = nn.ModuleList([operations.Conv2d(C, C, 3, padding=1, device=device, dtype=dtype) for _ in range(layers)])
        self.out = operations.Conv2d(C, D, 1, bias=False, device=device, dtype=dtype)

    def forward(self, u, h):
        y, bt = _to_2d(_ln_pair(u, h, self.D))
        y = Fn.gelu(self.inp(y))
        for c in self.convs:
            y = Fn.gelu(c(y))
        return _from_2d(self.out(y), bt)

    @torch.no_grad()
    def init_exact(self):
        self.out.weight = nn.Parameter(torch.zeros(self.D, self.C, 1, 1), requires_grad=False)


class ResNLDetail(nn.Module):
    """rewire2 ``detail="resnl"``: a = GELU(w1(cat[LN(u), LN(h)])); a = a + dw3x3(a) (optional); d = w2(a)."""

    def __init__(self, D, hidden=None, dw=False, device=None, dtype=None, operations=comfy.ops.manual_cast):
        super().__init__()
        Dh = int(hidden or D)
        self.D, self.Dh = D, Dh
        self.w1 = operations.Linear(2 * D, Dh, bias=True, device=device, dtype=dtype)
        self.w2 = operations.Linear(Dh, D, bias=False, device=device, dtype=dtype)
        self.dw_weight = nn.Parameter(torch.zeros(Dh, 1, 3, 3, device=device, dtype=dtype), requires_grad=False) if dw else None

    def forward(self, u, h):
        a = Fn.gelu(self.w1(_ln_pair(u, h, self.D)))
        if self.dw_weight is not None:
            y, bt = _to_2d(a)
            w = self.dw_weight.to(device=y.device, dtype=y.dtype)
            a = a + _from_2d(Fn.conv2d(y, w, padding=1, groups=self.Dh), bt)
        return self.w2(a)

    @torch.no_grad()
    def init_exact(self):
        self.w2.weight = nn.Parameter(torch.zeros(self.D, self.Dh), requires_grad=False)


def normalize_config(cfg, num_blocks):
    """Validate a rewire JSON config and resolve it with the reference ``load_rewire`` / ``Rewire`` semantics."""
    N = int(cfg.get("num_blocks", num_blocks))
    if N != num_blocks:
        raise ValueError(f"Anima rewire config is for {N} blocks but the DiT has {num_blocks}.")
    mode = cfg.get("mode", "subtract")
    inner, skips = int(cfg.get("inner", 0) or 0), bool(cfg.get("skips", False))
    learnable_pu, learnable_merge, proj = bool(cfg.get("learnable_pu", False)), bool(cfg.get("learnable_merge", False)), bool(cfg.get("proj", False))
    proj_up_detail = bool(cfg.get("proj_up_detail", True)) if proj else True
    detail = cfg.get("detail") or None
    detail_dw = bool(cfg.get("detail_dw", False)) if detail == "resnl" else False
    detail_hidden = cfg.get("detail_hidden")
    F, spans, inner_span = cfg.get("F"), cfg.get("spans"), cfg.get("inner_span")
    if detail not in (None, "resnl", "conv3"):
        raise ValueError(f"Anima rewire: unsupported detail branch {detail!r} (this plugin knows conv3 / resnl).")
    if mode not in ("subtract", "naive"):
        raise ValueError(f"Anima rewire: unsupported mode {mode!r}.")
    if proj and not learnable_pu:
        raise ValueError("Anima rewire: proj pooling extends the learnable P/U.")
    if spans is None or F is not None:  # F-symmetric (old files: F set)
        if F is None:
            raise ValueError("Anima rewire: config has neither F nor spans.")
        F = int(F)
        if not (1 <= F and 2 * F < N):
            raise ValueError(f"Anima rewire: F={F} invalid for {N} blocks.")
        spans = [(F, N - F)]
    else:
        spans = sorted((int(a), int(b)) for a, b in spans)
        if inner:
            raise ValueError("Anima rewire: inner only with the F-symmetric span.")
        if skips and len(spans) != 1:
            raise ValueError("Anima rewire: skips need a single span.")
        F = spans[0][0] if spans == [(spans[0][0], N - spans[0][0])] else None
    for (a, b), nxt in zip(spans, spans[1:] + [(N, N)]):
        if not (1 <= a < b <= N - 1 and b <= nxt[0]):
            raise ValueError(f"Anima rewire: spans must be disjoint inside [1, N-1), got {spans}.")
    if inner_span is not None:
        a2, b2 = (int(v) for v in inner_span)
        if F is None or not (F < a2 < b2 < N - F):
            raise ValueError(f"Anima rewire: inner_span {inner_span} must lie strictly inside [F, N-F).")
        inner_span, inner = [a2, b2], b2 - a2
    elif inner and not F + inner < N - F - inner:
        raise ValueError("Anima rewire: inner span empty.")
    if detail and (mode != "subtract" or inner or len(spans) != 1):
        raise ValueError("Anima rewire: detail branch needs one span, subtract mode, no inner level.")
    a0, b0 = spans[0]
    return {"num_blocks": N, "F": F, "spans": [list(s) for s in spans], "symmetric": F is not None, "inner": inner,
            "inner_span": inner_span, "mode": mode, "learnable_pu": learnable_pu, "learnable_merge": learnable_merge,
            "skips": skips, "n_skips": min(a0, N - b0) if skips else 0, "proj": proj, "proj_up_detail": proj_up_detail,
            "detail": detail, "detail_dw": detail_dw,
            "extra_skip_pairs": [[int(s_), int(d_)] for s_, d_ in (cfg.get("extra_skip_pairs") or [])] if skips else [],
            "detail_hidden": (int(detail_hidden) if detail_hidden else (1024 if detail == "conv3" else None)) if detail else None}


class AnimaRewire(nn.Module):
    """Learnables of the rewire, registered on the diffusion model as ``rewire`` (same layout as the reference
    ``Rewire``; ``p_weight``/``u_weight`` live in ``p.weight``/``u.weight``)."""

    def __init__(self, cfg, D, device=None, dtype=None, operations=comfy.ops.manual_cast):
        super().__init__()
        self.cfg = dict(cfg)
        self.D = D
        self.N, self.F, self.inner, self.mode = cfg["num_blocks"], cfg["F"], cfg["inner"], cfg["mode"]
        self.spans = [tuple(s) for s in cfg["spans"]]
        self.inner_span = tuple(cfg["inner_span"]) if cfg.get("inner_span") else None
        self.symmetric = cfg["symmetric"]
        self.learnable_merge = cfg["learnable_merge"]
        mk = lambda: RewireLevel(D, cfg["learnable_pu"], cfg["proj"], up_detail=cfg.get("proj_up_detail", True),
                                 device=device, dtype=dtype, operations=operations)
        lin = lambda i, o: operations.Linear(i, o, bias=False, device=device, dtype=dtype)
        self.level1 = mk()
        self.level2 = mk() if self.inner else None
        self.extra_levels = nn.ModuleList([mk() for _ in self.spans[1:]]) if len(self.spans) > 1 else None
        self.merge = lin(2 * D, D) if self.learnable_merge else None
        self.extra_merges = nn.ModuleList([lin(2 * D, D) for _ in self.spans[1:]]) if (self.learnable_merge and len(self.spans) > 1) else None
        n_skips = cfg.get("n_skips", self.F if cfg["skips"] else 0)
        self.skips = nn.ModuleList([lin(D, D) for _ in range(n_skips)]) if cfg["skips"] else None
        self.extra_skip_pairs = [tuple(pq) for pq in cfg.get("extra_skip_pairs") or []]
        self.extra_skips = nn.ModuleList([lin(D, D) for _ in self.extra_skip_pairs]) if self.extra_skip_pairs else None
        self.detail = None
        if cfg.get("detail") == "conv3":
            self.detail = ConvDetail(D, cfg["detail_hidden"], device=device, dtype=dtype, operations=operations)
        elif cfg.get("detail") == "resnl":
            self.detail = ResNLDetail(D, cfg["detail_hidden"], cfg["detail_dw"], device=device, dtype=dtype, operations=operations)
        self.identity_middle = False  # test hook: middle blocks -> identity
        self.skip_scale = 1.0  # A/B knob, set per call from transformer_options
        self.extra_skip_scale = 1.0

    def level(self, k):
        return self.level1 if k == 0 else self.extra_levels[k - 1]

    def merger(self, k):
        if not self.learnable_merge:
            return None
        return self.merge if k == 0 else self.extra_merges[k - 1]

    def add_detail(self, up_delta, h):
        if self.detail is None or getattr(self, "ablate_no_detail", False):
            return up_delta
        return up_delta + self.detail(up_delta, h)

    def inner_bounds(self):
        if self.inner_span is not None:
            return self.inner_span
        return (self.F + self.inner, self.N - self.F - self.inner)

    def levels(self):
        return [m for m in self.modules() if isinstance(m, RewireLevel)]

    @torch.no_grad()
    def init_exact(self):
        D = self.D
        for lv in self.levels():
            lv.init_exact()
        for m in ([self.merge] if self.merge is not None else []) + (list(self.extra_merges) if self.extra_merges is not None else []):
            m.weight = nn.Parameter(torch.cat([torch.eye(D), torch.eye(D)], dim=1), requires_grad=False)
        for m in (self.skips if self.skips is not None else []):
            m.weight = nn.Parameter(torch.zeros(D, D), requires_grad=False)
        if self.detail is not None:
            self.detail.init_exact()
        return self

    def summary(self):
        pooled = [i for a, b in self.spans for i in range(a, b)]
        full = [i for i in range(self.N) if i not in pooled]
        c = self.cfg
        pu = ("proj" + ("" if c.get("proj_up_detail", True) else " (no up_detail)")) if c["proj"] else "learnable" if c["learnable_pu"] else "analytic"
        parts = [f"spans={[list(s) for s in self.spans]}", f"full_res={full}" if full else "",
                 f"mode={self.mode}", "P/U=" + pu, f"detail={c.get('detail') or 'none'}",
                 f"merge={'learnable' if self.learnable_merge else 'add'}", f"skips={len(self.skips) if self.skips is not None else 0}"]
        if self.inner:
            parts.append(f"inner={list(self.inner_bounds())}")
        return ", ".join(p for p in parts if p)


# ----------------------------------------------------------------------------------------------- forward

def _rewired_blocks(model, rw, x, run, rope, coarse_rope):
    """Port of the reference ``rewired_blocks`` / ``_multi_span`` (rw not None)."""
    N = len(model.blocks)
    B, T, H, W, D = x.shape
    if not rw.inner and rw.skips is None:
        prev, rope2 = 0, None
        for k, (a, b) in enumerate(rw.spans):
            x = run(range(prev, a), x, rope)
            lvl = rw.level(k)
            h = x
            low = lvl.down(h)
            if rope2 is None:
                rope2 = coarse_rope(low)
            m = low if rw.identity_middle else run(range(a, b), low, rope2)
            if rw.mode == "naive":
                x = lvl.up(m, H, W)
            else:
                up_delta = lvl.up(m - low, H, W)
                if k == 0:
                    up_delta = rw.add_detail(up_delta, h)
                mg = rw.merger(k)
                x = mg(torch.cat([up_delta, h], dim=-1)) if mg is not None else h + up_delta
            prev = b
        return run(range(prev, N), x, rope)

    a0, b0 = rw.spans[0]  # = (F, N-F) for the F-symmetric rewire
    skip_src = []
    for i in range(a0):
        x = run([i], x, rope)
        skip_src.append(x)  # output of block i (skips only)
    h = x
    low = rw.level1.down(h)
    H2, W2 = low.shape[2], low.shape[3]
    rope2 = coarse_rope(low)
    if rw.identity_middle:
        m = low
    elif rw.inner:
        a2, b2 = rw.inner_bounds()
        m = run(range(a0, a2), low, rope2)
        low2 = rw.level2.down(m)
        m2 = run(range(a2, b2), low2, coarse_rope(low2))
        m = m + rw.level2.up(m2 - low2, H2, W2) if rw.mode == "subtract" else rw.level2.up(m2, H2, W2)
        m = run(range(b2, b0), m, rope2)
    else:
        m = run(range(a0, b0), low, rope2)
    if rw.mode == "naive":
        x = rw.level1.up(m, H, W)
    else:
        up_delta = rw.add_detail(rw.level1.up(m - low, H, W), h)
        x = rw.merge(torch.cat([up_delta, h], dim=-1)) if rw.merge is not None else h + up_delta
    msk = len(rw.skips) if rw.skips is not None else 0
    for i in range(b0, N):
        j = msk - 1 - (i - b0)
        if j >= 0:
            x = x + rw.skip_scale * rw.skips[j](skip_src[a0 - msk + j])  # output of block a0-msk+j -> input of block b0+msk-1-j
        if rw.extra_skips is not None:  # rewire.json extra_skip_pairs: (src, dst), dst sums all its pairs
            for k_, (s_, d_) in enumerate(rw.extra_skip_pairs):
                if d_ == i:
                    x = x + rw.skip_scale * rw.extra_skip_scale * rw.extra_skips[k_](skip_src[s_])
        x = run([i], x, rope)
    return x


def _current_sigma(transformer_options, timesteps):
    s = transformer_options.get("sigmas")
    if s is None:
        s = timesteps  # Anima / ModelSamplingAuraFlow: timestep = sigma * 1.0
    if s is None:
        return None
    return float(s.detach().float().max()) if torch.is_tensor(s) else float(s)


def _dense_requested(transformer_options, timesteps=None):
    if os.environ.get(ENV_VAR, "1").strip().lower() in ("0", "off", "false", "dense", "no"):
        return "env"
    if str(transformer_options.get(OPTION_KEY, "")).lower() == "dense":
        return "node"
    tau = float(transformer_options.get(SIGMA_KEY, 0.0) or 0.0)
    if tau > 0:
        sigma = _current_sigma(transformer_options, timesteps)
        if sigma is not None and sigma <= tau:
            return "sigma"
    return None


def rewired_forward(self, x, timesteps, context, fps=None, padding_mask=None, **kwargs):
    """MiniTrainDIT._forward with the block loop replaced by the rewired schedule."""
    transformer_options = kwargs.get("transformer_options", {})
    rw = getattr(self, "rewire", None)
    why = _dense_requested(transformer_options, timesteps) if rw is not None else "missing"
    if why is not None:
        if why == "sigma":
            _warn_once("dense-sigma", f"Anima rewire: sigma hybrid active, dense-bypass for sigma <= "
                                      f"{float(transformer_options.get(SIGMA_KEY)):g}.", logging.INFO)
        else:
            _warn_once("dense-" + why, f"Anima rewire: dense-bypass block loop forced ({'ANIMA_REWIRE=0' if why == 'env' else why}). "
                                       "The weights were trained for the rewired path; full dense-bypass sampling is for A/B only.")
        return type(self)._forward(self, x, timesteps, context, fps, padding_mask, **kwargs)

    mode = str(transformer_options.get(OPTION_KEY, "rewired")).lower()
    rw.ablate_no_detail = "no_detail" in mode
    rw.skip_scale = float(transformer_options.get(SKIP_KEY, 1.0))
    rw.extra_skip_scale = float(transformer_options.get(SKIP_X_KEY, 1.0))
    for lv in rw.levels():
        lv.u_balanced = "u_balanced" in mode
        lv.u_smooth = "u_smooth" in mode
    if mode != "rewired":
        _warn_once("ablate-" + mode, f"Anima rewire: A/B ablation mode {mode!r} (no_detail = conv3 detail branch off, "
                                     "u_balanced = up-sampling kernel phase sums equalised).", logging.INFO)

    orig_shape = list(x.shape)
    x = comfy.ldm.common_dit.pad_to_patch_size(x, (self.patch_temporal, self.patch_spatial, self.patch_spatial))
    x_B_C_T_H_W = x
    timesteps_B_T = timesteps
    crossattn_emb = context
    x_B_T_H_W_D, rope_emb, extra_pos_emb = self.prepare_embedded_sequence(x_B_C_T_H_W, fps=fps, padding_mask=padding_mask)
    if extra_pos_emb is not None:
        raise NotImplementedError("Anima rewire: extra_per_block_abs_pos_emb is not supported (as in the reference).")
    if timesteps_B_T.ndim == 1:
        timesteps_B_T = timesteps_B_T.unsqueeze(1)
    t_emb, adaln_lora = self.t_embedder[1](self.t_embedder[0](timesteps_B_T).to(x_B_T_H_W_D.dtype))
    t_emb = self.t_embedding_norm(t_emb)

    patches = transformer_options.get("patches", {})
    if "post_input" in patches:
        transformer_options = transformer_options.copy()
        transformer_options["model_patch_data"] = {}
        for patch in patches["post_input"]:
            out = patch({"img": x_B_T_H_W_D, "x": x_B_C_T_H_W, "transformer_options": transformer_options})
            x_B_T_H_W_D = out["img"]

    # Residual stream in fp32 for fp16 compute, as in the stock forward.
    if x_B_T_H_W_D.dtype == torch.float16:
        x_B_T_H_W_D = x_B_T_H_W_D.float()

    def run(idx, h, rope):
        for i in idx:
            transformer_options["block_index"] = i
            h = self.blocks[i](h, t_emb, crossattn_emb, rope_emb_L_1_1_D=rope, adaln_lora_B_T_3D=adaln_lora,
                               extra_per_block_pos_emb=None, transformer_options=transformer_options)
        return h

    def coarse_rope(low):
        # re-indexed integer positions 0..H2-1 / 0..W2-1 on the pooled grid (same pos_embedder, temporal unchanged)
        B, T, H2, W2, D = low.shape
        return self.pos_embedder.generate_embeddings(torch.Size((B, T, H2, W2, D)), fps=fps, device=low.device).unsqueeze(1).unsqueeze(0)

    x_B_T_H_W_D = _rewired_blocks(self, rw, x_B_T_H_W_D, run, rope_emb.unsqueeze(1).unsqueeze(0), coarse_rope)

    out = self.final_layer(x_B_T_H_W_D.to(crossattn_emb.dtype), t_emb, adaln_lora_B_T_3D=adaln_lora)
    return self.unpatchify(out)[:, :, :orig_shape[-3], :orig_shape[-2], :orig_shape[-1]]


# ----------------------------------------------------------------------------------------------- loading

def _canonical_key(k):
    """File key (reference names) -> module key."""
    return re.sub(r"(^|\.)(p|u)_weight$", lambda m: f"{m.group(1)}{m.group(2)}.weight", k)


def extract_rewire(sd, metadata=None):
    """Pop ``<prefix>rewire.*`` tensors from ``sd``. Returns (tensors without prefix, raw config or None) or None."""
    prefixes = set()
    for k in sd.keys():
        m = _KEY_RE.match(k)
        if m:
            prefixes.add(m.group(1))
    prefixes = [p for p in prefixes if (p + "blocks.0.self_attn.q_proj.weight") in sd or any(k.startswith(p + "blocks.0.") for k in sd)]
    if not prefixes:
        return None
    if len(prefixes) > 1:
        raise ValueError(f"Anima rewire: several rewire prefixes in one file: {sorted(prefixes)}")
    prefix = prefixes[0] + "rewire."
    tensors = {_canonical_key(k[len(prefix):]): sd.pop(k) for k in [k for k in sd.keys() if k.startswith(prefix)]}
    num_blocks = len({int(m.group(1)) for k in sd for m in [re.match(re.escape(prefixes[0]) + r"blocks\.(\d+)\.", k)] if m})
    metadata = metadata or {}
    cfg = None
    fmt = metadata.get("anima_rewire_format")
    if metadata.get("anima_rewire"):
        if fmt not in (None, REWIRE_FORMAT):
            raise ValueError(f"Anima rewire: unsupported format {fmt!r} (this plugin reads {REWIRE_FORMAT}).")
        cfg = json.loads(metadata["anima_rewire"])
    return {"tensors": tensors, "config": cfg, "num_blocks": num_blocks, "prefix": prefixes[0]}


def infer_config(tensors, num_blocks):
    """Best-effort config for files without metadata: F-symmetric span with F = number of skips (default 7)."""
    skips = sorted(int(k.split(".")[1]) for k in tensors if k.startswith("skips."))
    if any(k.startswith(("level2.", "extra_levels.", "extra_merges.")) for k in tensors):
        raise ValueError("Anima rewire: file has inner/multi-span tensors but no 'anima_rewire' metadata; cannot infer spans.")
    if any(k.startswith("detail.") for k in tensors):
        raise ValueError("Anima rewire: file has rewire2 detail tensors but no 'anima_rewire' metadata; cannot infer the span.")
    F = len(skips) if skips else DEFAULT_F
    return {"num_blocks": num_blocks, "F": F, "spans": [[F, num_blocks - F]], "inner": 0, "mode": "subtract",
            "learnable_pu": "level1.p.weight" in tensors, "learnable_merge": "merge.weight" in tensors,
            "skips": bool(skips), "proj": "level1.down_res.weight" in tensors,
            "proj_up_detail": "level1.up_detail.weight" in tensors}


def attach_rewire(patcher, found):
    """Register the rewire on ``patcher.model.diffusion_model`` and switch its ``_forward``."""
    from comfy.ldm.cosmos.predict2 import MiniTrainDIT

    dm = patcher.model.diffusion_model
    if not isinstance(dm, MiniTrainDIT):
        raise ValueError(f"Anima rewire tensors found, but the file loaded as {type(dm).__name__}, not an Anima/Cosmos DiT.")
    N, D = len(dm.blocks), dm.model_channels
    tensors = found["tensors"]
    if found["config"] is None:
        raw = infer_config(tensors, N)
        logging.warning("Anima rewire: file has rewire tensors but no '%s' metadata; inferred config from tensor shapes "
                        "(spans=%s). Re-export with metadata if this is not the default layout.", "anima_rewire", raw["spans"])
    else:
        raw = found["config"]
    cfg = normalize_config(raw, N)

    ref = dm.blocks[0].mlp.layer1.weight
    dtype = patcher.model.get_dtype()
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        dtype = next(iter(tensors.values())).dtype  # e.g. fp8 DiT storage: keep the small rewire in its file dtype
    device = ref.device if ref.device.type != "meta" else torch.device("cpu")
    rw = AnimaRewire(cfg, D, device=device, dtype=dtype)
    tensors = {k: v.to(device=device, dtype=dtype) for k, v in tensors.items()}
    rw.load_state_dict(tensors, strict=True, assign=patcher.is_dynamic())
    rw.requires_grad_(False)
    if hasattr(dm, "rewire"):
        del dm.rewire
    dm.add_module("rewire", rw)
    dm._forward = types.MethodType(rewired_forward, dm)
    dm.anima_rewire_config = cfg
    patcher.size = 0  # recompute model_size() including the rewire
    n_params = sum(t.numel() for t in tensors.values())
    variant = raw.get("variant") if isinstance(raw, dict) else None
    logging.info(f"Anima rewire ({REWIRE_FORMAT}{', ' + variant if variant else ''}) detected: {rw.summary()}; "
                 f"+{n_params / 1e6:.1f}M params."
                 + (f" {ENV_VAR}=0 is set: dense-bypass loop (A/B only for these weights)." if _dense_requested({}) else ""))
    return rw


def is_rewired(model):
    dm = getattr(getattr(model, "model", model), "diffusion_model", None)
    # duck-typed: the plugin package may be imported under more than one module name
    return dm is not None and getattr(dm, "anima_rewire_config", None) is not None and "rewire" in dm._modules


def install():
    import comfy.sd

    current = comfy.sd.load_diffusion_model_state_dict
    if getattr(current, "_anima_rewire_hook", False):
        return current
    original = current

    def load_diffusion_model_state_dict(sd, model_options={}, metadata=None, disable_dynamic=False):
        found = extract_rewire(sd, metadata)
        patcher = original(sd, model_options=model_options, metadata=metadata, disable_dynamic=disable_dynamic)
        if found is not None and patcher is not None:
            attach_rewire(patcher, found)
        return patcher

    load_diffusion_model_state_dict.__doc__ = original.__doc__
    load_diffusion_model_state_dict._anima_rewire_hook = True
    load_diffusion_model_state_dict._anima_rewire_original = original
    comfy.sd.load_diffusion_model_state_dict = load_diffusion_model_state_dict
    return load_diffusion_model_state_dict


def uninstall():
    import comfy.sd

    current = comfy.sd.load_diffusion_model_state_dict
    if getattr(current, "_anima_rewire_hook", False):
        comfy.sd.load_diffusion_model_state_dict = current._anima_rewire_original
