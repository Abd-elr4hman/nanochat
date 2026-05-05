"""
Mamba model wrapper (mirrors gpt.py structure).

The backbone is mambapy.Mamba; this file provides:
- token embedding (wte)
- final norm + lm_head with softcap (matches gpt.py)
- training/inference forward contract: (idx, targets) -> loss
- setup_optimizer with AdamW groups
- init_weights, estimate_flops, num_scaling_params

GPT-specific tricks are intentionally NOT carried over (rotary, sliding window,
value embeddings, smear, backout, resid_lambdas/x0_lambdas, kv_cache). Mamba is
its own opinionated block with its own internal residuals; the goal here is a
clean Mamba baseline, not a Mamba-with-attention-tricks-bolted-on.

================================================================================
COMPATIBILITY NOTES (for plugging into base_train.py)
================================================================================
- num_scaling_params(): Karpathy's scaling-law recipe in base_train.py uses
  `transformer_matrices + lm_head` (transformer matmul trunk + output head).
  For Mamba the equivalent is `backbone + lm_head`. We alias `backbone` under
  the key `transformer_matrices` so base_train.py's existing math just works.
  Also expose `value_embeds: 0` and `scalars: 0` so any other lookups don't
  KeyError silently.

================================================================================
TODO / FOLLOW-UPS
================================================================================
- Tri Dao SSM kernel: mambapy can call `mamba_ssm.ops.selective_scan_fn` if we
  set `use_cuda=True`. Plumb `use_cuda` through MambaConfig later for H100
  profiling. Caveats: kernel is fp32-only (no bf16/fp16); for bf16 we'd need
  Mamba-2's Triton kernel (`mamba_ssm.ops.triton.ssd_combined`, different API).
- Incremental decode: wire mambapy's `step()` for O(1) generation.

================================================================================
DESIGN DECISIONS LOG (revisit if results are off)
================================================================================

1. OPTIMIZER SPLIT
   - 2D backbone params -> Muon (matrix optimizer)
   - 1D backbone params (A_log, D, dt biases) -> AdamW
   Rationale: mirrors gpt.py's logic of "matrices to Muon, scalars to AdamW."
   Risk: Muon was tuned for transformer matmul shapes; not obvious it transfers
   cleanly to Mamba's different matrix shapes (input/output projections, conv,
   dt projection). May need to fall back to AdamW for the whole backbone if
   training is unstable.

2. WEIGHT INIT
   - wte:     normal(0, 0.8)     <- matches gpt.py
   - lm_head: normal(0, 0.001)   <- matches gpt.py
   - backbone: left to mambapy's defaults (battle-tested SSM-specific inits
               for A_log, D, dt projection). Don't clobber these.

3. FLOPS ESTIMATE
   Used simple `6 * params/token`. For Mamba the GPT attention term
   `12 * h * q * seq_len` is gone (Mamba is O(L), not O(L^2)). Embeddings
   excluded because they're lookups, not matmuls.

4. GENERATE
   Naive: re-run full sequence each step. Mamba's killer feature is O(1)
   incremental decoding via the `step()` API, but that's not wired up here.
   Fine for a training-time baseline; would matter for an inference benchmark.

5. KV CACHE
   `forward` accepts `kv_cache` for interface compatibility with GPT but
   ignores it. Mamba would use its own SSM state mechanism for incremental
   decoding, but we don't need this for the training comparison.

6. DROPPED FEATURES (vs gpt.py)
   - rotary embeddings (Mamba is inherently positional via its recurrence)
   - sliding window attention (no attention)
   - value embeddings, ve_gate (attention-specific)
   - smear (cheap bigram trick on the embedding side; Mamba's conv already
     gives local mixing)
   - backout (subtract mid-layer residual; very transformer-specific tuning)
   - resid_lambdas / x0_lambdas (per-layer residual scalars on top of
     transformer blocks; Mamba blocks have their own residuals)
================================================================================
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW, DistMuonAdamW

from mambapy.mamba import Mamba as MambaBackbone
from mambapy.mamba import MambaConfig as MambaBackboneConfig


@dataclass
class MambaConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_embd: int = 768
    # Mamba-specific
    d_state: int = 16
    d_conv: int = 4
    expand_factor: int = 2


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


class Linear(nn.Linear):
    """Same trick as gpt.py: master weights stay fp32, matmul casts down to activation dtype."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class Mamba(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE: same meta-device footgun as GPT — __init__ runs in meta context,
        so only shapes/dtypes here. Real init in init_weights().
        """
        super().__init__()
        self.config = config

        # Pad vocab for tensor-core / DDP efficiency (matches gpt.py)
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.padded_vocab_size = padded_vocab_size

        # Token embedding
        self.wte = nn.Embedding(padded_vocab_size, config.n_embd)

        # The Mamba backbone (mambapy expects its own config object)
        backbone_config = MambaBackboneConfig(
            d_model=config.n_embd,
            n_layers=config.n_layer,
            d_state=config.d_state,
            d_conv=config.d_conv,
            expand_factor=config.expand_factor,
        )
        self.backbone = MambaBackbone(backbone_config)

        # Output projection
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)

    @torch.no_grad()
    def init_weights(self):
        """
        wte:           normal, std=0.8  (match gpt.py)
        lm_head:       normal, std=0.001 (match gpt.py)
        backbone:      mambapy initializes its own params on materialization;
                       we leave their defaults alone.
        """
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Cast embedding to compute dtype (same as gpt.py, except for fp16 path)
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)

    def get_device(self):
        return self.wte.weight.device

    def estimate_flops(self):
        """
        Approximate forward+backward FLOPs per token.
        Mamba is O(L) in sequence length, so unlike GPT we don't have a
        seq-len-quadratic attention term. We use the matmul approximation:
        every matmul param contributes ~6 FLOPs/token (2 fwd + 4 bwd).
        Embeddings are pure lookups, not counted.
        """
        nparams = sum(p.numel() for p in self.parameters())
        nparams_exclude = self.wte.weight.numel()
        return 6 * (nparams - nparams_exclude)

    def num_scaling_params(self):
        wte = self.wte.weight.numel()
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        backbone = sum(p.numel() for p in self.backbone.parameters())
        total = wte + lm_head + backbone
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'lm_head': lm_head,
            'backbone': backbone,
            # Alias for compatibility with base_train.py's scaling-law math, which
            # uses 'transformer_matrices' + 'lm_head'. For Mamba, "all the trunk
            # matmul params" = backbone, so we expose it under the GPT key too.
            'transformer_matrices': backbone,
            'value_embeds': 0,
            'scalars': 0,
            'total': total,
        }

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        """
        Same optimizer recipe as GPT: AdamW for embeddings + lm_head, Muon for
        backbone matrix params. Mamba's internal scalars (dt biases, A_log, D)
        are 1D and go into AdamW alongside the scalar group.
        """
        model_dim = self.config.n_embd
        ddp, rank, local_rank, world_size = get_dist_info()

        # Split backbone params: strictly 2D matrices -> Muon; everything else -> AdamW.
        # NOTE: Muon's fused kernel assumes 2D weights. Mamba has a 3D conv1d weight
        # (shape: (d_inner, 1, d_conv)) that would crash Muon. The conv is also tiny
        # (~14K params total at d4) so AdamW is fine for it.
        matrix_params = [p for p in self.backbone.parameters() if p.ndim == 2]
        scalar_params = [p for p in self.backbone.parameters() if p.ndim != 2]
        embedding_params = [self.wte.weight]
        lm_head_params = list(self.lm_head.parameters())
        assert len(list(self.parameters())) == len(matrix_params) + len(scalar_params) + len(embedding_params) + len(lm_head_params)

        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling AdamW LRs by ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        param_groups = [
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=scalar_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        Factory = DistMuonAdamW if ddp else MuonAdamW
        optimizer = Factory(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        # kv_cache is accepted for interface compatibility with GPT but ignored;
        # incremental Mamba decoding uses its own state mechanism (not wired up here).
        B, T = idx.size()

        # Embed tokens
        x = self.wte(idx)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Mamba backbone: (B, T, D) -> (B, T, D)
        x = self.backbone(x)
        x = norm(x)

        # lm_head with softcap (same as gpt.py)
        softcap = 15
        logits = self.lm_head(x)
        logits = logits[..., :self.config.vocab_size]
        logits = logits.float()
        logits = softcap * torch.tanh(logits / softcap)

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive generation: re-run the full sequence each step.
        For efficient incremental decoding we'd plumb mambapy's step() API,
        but for a baseline this matches gpt.py's naive generate.
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)
        for _ in range(max_tokens):
            logits = self.forward(ids)
            logits = logits[:, -1, :]
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
