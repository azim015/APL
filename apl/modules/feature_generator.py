"""Surrogate Feature Generator ``theta_FG`` (Sec. V-B2).

A single lightweight cross-attention block synthesises diverse prompted text
features from one base prompt feature ``w`` and a conditioning signal
``alpha``:

.. math::

    Q &= \\mathrm{LN}(w)\\,\\theta_{in_1}, \\quad
    K = \\mathrm{LN}(\\alpha)\\,\\theta_{in_2}, \\quad
    V = \\mathrm{LN}(\\alpha)\\,\\theta_{in_3} \\\\
    \\mathrm{Attn} &= w + \\big(\\mathrm{softmax}(QK^\\top/\\sqrt{d_k})\\,V\\big)\\theta_{out} \\\\
    h &= \\mathrm{Attn} + \\mathrm{ReLU}\\big(\\mathrm{LN}(\\mathrm{Attn})\\,\\theta_{fc_1}\\big)\\theta_{fc_2}

Both residual additions of the paper are kept (``w`` into the attention block
and ``Attn`` into the MLP block).  The bottleneck width is ``d_mid = d / 2``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FeatureGenerator(nn.Module):
    """``h = theta_FG(w, alpha)`` with ``h`` of shape ``[..., C, d]``."""

    def __init__(self, cfg):
        super().__init__()
        d, d_k = cfg.dim, cfg.key_dim
        self.dim = d
        self.key_dim = d_k
        self.num_heads = max(1, cfg.num_heads)
        if d_k % self.num_heads != 0:
            raise ValueError("key_dim must be divisible by num_heads")
        self.head_dim = d_k // self.num_heads
        self.residual = cfg.residual

        self.ln_q = nn.LayerNorm(d)
        self.ln_kv = nn.LayerNorm(d)
        self.theta_in1 = nn.Linear(d, d_k, bias=False)
        self.theta_in2 = nn.Linear(d, d_k, bias=False)
        self.theta_in3 = nn.Linear(d, d_k, bias=False)
        self.theta_out = nn.Linear(d_k, d, bias=False)

        self.ln_mlp = nn.LayerNorm(d)
        self.theta_fc1 = nn.Linear(d, cfg.mid_dim, bias=False)
        self.theta_fc2 = nn.Linear(cfg.mid_dim, d, bias=False)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(cfg.dropout) if cfg.dropout > 0 else nn.Identity()

        self.reset_parameters()

    # ------------------------------------------------------------------ #
    def reset_parameters(self) -> None:
        for layer in (self.theta_in1, self.theta_in2, self.theta_in3,
                      self.theta_fc1):
            nn.init.normal_(layer.weight, std=0.02)
        # zero-init the output projections so the generator starts as identity
        nn.init.zeros_(self.theta_out.weight)
        nn.init.zeros_(self.theta_fc2.weight)

    # ------------------------------------------------------------------ #
    def forward(self, w: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        w:      ``[C, d]`` or ``[B, C, d]`` base prompted text features (query).
        alpha:  ``[L, d]`` or ``[B, L, d]`` conditioning signal (keys/values).

        Returns
        -------
        ``[B, C, d]`` (or ``[C, d]`` when both inputs are unbatched).
        """
        squeeze = w.dim() == 2 and alpha.dim() == 2
        w_b = w.unsqueeze(0) if w.dim() == 2 else w
        a_b = alpha.unsqueeze(0) if alpha.dim() == 2 else alpha
        if w_b.shape[0] == 1 and a_b.shape[0] > 1:
            w_b = w_b.expand(a_b.shape[0], -1, -1)
        if a_b.shape[0] == 1 and w_b.shape[0] > 1:
            a_b = a_b.expand(w_b.shape[0], -1, -1)
        if w_b.shape[0] != a_b.shape[0]:
            raise ValueError(
                f"batch mismatch between w {tuple(w.shape)} and alpha {tuple(alpha.shape)}"
            )

        b, c, _ = w_b.shape
        length = a_b.shape[1]

        q = self.theta_in1(self.ln_q(w_b))
        kv_in = self.ln_kv(a_b)
        k = self.theta_in2(kv_in)
        v = self.theta_in3(kv_in)

        q = q.view(b, c, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(b, length, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(b, length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        weights = torch.softmax(scores, dim=-1)
        context = weights @ v                                    # [B, H, C, hd]
        context = context.transpose(1, 2).reshape(b, c, self.key_dim)
        context = self.dropout(self.theta_out(context))

        attn = w_b + context if self.residual else context
        mlp = self.theta_fc2(self.act(self.theta_fc1(self.ln_mlp(attn))))
        h = attn + self.dropout(mlp) if self.residual else mlp

        return h.squeeze(0) if squeeze else h

    # ------------------------------------------------------------------ #
    def extra_repr(self) -> str:
        return f"dim={self.dim}, key_dim={self.key_dim}, heads={self.num_heads}"
