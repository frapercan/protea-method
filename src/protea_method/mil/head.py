"""Gated-attention MIL head for region-attributable GO prediction.

The head operates on frozen per-residue PLM embeddings (the ``(L, D)``
tensors produced by ``protea-backends`` once MIL.1 lands) and emits

1. A per-GO-term logit vector for the whole protein (``B, K``).
2. A per-GO-term residue-attention map (``B, L, K``) whose columns sum
   to 1 over the unmasked positions.

The attention head pairs (a) the global ranking signal needed by the
reranker with (b) a residue-level explanation that the MIL.4 features
(``attention_entropy``, ``interpro_overlap``, ``hierarchy_consistency``)
exploit and that the thesis interpretability chapter rests on.

Architecture follows Ilse, Tomczak and Welling (2018), "Attention-based
Deep Multiple Instance Learning" (ICML), with the gated variant

    A_unscaled = w^T (tanh(V H) (.) sigmoid(U H))

where ``H`` is the per-residue embedding tensor, ``V`` and ``U`` are
two attention-dim projection matrices, and ``(.)`` is element-wise
product. A separate vector ``w_k`` per GO term ``k`` yields one
attention map per term, matching the OPUS-GO recipe (Yang et al.,
bioRxiv 2024.12.17.629067) that targets per-term region attribution.

The softmax over residues is **mask-aware**: positions where the input
mask is ``False`` (padding) get ``-inf`` before the softmax, so their
attention is exactly ``0``. The unmasked positions sum to ``1`` for
every ``(batch, go_term)`` pair, which is the invariant the MIL.4
entropy and overlap features assume.

The torch import is guarded: a consumer who installs ``protea-method``
without the optional ``mil`` extra still imports the rest of the
package cleanly; touching ``protea_method.mil.head`` raises a clear
``ImportError`` pointing back at the install recipe.

References
----------
- Ilse, Tomczak, Welling. "Attention-based Deep Multiple Instance
  Learning". ICML, 2018.
- Yang et al. OPUS-GO. bioRxiv 2024.12.17.629067, 2024.
"""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover - import-time guard
    raise ImportError(
        "The MIL head requires torch. Install with: "
        'pip install "protea-method[mil]"'
    ) from exc


class GatedAttentionMILHead(nn.Module):
    """Gated-attention MIL head with per-GO-term residue attribution.

    Parameters
    ----------
    embedding_dim:
        Channel size ``D`` of the per-residue PLM embeddings. Matches
        the backend output (e.g. 1280 for ESM-2 t33, 1024 for ProtT5).
    n_go_terms:
        Number of GO terms ``K`` in the prediction head. The forward
        pass returns one logit and one attention map per term.
    attention_dim:
        Inner width of the gated-attention projection (``V`` and
        ``U``). 256 is the Ilse et al. default and keeps the head
        around ~1 M parameters for typical ``D``.

    Notes
    -----
    The head consumes ``frozen`` PLM features and is intended to be
    the only trainable block in MIL.2b. Backprop into the PLM is out
    of scope here; the training loop is expected to ``.detach()`` or
    pre-compute residue tensors.
    """

    def __init__(
        self,
        *,
        embedding_dim: int,
        n_go_terms: int,
        attention_dim: int = 256,
    ) -> None:
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if n_go_terms <= 0:
            raise ValueError("n_go_terms must be positive")
        if attention_dim <= 0:
            raise ValueError("attention_dim must be positive")

        self.embedding_dim = embedding_dim
        self.n_go_terms = n_go_terms
        self.attention_dim = attention_dim

        # Gated attention: two parallel D -> attention_dim projections,
        # combined element-wise (tanh gate * sigmoid gate), then mapped
        # to K attention scores via one vector per GO term.
        self.attention_V = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.attention_U = nn.Linear(embedding_dim, attention_dim, bias=False)
        # ``attention_w[:, k]`` is the per-term attention scoring vector.
        self.attention_w = nn.Linear(attention_dim, n_go_terms, bias=False)

        # Per-GO-term classifier on the pooled vector. Implemented as a
        # single (K, D) weight + (K,) bias so we can score all terms in
        # one matmul without materialising K separate Linear modules.
        self.classifier_weight = nn.Parameter(torch.empty(n_go_terms, embedding_dim))
        self.classifier_bias = nn.Parameter(torch.zeros(n_go_terms))
        nn.init.xavier_uniform_(self.classifier_weight)

    def forward(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the gated-attention pooling and per-term classifier.

        Parameters
        ----------
        embeddings:
            Per-residue PLM features, shape ``(B, L, D)``, float.
        mask:
            Boolean tensor of shape ``(B, L)``. ``True`` marks a real
            residue; ``False`` marks padding. Positions where
            ``mask=False`` receive ``-inf`` pre-softmax so their
            attention weight is exactly ``0``.

        Returns
        -------
        logits:
            ``(B, K)`` per-protein logits, one per GO term.
        attention:
            ``(B, L, K)`` per-residue attention weights. For every
            ``(b, k)`` pair, attention sums to ``1`` over the
            ``mask[b]`` positions and to ``0`` over padded positions.
        """
        self._validate_inputs(embeddings, mask)

        # Gated attention scores: (B, L, K).
        gate_tanh = torch.tanh(self.attention_V(embeddings))
        gate_sigmoid = torch.sigmoid(self.attention_U(embeddings))
        gated = gate_tanh * gate_sigmoid
        attention_logits = self.attention_w(gated)

        # Mask-aware softmax over the residue axis. Padding positions
        # become -inf so the softmax assigns them exactly 0 mass; the
        # unmasked positions then form a proper probability over
        # residues for every (batch, go_term) pair.
        mask_expanded = mask.unsqueeze(-1)  # (B, L, 1)
        attention_logits = attention_logits.masked_fill(
            ~mask_expanded,
            float("-inf"),
        )
        attention = torch.softmax(attention_logits, dim=1)  # (B, L, K)

        # Per-term pooled representation z[b, k] = sum_l A[b, l, k] * H[b, l].
        # Computed as (B, K, L) @ (B, L, D) -> (B, K, D) via einsum.
        pooled = torch.einsum("blk,bld->bkd", attention, embeddings)

        # Per-term logit: dot the pooled vector with the classifier row
        # for that term, add the term bias. Result shape (B, K).
        logits = torch.einsum("bkd,kd->bk", pooled, self.classifier_weight)
        logits = logits + self.classifier_bias

        return logits, attention

    def _validate_inputs(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Cheap shape and dtype checks. Run once per forward."""
        if embeddings.dim() != 3:
            raise ValueError(
                f"embeddings must be (B, L, D); got shape {tuple(embeddings.shape)}"
            )
        if embeddings.shape[-1] != self.embedding_dim:
            raise ValueError(
                f"embeddings last dim {embeddings.shape[-1]} != "
                f"embedding_dim {self.embedding_dim}"
            )
        if mask.dim() != 2:
            raise ValueError(
                f"mask must be (B, L); got shape {tuple(mask.shape)}"
            )
        if mask.shape != embeddings.shape[:2]:
            raise ValueError(
                f"mask shape {tuple(mask.shape)} does not match "
                f"embeddings batch/length {tuple(embeddings.shape[:2])}"
            )
        if mask.dtype != torch.bool:
            raise ValueError(f"mask must be bool; got dtype {mask.dtype}")


__all__ = ["GatedAttentionMILHead"]
