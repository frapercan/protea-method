"""Tests for the MIL.2a gated-attention head.

Skipped automatically when torch is not installed (i.e. when the
``mil`` extra was not selected at install time). The default
``protea-method`` install path is torch-free; CI runs this file under
``poetry install --extras mil``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from protea_method.mil.head import GatedAttentionMILHead  # noqa: E402


def test_forward_shapes() -> None:
    """Full-true mask: forward returns (B, K) logits and (B, L, K) attention."""
    torch.manual_seed(0)
    batch, length, dim = 2, 10, 64
    n_go = 5
    head = GatedAttentionMILHead(
        embedding_dim=dim,
        n_go_terms=n_go,
        attention_dim=32,
    )
    embeddings = torch.randn(batch, length, dim)
    mask = torch.ones(batch, length, dtype=torch.bool)

    logits, attention = head(embeddings, mask)

    assert logits.shape == (batch, n_go)
    assert attention.shape == (batch, length, n_go)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(attention).all()

    # Full mask: attention is a proper distribution over residues for
    # every (batch, go_term) pair.
    sums = attention.sum(dim=1)  # (B, K)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_partial_mask() -> None:
    """Padded positions get zero attention; unmasked positions sum to 1."""
    torch.manual_seed(0)
    batch, length, dim = 2, 10, 64
    n_go = 3
    head = GatedAttentionMILHead(
        embedding_dim=dim,
        n_go_terms=n_go,
        attention_dim=16,
    )
    embeddings = torch.randn(batch, length, dim)

    # First half of every sequence is "real"; second half is padding.
    half = length // 2
    mask = torch.zeros(batch, length, dtype=torch.bool)
    mask[:, :half] = True

    _, attention = head(embeddings, mask)

    # Masked positions get exactly zero mass for every (batch, term).
    masked_slice = attention[:, half:, :]
    assert torch.equal(masked_slice, torch.zeros_like(masked_slice))

    # Unmasked positions sum to 1 for every (batch, term).
    unmasked_sums = attention[:, :half, :].sum(dim=1)  # (B, K)
    assert torch.allclose(
        unmasked_sums,
        torch.ones_like(unmasked_sums),
        atol=1e-5,
    )
