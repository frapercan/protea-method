"""Integration test for the ``method_main`` LAFA entrypoint."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd

FIXTURES = Path(__file__).parent / "fixtures"
REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_method_main() -> object:
    """Import ``method_main`` from the repo root as a module.

    The script lives next to ``pyproject.toml`` (not in ``src/``), so
    pytest's default rootdir-based path injection does not find it.
    """
    spec = importlib.util.spec_from_file_location(
        "method_main", REPO_ROOT / "method_main.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["method_main"] = module
    spec.loader.exec_module(module)
    return module


def _embedding_parquet(path: Path, accessions: list[str], dim: int = 4) -> None:
    rng = np.random.default_rng(0)
    rows = []
    for acc in accessions:
        rows.append({"accession": acc, "embedding": rng.standard_normal(dim).astype(np.float32).tolist()})
    pd.DataFrame(rows).to_parquet(path)


def test_method_main_end_to_end_writes_tsv(tmp_path: Path) -> None:
    module = _load_method_main()
    query_embeds = tmp_path / "q.parquet"
    reference_embeds = tmp_path / "r.parquet"
    _embedding_parquet(query_embeds, ["P00001", "P00002", "P00003"])
    _embedding_parquet(reference_embeds, ["P00001", "P00002", "P00003"])
    output = tmp_path / "preds.tsv"

    rc = module.main([  # type: ignore[attr-defined]
        "--query_file", str(FIXTURES / "tiny.fasta"),
        "--train_sequences", str(FIXTURES / "tiny.fasta"),
        "--annot_file", str(FIXTURES / "tiny.gaf"),
        "--graph", str(FIXTURES / "tiny.obo"),
        "--output_file", str(output),
        "--query_embeds", str(query_embeds),
        "--reference_embeds", str(reference_embeds),
        "--top_k", "2",
    ])
    assert rc == 0
    assert output.exists()
    lines = output.read_text().splitlines()
    assert lines, "expected at least one prediction line"
    for line in lines:
        parts = line.split("\t")
        assert len(parts) == 3
        assert parts[0].startswith("P000")
        assert parts[1].startswith("GO:")


def test_method_main_errors_when_input_missing(tmp_path: Path) -> None:
    module = _load_method_main()
    rc = module.main([  # type: ignore[attr-defined]
        "--query_file", str(tmp_path / "missing.fasta"),
        "--train_sequences", str(FIXTURES / "tiny.fasta"),
        "--annot_file", str(FIXTURES / "tiny.gaf"),
        "--graph", str(FIXTURES / "tiny.obo"),
        "--output_file", str(tmp_path / "out.tsv"),
    ])
    assert rc == 2


def test_method_main_self_contained_via_mock_backend(tmp_path: Path) -> None:
    """LAFA-EMB.1: omitting embeddings triggers the in-container embedder."""
    module = _load_method_main()
    output = tmp_path / "preds.tsv"
    cache_dir = tmp_path / "embed-cache"
    rc = module.main([  # type: ignore[attr-defined]
        "--query_file", str(FIXTURES / "tiny.fasta"),
        "--train_sequences", str(FIXTURES / "tiny.fasta"),
        "--annot_file", str(FIXTURES / "tiny.gaf"),
        "--graph", str(FIXTURES / "tiny.obo"),
        "--output_file", str(output),
        "--backend_id", "mock_constant",
        "--embed_cache_dir", str(cache_dir),
        "--top_k", "2",
    ])
    assert rc == 0
    assert output.exists()
    # Cache miss on first run should have persisted .npz + .acc.txt under
    # the mock_constant prefix.
    cached = list(cache_dir.rglob("*.npz"))
    assert cached, "expected the mock backend to populate the cache"


def test_method_main_errors_when_only_one_embeds_given(tmp_path: Path) -> None:
    """Supplying only one of the two embed flags must be a clean error."""
    module = _load_method_main()
    rc = module.main([  # type: ignore[attr-defined]
        "--query_file", str(FIXTURES / "tiny.fasta"),
        "--train_sequences", str(FIXTURES / "tiny.fasta"),
        "--annot_file", str(FIXTURES / "tiny.gaf"),
        "--graph", str(FIXTURES / "tiny.obo"),
        "--output_file", str(tmp_path / "out.tsv"),
        "--query_embeds", str(tmp_path / "q.parquet"),
    ])
    assert rc == 2
