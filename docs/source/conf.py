"""Sphinx configuration for protea-method."""

import os
import sys
import types

# Make the package importable from the source tree.
sys.path.insert(0, os.path.abspath("../../src"))
# method_main.py lives at the repo root (not inside src/), add it too.
sys.path.insert(0, os.path.abspath("../../"))

# ---------------------------------------------------------------------------
# Minimal stub for protea_contracts so that module-level code in
# protea_method that uses EMBEDDING_PCA_DIM in range() does not crash.
# autodoc_mock_imports replaces every listed module with MagicMock(), but
# MagicMock() is not an integer, which breaks `range(EMBEDDING_PCA_DIM)`.
# Providing a typed stub here takes precedence over the autodoc mock.
# ---------------------------------------------------------------------------
_contracts_stub = types.ModuleType("protea_contracts")
_contracts_stub.EMBEDDING_PCA_DIM = 16  # canonical value from the codebase
_contracts_stub.ALL_FEATURES = []
_contracts_stub.CATEGORICAL_FEATURES = []
_contracts_stub.NUMERIC_FEATURES = []
_contracts_stub.LABEL_COLUMN = "label"
sys.modules["protea_contracts"] = _contracts_stub

# ---------------------------------------------------------------------------
# Stub for protea_backends so autodoc can import embed.backend without
# requiring the optional extras ([esm] / [t5]) to be installed.
# ---------------------------------------------------------------------------
_backends_stub = types.ModuleType("protea_backends")
sys.modules["protea_backends"] = _backends_stub

project = "protea-method"
copyright = "2025, Francisco Miguel Perez Canales"
author = "Francisco Miguel Perez Canales"
release = "0.3.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
]

# Mock all heavy / optional imports so sphinx-build never fails on a
# missing dependency.  Add any new optional dep here.
# Note: protea_contracts and protea_backends are intentionally absent here;
# they are handled by the typed stubs above.
autodoc_mock_imports = [
    "lightgbm",
    "faiss",
    "numpy",
    "pandas",
    "torch",
    "transformers",
    "scipy",
    "sklearn",
    "pyarrow",
    "sentencepiece",
]

# autodoc defaults: include members without explicit __all__ filter,
# show inherited members, preserve source order.
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"

# Napoleon settings (Google and NumPy style docstrings).
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}

templates_path = ["_templates"]
exclude_patterns = []

html_theme = "alabaster"
html_title = "protea-method"
html_static_path = ["_static"]

master_doc = "index"

# Suppress duplicate-object warnings that arise from Python dataclass fields
# being indexed both as class attributes and in the __init__ signature when
# sphinx.ext.autodoc processes them with undoc-members.
suppress_warnings = ["ref.duplicate_label", "autodoc.duplicate"]
