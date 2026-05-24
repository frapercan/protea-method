"""Round-trip stability for the public dataclasses on protea-method's surface.

protea-method does not own pydantic payloads (those live in
:mod:`protea_contracts`), but it does ship two frozen dataclasses that
LAFA / runner callers construct from JSON or YAML config files:
:class:`protea_method.PredictConfig` and
:class:`protea_method.embed.backend.BackendSpec`. Both must round-trip
through :func:`dataclasses.asdict` + ``cls(**dict)`` so a config that
was saved by the caller can be re-loaded without field loss.

The round-trip is also the contract the LAFA container's CLI relies
on: the CLI parses YAML, builds the dataclass, runs the pipeline, and
stores a copy of the asdict() form alongside the predictions for
provenance.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from protea_method import PredictConfig
from protea_method.embed.backend import BACKEND_IDS, BackendSpec, resolve_backend


def test_predict_config_default_roundtrip() -> None:
    """``PredictConfig`` with all defaults must round-trip via ``asdict``.

    Catches drift in the default values (e.g., a field gains a new
    default factory whose output is not equality-stable through
    ``dataclasses.asdict``).
    """
    instance = PredictConfig()
    dumped = dataclasses.asdict(instance)
    rebuilt = PredictConfig(**dumped)
    assert rebuilt == instance, (
        "PredictConfig default round-trip failed; a field default may "
        "have drifted out of sync with the asdict() representation."
    )


def test_predict_config_custom_roundtrip() -> None:
    """``PredictConfig`` with caller-set values must round-trip."""
    instance = PredictConfig(
        k=10,
        metric="l2",
        backend="faiss",
        distance_threshold=0.7,
        aspect_separated=True,
        compute_v6_features=False,
        compute_taxonomy=True,
        pre_normalized=True,
        prediction_set_id="test-set-001",
        extra={"trace": True},
    )
    dumped = dataclasses.asdict(instance)
    rebuilt = PredictConfig(**dumped)
    assert rebuilt == instance


def test_predict_config_extra_dict_isolation() -> None:
    """``PredictConfig.extra`` default factory yields independent dicts.

    A shared mutable default would cause two PredictConfig instances
    to share a single ``extra`` dict, a classic dataclass footgun.
    """
    a = PredictConfig()
    b = PredictConfig()
    a.extra["leaked"] = True
    assert "leaked" not in b.extra, (
        "PredictConfig.extra shares state across instances; the "
        "default_factory contract is broken."
    )


@pytest.mark.parametrize("backend_id", sorted(BACKEND_IDS))
def test_backend_spec_roundtrip(backend_id: str) -> None:
    """Every registered ``BackendSpec`` must round-trip via ``asdict``.

    Catches drift in the spec dataclass shape (a new field added
    without updating the YAML-config parser on the LAFA side would
    show up here if the new field isn't pickled by asdict()).
    """
    spec = resolve_backend(backend_id)
    dumped = dataclasses.asdict(spec)
    rebuilt = BackendSpec(**dumped)
    assert rebuilt == spec


def test_backend_spec_all_fields_present_in_asdict() -> None:
    """``asdict`` output must carry every field declared on ``BackendSpec``.

    Pin-test against silent removal of a field; the LAFA config
    consumer relies on the full key set.
    """
    sample = next(iter(BACKEND_IDS.values()))
    dumped: dict[str, Any] = dataclasses.asdict(sample)
    declared = {f.name for f in dataclasses.fields(BackendSpec)}
    assert set(dumped) == declared, (
        f"BackendSpec asdict() keys diverge from declared fields: "
        f"dumped={sorted(dumped)!r}, declared={sorted(declared)!r}."
    )


def test_backend_ids_table_is_non_empty() -> None:
    """At least one backend must remain registered.

    Catches the case where ``BACKEND_IDS`` is accidentally cleared:
    LAFA would fail to resolve any backend with a confusing
    ``ValueError`` instead of a clear "no backends installed" message.
    """
    assert BACKEND_IDS, "BACKEND_IDS table is empty; LAFA cannot resolve any backend."
