"""Releasing the corpus tensor between torch searches.

Its own module because ``knn_search`` is at the file-size ceiling. That is the
guard doing its job: the file is full, and adding to it is the change that should
be resisted rather than the one that should be waved through.
"""

from __future__ import annotations

from typing import Any

__all__ = ["_released"]

def _released(results: list, R_t: Any, device: Any) -> list:
    """Hand back the results, freeing the corpus tensor first when it was on GPU.

    Written to return rather than to be called on its own line, so the port costs
    _search_torch no lines. That function is already 98 lines against a ceiling of
    60 and sits in the smell baseline as a known offender, and the right fix for
    that is to extract its chunk loop. This is not the change to do it in: it is
    the function the OOM cursor fix just landed in, on the branch the rung 1
    recompute runs from, and a restructure does not belong inside a one-symbol
    port.

    Without this, looping ``_search_torch`` across the three GO aspects pins
    about 10 GB on a 12 GB card, and the corpus-fits-in-VRAM check inside
    ``_torch_target_device`` then flips the device back to CPU for the rest of
    the run. The run completes, slower, and says nothing: the only symptom is
    that the second and third aspects did not use the card the first one did.

    Discovered 2026-05-27 on an RTX 3060 with ankh-large, 1536 dimensions over
    527k proteins, at 3.2 GB per aspect copy.

    The campaign no longer reaches this path, because PROTEA pins
    ``PROTEA_KNN_DEVICE`` to cpu rather than letting it resolve to "auto". It is
    still worth carrying, for the reason the pin exists: "auto" resolves to CUDA
    whenever a card is visible, so any caller reaching this library without that
    pin gets the leak.
    """
    if device.type == "cuda":
        import torch as _torch

        del R_t
        _torch.cuda.empty_cache()
    return results
