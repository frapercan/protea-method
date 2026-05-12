"""Multiple Instance Learning extensions for region-attributable GO prediction.

The MIL submodule packages the gated-attention head from MIL.2 plus
upcoming training/integration code (MIL.2b, MIL.2c). It depends on
``torch``, which ships behind the optional ``mil`` extra so the default
``protea-method`` install path stays slim (see README).

Importing this package is cheap; importing ``protea_method.mil.head``
triggers the torch import. Consumers who never touch the MIL head can
keep using the rest of ``protea_method`` torch-free.
"""

from __future__ import annotations

__all__: list[str] = []
