"""The same bank must return the same donors however the machine reduces.

WHY THIS EXISTS. On 2026-09-08 the same axis C baseline, same configuration and
same code, predicted twice, differed by 82 and -73 rows out of ten million. The
cause was not the code: every job splits its batches over two machines, OpenBLAS
is built DYNAMIC_ARCH and picks its micro-kernel and thread partition from the
detected CPU, so ``1 - Q@R.T`` reduces in a different order on each host. 838,885
rows carried the same donor at a different distance, every delta an exact
multiple of 2^-24, up to 13 ulps. Where the k-th distance ties, one ulp decides
which donor is admitted.

WHY THE TEST LOOKS LIKE THIS. The property is invariance to the reduction order,
and the only handle on that order from a test is the thread count, which OpenBLAS
reads once at import. So each reading runs in its own process. The sizes are the
smallest that were measured to discriminate: at 64x2000x256 the float32 product
already differs between one thread and eight. A smaller bench does not exercise
the multithreaded path at all and would pass against the very code this pins --
which is exactly the trap this campaign kept falling into, a check whose silence
had never been shown to be refusable.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

#: Measured to discriminate: the float32 product differs between 1 and 8 threads
#: at this size. Raising it costs time; lowering it risks a test that cannot fail.
#: The bank that makes the property observable. Random gaussian vectors do NOT
#: tie, so the ulp noise never reaches the boundary and a test built on them
#: passes against the very code it is meant to pin -- verified, not assumed.
#: What the real bank has and random data does not is families of near-identical
#: sequences (ubiquitin, histone paralogs) sitting exactly where the cut falls.
#: This reproduces that: fifty near-twins separated by 1e-7, which float32
#: cannot resolve and the reduction order therefore decides.
PROBE = """
import hashlib, numpy as np
from protea_method.knn_search import search_knn
rng = np.random.default_rng(5)
D, NR, K = 512, 4000, 60
R = rng.standard_normal((NR, D), dtype=np.float32)
base = R[100].copy()
for i in range(40, 90):
    R[i] = base + rng.standard_normal(D).astype(np.float32) * 1e-7
Q = R[40:44] + rng.standard_normal((4, D)).astype(np.float32) * 1e-6
acc = ["A%05d" % i for i in range(NR)]
hits = search_knn(Q, R, acc, K, backend="numpy", metric={metric!r})
donors = [[a for a, _ in row] for row in hits]
print(hashlib.sha256(repr(donors).encode()).hexdigest())
"""


def _donors_at(threads: str, *, metric: str = "cosine") -> str:
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS=threads,
        OMP_NUM_THREADS=threads,
        OPENBLAS_CORETYPE="HASWELL",
    )
    out = subprocess.run(
        [sys.executable, "-c", PROBE.format(metric=metric)],
        capture_output=True, text=True, env=env, timeout=300,
    )
    assert out.returncode == 0, out.stderr[-2000:]
    return out.stdout.strip()


@pytest.mark.parametrize("metric", ["cosine", "l2"])
def test_the_same_donors_come_back_however_many_threads_reduce(metric: str) -> None:
    """One thread and eight must admit the same k, not merely a similar k."""
    seen = {_donors_at(t, metric=metric) for t in ("1", "8", "16")}
    assert len(seen) == 1, (
        f"{metric}: the donor set depends on the thread partition, so two "
        "machines with different CPUs disagree about who the k-th neighbour is"
    )


def test_the_bank_makes_the_property_observable() -> None:
    """Guard on the guard: prove these data CAN expose the defect.

    Runs the pre-fix selection -- float32 accumulation, exactly what the live
    code did until 2026-09-08 -- over the same bank, and asserts that IT does
    disagree across thread counts. Without this the test above is a check whose
    silence has never been shown to be refusable, which is the failure mode this
    campaign hit three times in one week: a first attempt at this very test used
    random vectors, passed against the unfixed code, and proved nothing.
    """
    code = """
import hashlib, numpy as np
rng = np.random.default_rng(5)
D, NR, K = 512, 4000, 60
R = rng.standard_normal((NR, D), dtype=np.float32)
base = R[100].copy()
for i in range(40, 90):
    R[i] = base + rng.standard_normal(D).astype(np.float32) * 1e-7
Q = R[40:44] + rng.standard_normal((4, D)).astype(np.float32) * 1e-6
Rn = R / (np.linalg.norm(R, axis=1, keepdims=True) + 1e-9)
Qn = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-9)
dist = 1.0 - (Qn @ Rn.T)
part = np.argpartition(dist, K - 1, axis=1)[:, :K]
rows = np.arange(dist.shape[0])[:, None]
top = part[rows, np.argsort(dist[rows, part], axis=1)]
print(hashlib.sha256(repr(top.tolist()).encode()).hexdigest())
"""
    seen = set()
    for threads in ("1", "8", "16"):
        env = dict(os.environ, OPENBLAS_NUM_THREADS=threads, OMP_NUM_THREADS=threads,
                   OPENBLAS_CORETYPE="HASWELL")
        r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, env=env, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        seen.add(r.stdout.strip())
    assert len(seen) > 1, (
        "float32 accumulation no longer disagrees across thread counts on this "
        "bank, so the invariance test above can pass without meaning anything"
    )


def test_the_result_is_the_exact_float64_answer() -> None:
    """Invariance is not enough: it must be invariant to the RIGHT value."""
    import numpy as np

    from protea_method.knn_search import search_knn

    rng = np.random.default_rng(7)
    Q = rng.standard_normal((40, 128), dtype=np.float32)
    R = rng.standard_normal((3000, 128), dtype=np.float32)
    acc = [f"A{i:05d}" for i in range(3000)]

    got = [[a for a, _ in row] for row in
           search_knn(Q, R, acc, 50, backend="numpy", metric="cosine")]

    Qn = (Q / np.linalg.norm(Q, axis=1, keepdims=True)).astype(np.float64)
    Rn = (R / np.linalg.norm(R, axis=1, keepdims=True)).astype(np.float64)
    exact = 1.0 - Qn @ Rn.T
    want = [[acc[j] for j in np.argsort(exact[i], kind="stable")[:50]] for i in range(40)]
    assert got == want


class TestTheUncoveredBackendsSayThatTheyAre:
    """The fix must not stop applying in silence.

    Every prediction set stored to date used ``numpy``, so fixing that path
    covers everything measured. But ``search_backend`` DEFAULTS to ``faiss`` in
    five export and training payloads, so a dataset export or a reranker run
    falls onto the uncovered path without anybody choosing it -- and that is
    where donor identity becomes a feature, i.e. exactly where the ulp noise
    would be read as signal. A warning rather than a refusal, because those
    backends are legitimate for work that does not compare runs; what is not
    legitimate is not knowing.
    """

    def test_the_covered_set_names_numpy_and_only_numpy(self) -> None:
        from protea_method.knn_search import ORDER_INVARIANT_BACKENDS

        assert ORDER_INVARIANT_BACKENDS == frozenset({"numpy"}), (
            "the covered set changed; if a backend gained float64 accumulation "
            "say so here, and if one lost it this test is the alarm"
        )

    def test_the_invariant_backend_is_quiet(self) -> None:
        import warnings as w

        import numpy as np

        from protea_method.knn_search import search_knn

        Q = np.zeros((2, 4), dtype=np.float32)
        R = np.eye(4, dtype=np.float32)
        with w.catch_warnings(record=True) as caught:
            w.simplefilter("always")
            search_knn(Q, R, ["A", "B", "C", "D"], 2, backend="numpy")
        assert [c for c in caught if issubclass(c.category, RuntimeWarning)] == []

    def test_an_order_dependent_backend_says_so_once(self) -> None:
        import warnings as w

        import numpy as np

        import protea_method.knn_search as ks

        ks._WARNED_NON_INVARIANT.clear()
        Q = np.zeros((2, 4), dtype=np.float32)
        R = np.eye(4, dtype=np.float32)
        seen = []
        for _ in range(3):
            with w.catch_warnings(record=True) as caught:
                w.simplefilter("always")
                try:
                    ks.search_knn(Q, R, ["A", "B", "C", "D"], 2, backend="faiss")
                except Exception:
                    pass
                seen += [c for c in caught if issubclass(c.category, RuntimeWarning)]
        ks._WARNED_NON_INVARIANT.clear()
        assert len(seen) == 1, "once per process, not once per batch"
        assert "reduction order" in str(seen[0].message)

    def test_an_unknown_backend_is_still_refused_before_the_warning(self) -> None:
        """The warning must not swallow the existing refusal."""
        import numpy as np
        import pytest as pt

        from protea_method.knn_search import search_knn

        with pt.raises(ValueError, match="Unknown search backend"):
            search_knn(np.zeros((1, 2), dtype=np.float32),
                       np.eye(2, dtype=np.float32), ["A", "B"], 1, backend="inventado")
