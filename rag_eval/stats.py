"""
Bootstrap confidence intervals.

With ~1,200 questions a 1–2 point difference between two retrievers can be
noise. Because every retriever answers the *same* questions, comparisons use
a paired bootstrap: resample questions, recompute both scores on the same
sample, and look at the distribution of the difference.
"""

import numpy as np

N_BOOT = 2000
SEED = 0


def hits(first_ranks, k: int) -> np.ndarray:
    """Per-question 0/1 hit@k from the rank of the first relevant chunk (0 = not found)."""
    r = np.asarray(first_ranks)
    return ((r > 0) & (r <= k)).astype(float)


def ci(values, n_boot: int = N_BOOT, seed: int = SEED) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(a, b, n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    """Mean of (a - b) with a 95% CI; 'significant' if the CI excludes 0."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    lo, hi = ci(d, n_boot, seed)
    return {"diff": float(d.mean()), "ci_low": lo, "ci_high": hi, "significant": bool(lo > 0 or hi < 0)}
