"""Percentile bootstrap intervals over questions.

Inputs are one number per question (1/0 for right/wrong, a reciprocal rank, a
latency). The paired version is the one to use for comparing two configs run
on the same questions: both get the same resampled questions, so the
question-to-question variation that the two share cancels out instead of
widening the interval. The grouped version is for several numbers per
question, such as one per config and model."""

import numpy as np


def _resample(n: int, B: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, (B, n))


def _as_vector(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"expected one value per question, got shape {x.shape}")
    return x


def bootstrap_ci(x, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi): the mean of x and a 1 - alpha percentile interval for it."""
    x = _as_vector(x)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    means = x[_resample(len(x), B, seed)].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)


def paired_diff_ci(a, b, B: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean(a) - mean(b), lo, hi) for two configs scored on the same questions,
    in the same order. Question indices are resampled once and applied to both."""
    a = _as_vector(a)
    b = _as_vector(b)
    if len(a) != len(b):
        raise ValueError(f"a has {len(a)} questions and b has {len(b)}; they must be the same questions")
    if len(a) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = _resample(len(a), B, seed)
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return float(a.mean() - b.mean()), float(lo), float(hi)


def grouped_bootstrap_ci(x, groups, B: int = 2000, seed: int = 0,
                         alpha: float = 0.05) -> tuple[float, float, float]:
    """(mean, lo, hi) for values that come several to a question, with groups[i] the
    question x[i] belongs to. Whole questions are resampled, each with all of its
    values: answers to the same question tend to be right or wrong together, so
    resampling the values one at a time would make the interval too narrow."""
    x = _as_vector(x)
    groups = np.asarray(groups)
    if len(groups) != len(x):
        raise ValueError(f"x has {len(x)} values and groups has {len(groups)}")
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    _, g = np.unique(groups, return_inverse=True)
    sums, counts = np.bincount(g, weights=x), np.bincount(g)
    idx = _resample(len(sums), B, seed)
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(lo), float(hi)
