import numpy as np
import pytest

from fqa.stats import bootstrap_ci, grouped_bootstrap_ci, paired_diff_ci


def right_or_wrong(n=150, p=0.6, seed=0):
    return (np.random.default_rng(seed).random(n) < p).astype(int)


def test_bootstrap_ci_contains_the_mean():
    x = right_or_wrong()
    mean, lo, hi = bootstrap_ci(x)
    assert mean == pytest.approx(x.mean())
    assert lo < mean < hi
    # about as wide as the normal approximation for a proportion
    se = np.sqrt(mean * (1 - mean) / len(x))
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.2)


def test_bootstrap_ci_is_reproducible_with_the_seed():
    x = np.random.default_rng(1).normal(size=150)
    assert bootstrap_ci(x, seed=3) == bootstrap_ci(x, seed=3)
    assert bootstrap_ci(x, seed=3) != bootstrap_ci(x, seed=4)
    assert bootstrap_ci(x, seed=3)[0] == bootstrap_ci(x, seed=4)[0]


def test_bootstrap_ci_alpha_and_edge_cases():
    x = right_or_wrong()
    _, lo95, hi95 = bootstrap_ci(x, alpha=0.05)
    _, lo50, hi50 = bootstrap_ci(x, alpha=0.5)
    assert lo95 <= lo50 < hi50 <= hi95
    assert bootstrap_ci([1, 1, 1]) == (1.0, 1.0, 1.0)
    assert bootstrap_ci([True, False])[0] == 0.5
    assert all(np.isnan(bootstrap_ci([])))
    with pytest.raises(ValueError):
        bootstrap_ci([[1, 0], [0, 1]])


def test_paired_diff_ci_resamples_both_configs_together():
    b = right_or_wrong(seed=1)
    a = b + 1
    # resampling a and b separately would scatter the difference around 1;
    # with the same questions in both it is exactly 1 every time
    assert paired_diff_ci(a, b) == pytest.approx((1.0, 1.0, 1.0))


def test_paired_diff_ci_finds_a_small_consistent_gain():
    b = right_or_wrong(seed=2)
    a = b.copy()
    a[np.flatnonzero(b == 0)[:10]] = 1  # a gets ten more questions right and none fewer
    diff, lo, hi = paired_diff_ci(a, b)
    assert diff == pytest.approx(10 / 150)
    assert 0 < lo <= diff <= hi
    # the two separate intervals overlap, so eyeballing them would call it a draw
    assert bootstrap_ci(a)[1] < bootstrap_ci(b)[2]


def test_paired_diff_ci_identical_and_mismatched():
    x = right_or_wrong()
    assert paired_diff_ci(x, x) == (0.0, 0.0, 0.0)
    assert paired_diff_ci(x, 1 - x, seed=5) == paired_diff_ci(x, 1 - x, seed=5)
    with pytest.raises(ValueError):
        paired_diff_ci(x, x[:-1])


def test_grouped_bootstrap_ci_resamples_whole_questions():
    x = right_or_wrong(n=60)
    # the same 60 answers again for 15 configs and models say no more than one copy does,
    # so the interval should be the one for 60 questions, not the far narrower one for 900 answers
    many, ids = np.tile(x, 15), np.tile(np.arange(60), 15)
    assert grouped_bootstrap_ci(many, ids) == pytest.approx(bootstrap_ci(x))
    _, lo, hi = bootstrap_ci(many)
    assert hi - lo < 0.6 * (bootstrap_ci(x)[2] - bootstrap_ci(x)[1])


def test_grouped_bootstrap_ci_keeps_the_pooled_mean():
    x, groups = [1, 1, 0, 1, 0, 0], ["a", "a", "a", "b", "c", "c"]
    mean, lo, hi = grouped_bootstrap_ci(x, groups)
    assert mean == 0.5  # 3 of the 6 answers, where the three questions' own means average 5/9
    assert lo <= mean <= hi
    assert all(np.isnan(grouped_bootstrap_ci([], [])))
    with pytest.raises(ValueError):
        grouped_bootstrap_ci([1, 0], ["a"])
