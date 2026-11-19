"""check_smoothing_m.py 的測試。可直接用 python3 tests/report/test_check_smoothing_m.py 跑（不需要 pytest）。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from check_smoothing_m import (OUT_TXT, assert_trainval_only, estimate, load_trainval, mle,  # noqa: E402
                               mom, problem_counts)
from features_r1 import ITEM_M  # noqa: E402

_cache = {}


def _tv():
    if "tv" not in _cache:
        _cache["tv"] = load_trainval()
    return _cache["tv"]


def _simulate(a=2.0, b=1.0, n_problems=3000, n_each=30, seed=0):
    """已知 Beta(a, b) 先驗（m = a + b）的模擬題目，檢查估計器能還原 m。"""
    rng = np.random.default_rng(seed)
    p = rng.beta(a, b, n_problems)
    n = np.full(n_problems, float(n_each))
    return n, rng.binomial(n_each, p).astype(float)


def test_moments_recovers_known_m():
    n, k = _simulate()
    _, tau2, m = mom(n, k)
    assert tau2 > 0 and abs(m - 3.0) / 3.0 < 0.1, m


def test_mle_recovers_known_m():
    n, k = _simulate()
    pbar = k.sum() / n.sum()
    (a, b), ok = mle(n, k, (pbar * 5, (1 - pbar) * 5))
    assert ok and abs((a + b) - 3.0) / 3.0 < 0.1, (a, b)


def test_moments_nan_when_no_difficulty_spread():
    """所有題目答對率相同（只有抽樣變異）時 tau2 <= 0，m 不可估 → NaN，不會丟出錯誤。"""
    n = np.full(50, 100.0)
    k = np.full(50, 60.0)
    assert np.isnan(mom(n, k)[2])


def test_load_returns_only_trainval():
    tv = _tv()
    assert len(tv) == 30516 and (tv["role"] == "trainval").all() and tv["fold"].between(0, 4).all()
    assert set(tv["skill_id"]) == {"47", "311"}


def test_rejects_test_rows():
    bad = _tv().head(10).copy()
    bad.loc[bad.index[0], ["role", "fold"]] = ["test", -1]
    for fn in (assert_trainval_only, estimate):
        try:
            fn(bad)
        except AssertionError:
            continue
        raise RuntimeError(f"{fn.__name__} 沒有擋下 test 列")


def test_estimate_matches_adopted_m_and_file():
    overall, per_fold = estimate(_tv())
    assert overall["m_adopted"].tolist() == [ITEM_M["47"], ITEM_M["311"]]
    assert overall["mle_converged"].all() and len(per_fold) == 5
    # method of moments 的整體估計四捨五入後要等於目前採用的 m（只回報，不改 m）
    assert [int(round(x)) for x in overall["m_moments"]] == [5, 20]
    # 每題計數的總和 = 該技能列數
    for s in ("47", "311"):
        n, _ = problem_counts(_tv()[_tv()["skill_id"] == s])
        assert n.sum() == (_tv()["skill_id"] == s).sum()


def test_output_text_has_only_aggregates():
    assert os.path.exists(OUT_TXT)
    text = open(OUT_TXT, encoding="utf-8").read()
    for bad in ("user_id", "order_id", "problem_id", "school_id"):
        assert bad not in text, bad
    assert "30,516" in text and "不修改 m" in text


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} 個測試全過")
