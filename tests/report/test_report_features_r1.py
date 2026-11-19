"""report_features_r1.py 的測試。可直接用 python3 tests/report/test_report_features_r1.py 跑（不需要 pytest）。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from features import OUT_PATH  # noqa: E402
from features_r1 import R1_FEATURES  # noqa: E402
from report_features_r1 import OUT_CSV, assert_trainval_only, build_table, load_trainval  # noqa: E402

_cache = {}


def _tv():
    if "tv" not in _cache:
        _cache["tv"] = load_trainval()
    return _cache["tv"]


def _csv():
    return pd.read_csv(OUT_CSV, float_precision="round_trip")


def test_thirteen_features_in_order_and_no_ids():
    d = _csv()
    assert d["feature"].tolist() == R1_FEATURES and len(d) == 13
    for bad in ("user_id", "school_id", "order_id", "problem_id", "p_correct"):
        assert bad not in d.columns, bad
    assert (d["n_rows"] == 30516).all() and (d["n_missing_after_fill"] == 0).all()


def test_load_only_trainval_and_rejects_test():
    tv = _tv()
    assert len(tv) == 30516 and (tv["role"] == "trainval").all()
    bad = tv.head(10).copy()
    bad.loc[bad.index[0], ["role", "fold"]] = ["test", -1]
    for fn in (assert_trainval_only, build_table):
        try:
            fn(bad)
        except AssertionError:
            continue
        raise RuntimeError(f"{fn.__name__} 沒有擋下 test 列")


def test_csv_equals_fresh_build():
    fresh, saved = build_table(_tv()), _csv()
    num = fresh.select_dtypes("number").columns
    assert np.allclose(fresh[num].to_numpy(float), saved[num].to_numpy(float), rtol=0, atol=1e-12)
    assert fresh["feature"].tolist() == saved["feature"].tolist()


def test_independent_numbers_from_features_r0():
    """skill_311、has_history 的平均與 t=1 的列數，直接從 features_r0.csv（只取 trainval）獨立算一次。"""
    f = pd.read_csv(OUT_PATH, usecols=["role", "skill_311", "has_history", "t"])
    f = f[f["role"] == "trainval"]
    d = _csv().set_index("feature")
    assert abs(d.loc["skill_311", "mean"] - f["skill_311"].mean()) < 1e-12
    assert abs(d.loc["has_history", "mean"] - f["has_history"].mean()) < 1e-12
    n_t1 = int((f["t"] == 1).sum())
    for name in ("last_correct", "hint_rate", "last_hint", "mean_log_time", "last_log_time",
                 "mean_log_att", "last_log_att", "scaffold_rate"):
        assert d.loc[name, "n_filled_or_default"] == n_t1, name
    assert d.loc["has_history", "n_filled_or_default"] == 0 and d.loc["log1p_c", "n_filled_or_default"] == 0


def test_ranges_are_sensible():
    d = _csv().set_index("feature")
    for name in ("skill_311", "has_history", "last_correct", "last_hint", "hint_rate", "scaffold_rate", "item_diff"):
        assert 0 <= d.loc[name, "min"] and d.loc[name, "max"] <= 1, name
    assert d.loc["log1p_c", "min"] == 0 and d.loc["log1p_w", "min"] == 0
    assert (d["std"] > 0).all()


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"{len(tests)} 個測試全過")
