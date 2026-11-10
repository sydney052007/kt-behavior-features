"""跨學校補充實驗（R0 vs R0d）的測試。可直接用 python3 tests/test_xschool_r0.py 跑。"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cv import run_cv  # noqa: E402
from evaluate import per_fold_metric  # noqa: E402
from features import R0_FEATURES  # noqa: E402
from features_r1 import R0D_FEATURES, load_r1_frame  # noqa: E402
from run_r1_lr import make_model as make_lr  # noqa: E402
from run_xschool_r0 import HIGH, LOW, classify, student_split_reference  # noqa: E402
from xschool import load_xschool_frame  # noqa: E402

_cache = {}


def _x():
    if "x" not in _cache:
        _cache["x"] = load_xschool_frame()
    return _cache["x"]


def test_r0_features_exclude_item_diff_and_last_correct():
    assert R0_FEATURES == ["log1p_c", "log1p_w", "skill_311", "has_history"]
    for c in ("item_diff", "last_correct"):
        assert c not in R0_FEATURES
    assert set(R0D_FEATURES) - set(R0_FEATURES) == {"item_diff", "last_correct"}    # R0d 只多這兩個


def test_groups_disjoint_and_no_test_students():
    x = _x()
    full = load_r1_frame()
    test_users = set(full.loc[full["role"] == "test", "user_id"])
    assert len(test_users) == 262 and test_users.isdisjoint(set(x["user_id"]))
    assert (x["role"] == "trainval").all()
    res = run_cv(x, R0_FEATURES, make_lr)
    _cache["res"] = res
    for f, (tr, va) in res.fold_users.items():
        assert tr.isdisjoint(va) and tr.isdisjoint(test_users) and va.isdisjoint(test_users)
        g_tr, g_va = set(x.loc[x["fold"] != f, "grp"]), set(x.loc[x["fold"] == f, "grp"])
        assert g_tr.isdisjoint(g_va), f"fold {f} 訓練與驗證的組別有交集"
    assert len(res.oof) == len(x) and res.oof["order_id"].is_unique           # 覆蓋每列一次


def test_r0_under_student_folds_reproduces_existing_csv():
    full = load_r1_frame()
    tv = full[full["role"] == "trainval"].reset_index(drop=True)              # 原本的學生 fold
    res = run_cv(tv, R0_FEATURES, make_lr)
    prev = pd.read_csv("results/02_analysis/r1_lr_cv.csv")
    prev = prev[(prev["model"] == "R0") & (prev["eval"] == "A")].set_index("metric")
    roc, pr = per_fold_metric(res.oof), per_fold_metric(res.oof, "pr_auc")
    assert abs(roc.mean() - prev.at["roc_auc", "value"]) < 1e-9
    assert abs(roc.std(ddof=1) - prev.at["roc_auc", "std"]) < 1e-9
    assert abs(pr.mean() - prev.at["pr_auc", "value"]) < 1e-9
    # 也和最早的 R0 結果（沒有 StandardScaler）在小數點三位一致
    first = pd.read_csv("results/02_analysis/r0_lr_cv.csv")
    first = first[(first["eval"] == "A") & (first.metric == "roc_auc")]["value"].iloc[0]
    assert round(roc.mean(), 3) == round(first, 3)


def test_decision_rule_boundaries():
    assert (HIGH, LOW) == (0.06, 0.04)
    assert "可搬移" in classify(0.06) and "可搬移" in classify(0.10)
    assert "不下結論" in classify(0.0599) and "不下結論" in classify(0.04)
    assert "學校／班級效應" in classify(0.0399) and "學校／班級效應" in classify(-0.01)


def test_student_split_reference_is_computed_from_csv():
    ref = student_split_reference()
    assert abs((ref["r0d"] - ref["r0"]) - ref["diff_mean"]) < 1e-9
    assert np.isclose(ref["diff_folds"].mean(), ref["diff_mean"], atol=1e-12) and ref["n_better"] == 5


if __name__ == "__main__":
    tests = [test_r0_features_exclude_item_diff_and_last_correct, test_groups_disjoint_and_no_test_students,
             test_r0_under_student_folds_reproduces_existing_csv, test_decision_rule_boundaries,
             test_student_split_reference_is_computed_from_csv]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
