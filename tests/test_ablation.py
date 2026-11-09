"""LR ablation 的測試。可直接用 python3 tests/test_ablation.py 跑。"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import R0_FEATURES  # noqa: E402
from features_r1 import R0D_FEATURES, R1_FEATURES, load_r1_frame  # noqa: E402
from run_ablation_lr import COMPARISONS, FEATURE_SETS, run_sets  # noqa: E402

_cache = {}

# 依規格逐字寫出的期望欄位（獨立於 run_ablation_lr.py 的組裝方式）
EXPECTED = {
    "R0+item_diff": ["log1p_c", "log1p_w", "skill_311", "has_history", "item_diff"],
    "R0+last_correct": ["log1p_c", "log1p_w", "skill_311", "has_history", "last_correct"],
    "R0d+hint": R0D_FEATURES + ["hint_rate", "last_hint"],
    "R0d+time": R0D_FEATURES + ["mean_log_time", "last_log_time"],
    "R0d+att": R0D_FEATURES + ["mean_log_att", "last_log_att"],
    "R0d+scaffold": R0D_FEATURES + ["scaffold_rate"],
    "R0d+mean_only": R0D_FEATURES + ["hint_rate", "mean_log_time", "mean_log_att", "scaffold_rate"],
    "R0d+last_only": R0D_FEATURES + ["last_hint", "last_log_time", "last_log_att"],
    "R1-hint": [c for c in R1_FEATURES if c not in ("hint_rate", "last_hint")],
    "R1-time": [c for c in R1_FEATURES if c not in ("mean_log_time", "last_log_time")],
    "R1-att": [c for c in R1_FEATURES if c not in ("mean_log_att", "last_log_att")],
    "R1-scaffold": [c for c in R1_FEATURES if c != "scaffold_rate"],
}


def test_feature_set_column_names():
    assert FEATURE_SETS["R0"] == R0_FEATURES and FEATURE_SETS["R0d"] == R0D_FEATURES
    assert FEATURE_SETS["R1"] == R1_FEATURES
    assert len(FEATURE_SETS) == 15
    for name, cols in EXPECTED.items():
        assert set(FEATURE_SETS[name]) == set(cols), f"{name} 欄位不符規格"
        assert len(FEATURE_SETS[name]) == len(set(FEATURE_SETS[name])), f"{name} 有重複欄位"
    for cols in FEATURE_SETS.values():
        assert set(cols) <= set(R1_FEATURES)         # 沒有目標列自己的行為欄位或其他不明欄位
    # 具體數量：加法 5 / 5 / 8 / 8 / 8 / 7 / 10 / 9；減法 11 / 11 / 11 / 12
    sizes = {n: len(c) for n, c in FEATURE_SETS.items()}
    assert sizes["R0+item_diff"] == 5 and sizes["R0d+mean_only"] == 10 and sizes["R1-scaffold"] == 12
    # 每個比較的兩組都存在，且其中一組嚴格包含另一組
    # （例如 R0+item_diff 對 R0 是加了一個特徵；對 R0d 則是少了 last_correct）
    for new, ref in COMPARISONS:
        assert new in FEATURE_SETS and ref in FEATURE_SETS
        a, b = set(FEATURE_SETS[new]), set(FEATURE_SETS[ref])
        assert a < b or b < a, f"{new} 與 {ref} 沒有包含關係"
        if "-" in new:
            assert a < b                                  # 減法 ablation 是 R1 的子集
    assert len(COMPARISONS) == 14


def _runs():
    if "runs" not in _cache:
        df = load_r1_frame()
        assert (df["role"] == "test").sum() > 0
        tv = df[df["role"] == "trainval"].reset_index(drop=True)
        names = ["R0", "R0d", "R0+item_diff", "R0d+mean_only", "R1-time", "R1"]
        _cache["tv"], _cache["df"] = tv, df
        _cache["runs"] = run_sets(tv, names)
    return _cache["df"], _cache["tv"], _cache["runs"]


def test_no_test_rows():
    df, tv, runs = _runs()
    test_orders = set(df.loc[df["role"] == "test", "order_id"])
    for name, (res, _) in runs.items():
        assert test_orders.isdisjoint(res.oof["order_id"]), f"{name} 含 test 列"
        test_users = set(df.loc[df["role"] == "test", "user_id"])
        for tr, va in res.fold_users.values():
            assert tr.isdisjoint(test_users) and va.isdisjoint(test_users)


def test_same_folds_and_validation_rows_across_sets():
    df, tv, runs = _runs()
    base = runs["R0"][0]
    for name, (res, _) in runs.items():
        assert res.oof[["order_id", "fold", "t", "user_id"]].equals(base.oof[["order_id", "fold", "t", "user_id"]]), name
        assert res.oof["correct"].equals(base.oof["correct"]), name
        for f in range(5):
            assert res.fold_users[f] == base.fold_users[f], f"{name} fold {f} 的訓練／驗證學生不同"
    assert len(base.oof) == len(tv) and base.oof["order_id"].is_unique


if __name__ == "__main__":
    tests = [test_feature_set_column_names, test_no_test_rows, test_same_folds_and_validation_rows_across_sets]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
