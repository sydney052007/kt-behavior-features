"""RandomForest 5-fold 的測試。可直接用 python3 tests/test_rf_cv.py 跑。

為了速度，需要真的跑 RF 的測試只用 R0d 與 R1 各一組超參數（不是縮小正式的網格）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cv import run_cv  # noqa: E402
from evaluate import per_fold_auc  # noqa: E402
from features_r1 import load_r1_frame  # noqa: E402
from run_ablation_lr import caching_builder  # noqa: E402
from run_rf import FEATURE_SETS, GRID, make_rf  # noqa: E402

_cache = {}


def _data():
    if "df" not in _cache:
        df = load_r1_frame()
        assert (df["role"] == "test").sum() > 0
        _cache["df"] = df
        _cache["tv"] = df[df["role"] == "trainval"].reset_index(drop=True)
        _cache["builder"] = caching_builder()
    return _cache["df"], _cache["tv"], _cache["builder"]


def _run(name, cfg):
    key = (name, cfg)
    if key not in _cache:
        _, tv, builder = _data()
        _cache[key] = run_cv(tv, FEATURE_SETS[name], make_rf(*cfg), feature_builder=builder)
    return _cache[key]


def test_grid_has_exactly_nine_distinct_settings():
    assert len(GRID) == 9 and len(set(GRID)) == 9
    assert set(GRID) == {(d, l) for d in (8, 16, None) for l in (5, 20, 50)}
    rf = make_rf(16, 20)()
    p = rf.get_params()
    assert p["n_estimators"] == 300 and p["max_features"] == "sqrt" and p["random_state"] == 42
    assert p["n_jobs"] == -1 and p["max_depth"] == 16 and p["min_samples_leaf"] == 20
    assert set(FEATURE_SETS) == {"R0", "R0d", "R1"}


def test_train_val_students_disjoint_and_no_test_rows():
    df, tv, _ = _data()
    test_users = set(df.loc[df["role"] == "test", "user_id"])
    test_orders = set(df.loc[df["role"] == "test", "order_id"])
    res = _run("R1", (16, 20))
    for f, (tr, va) in res.fold_users.items():
        assert tr.isdisjoint(va), f"fold {f} 訓練與驗證學生有交集"
        assert tr.isdisjoint(test_users) and va.isdisjoint(test_users)
    assert test_orders.isdisjoint(res.oof["order_id"])
    assert len(res.oof) == len(tv) and res.oof["order_id"].is_unique
    try:
        run_cv(df, FEATURE_SETS["R1"], make_rf(16, 20), feature_builder=_data()[2])
    except AssertionError:
        pass
    else:
        raise AssertionError("run_cv 應該拒絕含 test 列的資料")


def test_three_feature_sets_use_same_folds_and_validation_rows():
    base = _run("R0", (16, 20))
    for name in ["R0d", "R1"]:
        res = _run(name, (16, 20))
        cols = ["order_id", "fold", "t", "user_id", "correct"]
        assert res.oof[cols].equals(base.oof[cols]), f"{name} 的驗證列與 R0 不同"
        for f in range(5):
            assert res.fold_users[f] == base.fold_users[f]


def test_same_seed_reproduces():
    _, tv, builder = _data()
    first = _run("R1", (8, 50))
    again = run_cv(tv, FEATURE_SETS["R1"], make_rf(8, 50), feature_builder=builder)
    # n_jobs=-1 時各棵樹的預測由多執行緒加總，順序不固定 → 浮點誤差約 1e-16（樹本身完全相同）；
    # 所以「可重現」= 在 1e-12 內相同，AUC 在 1e-9 內相同，而不是逐位元組相同
    assert first.oof[["order_id", "fold"]].equals(again.oof[["order_id", "fold"]])
    assert np.abs(first.oof["p_correct"].to_numpy() - again.oof["p_correct"].to_numpy()).max() < 1e-12, "同一種子重跑結果不同"
    assert all(abs(first.thresholds[f] - again.thresholds[f]) < 1e-12 for f in range(5))
    assert np.abs(per_fold_auc(first.oof) - per_fold_auc(again.oof)).max() < 1e-9
    other = run_cv(tv, FEATURE_SETS["R1"], make_rf(16, 20), feature_builder=builder)
    assert not first.oof["p_correct"].equals(other.oof["p_correct"])     # 不同設定結果不同（確認不是空轉）


if __name__ == "__main__":
    tests = [test_grid_has_exactly_nine_distinct_settings,
             test_train_val_students_disjoint_and_no_test_rows,
             test_three_feature_sets_use_same_folds_and_validation_rows,
             test_same_seed_reproduces]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
