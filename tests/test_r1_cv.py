"""R0 / R0d / R1 交叉驗證的測試。可直接用 python3 tests/test_r1_cv.py 跑。

用真實資料的 R1 特徵跑一次 run_cv（每個 fold 內各算一次特徵）。
"""
import os
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cv import run_cv  # noqa: E402
from features_r1 import R1_FEATURES, load_r1_frame, make_fold_features  # noqa: E402

_cache = {}


def _model():
    return Pipeline([("scale", StandardScaler()), ("lr", LogisticRegression(C=np.inf, max_iter=1000))])


def _all_rows():
    if "all" not in _cache:
        _cache["all"] = load_r1_frame()
    return _cache["all"]


def _tv():
    return _all_rows().query("role == 'trainval'").reset_index(drop=True)


def _cv():
    if "cv" not in _cache:
        tv = _tv()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _cache["cv"] = (tv, run_cv(tv, R1_FEATURES, _model, feature_builder=make_fold_features))
    return _cache["cv"]


def test_train_val_students_disjoint_within_each_fold():
    tv, res = _cv()
    assert len(res.fold_users) == 5
    for f, (tr, va) in res.fold_users.items():
        assert tr.isdisjoint(va), f"fold {f} 訓練與驗證學生有交集"
        assert va == set(tv.loc[tv["fold"] == f, "user_id"])
        assert tr == set(tv.loc[tv["fold"] != f, "user_id"])


def test_no_test_rows_anywhere():
    allrows = _all_rows()
    test_users = set(allrows.loc[allrows["role"] == "test", "user_id"])
    test_orders = set(allrows.loc[allrows["role"] == "test", "order_id"])
    assert test_users and test_orders
    _, res = _cv()
    for tr, va in res.fold_users.values():
        assert tr.isdisjoint(test_users) and va.isdisjoint(test_users)
    assert test_orders.isdisjoint(res.oof["order_id"])
    try:
        run_cv(allrows, R1_FEATURES, _model, feature_builder=make_fold_features)
    except AssertionError:
        pass
    else:
        raise AssertionError("run_cv 應該拒絕含 test 列的資料")


def test_threshold_uses_only_training_rows():
    """把 fold 0『驗證列』的 correct 全部改成亂數，fold 0 的閾值與模型係數必須完全不變
    （它們只由 fold 0 的訓練列決定）。"""
    tv, res = _cv()
    mod = tv.copy()
    rng = np.random.default_rng(0)
    m = mod["fold"] == 0
    mod.loc[m, "correct"] = rng.integers(0, 2, int(m.sum()))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res2 = run_cv(mod, R1_FEATURES, _model, feature_builder=make_fold_features)
    assert res2.thresholds[0] == res.thresholds[0]
    assert np.array_equal(res2.models[0].named_steps["lr"].coef_, res.models[0].named_steps["lr"].coef_)
    # 其他 fold 的訓練列含有被改過的 fold 0 列，所以它們的閾值應該不同（確認這個測試不是空轉）
    assert any(res2.thresholds[f] != res.thresholds[f] for f in range(1, 5))


def test_oof_covers_every_trainval_row_exactly_once():
    tv, res = _cv()
    assert len(res.oof) == len(tv) == 30516
    assert res.oof["order_id"].is_unique and set(res.oof["order_id"]) == set(tv["order_id"])
    assert res.oof["p_correct"].between(0, 1).all() and res.oof["p_correct"].notna().all()
    merged = res.oof.merge(tv[["order_id", "correct", "fold"]], on="order_id", suffixes=("", "_tv"))
    assert (merged["correct"] == merged["correct_tv"]).all() and (merged["fold"] == merged["fold_tv"]).all()


if __name__ == "__main__":
    tests = [test_train_val_students_disjoint_within_each_fold,
             test_no_test_rows_anywhere,
             test_threshold_uses_only_training_rows,
             test_oof_covers_every_trainval_row_exactly_once]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
