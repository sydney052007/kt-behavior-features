"""R0 + LR 交叉驗證與評估的測試。可直接用 python3 tests/test_r0_cv.py 跑。

測試 1、2、4 用真實的 features_r0.csv 跑一次 run_cv；3、5 用構造的小例子。
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from cv import run_cv  # noqa: E402
from evaluate import (_cluster_bootstrap_ci, cluster_resample_indices,  # noqa: E402
                      select_threshold)
from features import OUT_PATH, R0_FEATURES  # noqa: E402

_cache = {}


def _all_rows() -> pd.DataFrame:
    if "all" not in _cache:
        _cache["all"] = pd.read_csv(OUT_PATH, dtype={"skill_id": str})
    return _cache["all"]


def _cv():
    """用真實資料跑一次 5-fold（LR 很快），結果共用。"""
    if "cv" not in _cache:
        tv = _all_rows().query("role == 'trainval'").reset_index(drop=True)
        _cache["cv"] = (tv, run_cv(tv, R0_FEATURES,
                                   lambda: LogisticRegression(C=np.inf, max_iter=1000)))
    return _cache["cv"]


def test_train_val_students_disjoint_within_each_fold():
    tv, res = _cv()
    assert len(res.fold_users) == 5
    for f, (tr, va) in res.fold_users.items():
        assert tr.isdisjoint(va), f"fold {f} 訓練與驗證學生有交集"
        # 驗證學生正好是 fold == f 的學生；訓練學生正好是其餘 fold 的學生
        assert va == set(tv.loc[tv["fold"] == f, "user_id"])
        assert tr == set(tv.loc[tv["fold"] != f, "user_id"])


def test_no_test_students_in_any_train_or_val():
    allrows = _all_rows()
    test_users = set(allrows.loc[allrows["role"] == "test", "user_id"])
    assert len(test_users) > 0
    _, res = _cv()
    for f, (tr, va) in res.fold_users.items():
        assert tr.isdisjoint(test_users) and va.isdisjoint(test_users)
    assert set(res.oof["user_id"]).isdisjoint(test_users)
    # 把 test 列混進去，run_cv 必須直接拒絕
    try:
        run_cv(allrows, R0_FEATURES, lambda: LogisticRegression(C=np.inf))
    except AssertionError:
        pass
    else:
        raise AssertionError("run_cv 應該拒絕含 test 列的資料")


def test_threshold_uses_only_training_predictions():
    rng = np.random.default_rng(0)
    y_tr = rng.integers(0, 2, 300)
    p_tr = np.clip(0.5 + 0.3 * (y_tr - 0.5) + rng.normal(0, 0.2, 300), 0.01, 0.99)
    thr = select_threshold(y_tr, p_tr)
    # select_threshold 只收訓練資料；驗證列的標籤不管怎麼改都碰不到它。
    # 這裡驗證兩件事：(1) 同樣的訓練輸入結果完全相同；(2) 閾值真的是訓練 F1 的最大值。
    assert select_threshold(y_tr.copy(), p_tr.copy()) == thr
    y_val = rng.integers(0, 2, 100)
    y_val_flipped = 1 - y_val                      # 改變驗證標籤
    assert select_threshold(y_tr, p_tr) == thr     # 閾值不受影響
    assert not np.array_equal(y_val, y_val_flipped)

    def f1_at(t):
        score, y = 1 - p_tr, 1 - y_tr
        pred = score >= t
        tp = (pred & (y == 1)).sum()
        return 2 * tp / max(pred.sum() + y.sum(), 1)
    best = max(f1_at(t) for t in np.unique(1 - p_tr))
    assert abs(f1_at(thr) - best) < 1e-12


def test_oof_covers_every_trainval_row_exactly_once():
    tv, res = _cv()
    assert len(res.oof) == len(tv) == 30516
    assert res.oof["order_id"].is_unique
    assert set(res.oof["order_id"]) == set(tv["order_id"])
    assert res.oof["p_correct"].between(0, 1).all()
    # 預測回去的標籤也沒被弄亂
    merged = res.oof.merge(tv[["order_id", "correct", "fold"]], on="order_id", suffixes=("", "_tv"))
    assert (merged["correct"] == merged["correct_tv"]).all()
    assert (merged["fold"] == merged["fold_tv"]).all()


def _toy_cut(n_students=60, seed=1):
    rng = np.random.default_rng(seed)
    uid = np.repeat(np.arange(n_students), rng.integers(1, 3, n_students))  # 有些學生 2 列
    y = rng.integers(0, 2, len(uid))
    score = np.clip(0.4 * y + rng.random(len(uid)) * 0.8, 0, 1)
    return y, score, uid


def test_bootstrap_is_reproducible_and_resamples_students():
    y, score, uid = _toy_cut()
    a = _cluster_bootstrap_ci(y, score, uid, n_boot=200, seed=7)
    b = _cluster_bootstrap_ci(y, score, uid, n_boot=200, seed=7)
    assert a == b, "固定種子重跑結果應該相同"
    c = _cluster_bootstrap_ci(y, score, uid, n_boot=200, seed=8)
    assert a != c, "不同種子應該不同（確認種子真的有被用到）"

    # 重抽單位是學生：每次重抽中，同一位學生的所有列出現的次數一定相同
    codes, _ = pd.factorize(uid)
    groups = [np.flatnonzero(codes == k) for k in range(codes.max() + 1)]
    rng = np.random.default_rng(0)
    for _ in range(50):
        idx = cluster_resample_indices(groups, rng)
        counts = np.bincount(idx, minlength=len(uid))
        for g in groups:
            assert len(set(counts[g])) == 1, "同一位學生的列被拆開了"
        assert counts.sum() == len(idx)
    # 兩列的學生確實存在，否則上面的檢查沒有意義
    assert any(len(g) == 2 for g in groups)


if __name__ == "__main__":
    tests = [test_train_val_students_disjoint_within_each_fold,
             test_no_test_students_in_any_train_or_val,
             test_threshold_uses_only_training_predictions,
             test_oof_covers_every_trainval_row_exactly_once,
             test_bootstrap_is_reproducible_and_resamples_students]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
