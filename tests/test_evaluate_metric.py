"""evaluate.py 的 per_fold_metric 與 paired_delta_c(metric=...) 的測試。
可直接用 python3 tests/test_evaluate_metric.py 跑（用構造的小資料，不讀真實資料）。
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from evaluate import paired_delta_c, per_fold_auc, per_fold_metric  # noqa: E402


def _toy(n_users=120, seed=0):
    """每位學生 12 列（t = 1…12）、分成 5 個 fold；兩個模型的分數不同。"""
    rng = np.random.default_rng(seed)
    uid = np.repeat(np.arange(n_users), 12)
    t = np.tile(np.arange(1, 13), n_users)
    fold = np.repeat(rng.integers(0, 5, n_users), 12)
    correct = rng.integers(0, 2, len(uid))
    base = pd.DataFrame({"user_id": uid, "skill_id": "47", "order_id": np.arange(len(uid)), "t": t,
                         "fold": fold, "correct": correct})
    old = base.assign(p_correct=np.clip(0.3 + 0.2 * correct + rng.random(len(uid)) * 0.5, 0.01, 0.99))
    new = base.assign(p_correct=np.clip(0.2 + 0.4 * correct + rng.random(len(uid)) * 0.4, 0.01, 0.99))
    return old, new


def test_per_fold_metric_matches_sklearn():
    old, _ = _toy()
    roc, pr = per_fold_metric(old, "roc_auc"), per_fold_metric(old, "pr_auc")
    assert per_fold_auc(old).equals(roc)                       # 舊函式與新函式的 ROC 版本一致
    for f, g in old.groupby("fold"):
        y_wrong, score = 1 - g["correct"], 1 - g["p_correct"]  # 答錯為正類
        assert np.isclose(roc[f], roc_auc_score(y_wrong, score))
        assert np.isclose(pr[f], average_precision_score(y_wrong, score))
    assert not np.allclose(roc.to_numpy(), pr.to_numpy())      # 兩個指標確實不同（不是空轉）


def test_paired_delta_c_metric_option():
    old, new = _toy()
    kw = dict(cut_points=(6, 11), n_boot=200, seed=7)
    default = paired_delta_c(old, new, **kw)
    roc = paired_delta_c(old, new, metric="roc_auc", **kw)
    pr = paired_delta_c(old, new, metric="pr_auc", **kw)
    assert default.equals(roc), "預設必須維持 ROC-AUC"
    assert not pr["delta_auc"].equals(roc["delta_auc"])
    # 點估計 = 該切點那一列上 new 的指標 − old 的指標（獨立重算）
    for k in (6, 11):
        a, b = old[old["t"] == k], new[new["t"] == k]
        y = 1 - a["correct"].to_numpy()
        for df_out, fn in [(roc, roc_auc_score), (pr, average_precision_score)]:
            exp = fn(y, 1 - b["p_correct"].to_numpy()) - fn(y, 1 - a["p_correct"].to_numpy())
            got = df_out.loc[df_out["group"] == f"t={k}", "delta_auc"].iloc[0]
            assert np.isclose(got, exp), (k, got, exp)
    # 信賴區間包住（或貼近）點估計；固定種子重跑相同
    for out in (roc, pr):
        assert (out["ci_low"] <= out["ci_high"]).all()
    again = paired_delta_c(old, new, metric="pr_auc", **kw)
    assert again.equals(pr), "固定種子重跑結果應相同"
    try:
        paired_delta_c(old, new, metric="f1", **kw)
    except KeyError:
        pass
    else:
        raise AssertionError("不支援的 metric 應該報錯")


def test_paired_delta_c_is_paired():
    """兩個一模一樣的模型，Δ 與整個信賴區間都必須是 0（若不是配對重抽，區間不會是 0）。"""
    old, _ = _toy()
    for metric in ("roc_auc", "pr_auc"):
        out = paired_delta_c(old, old.copy(), n_boot=100, seed=3, metric=metric)
        assert (out["delta_auc"] == 0).all() and (out["ci_low"] == 0).all() and (out["ci_high"] == 0).all()


if __name__ == "__main__":
    tests = [test_per_fold_metric_matches_sklearn, test_paired_delta_c_metric_option, test_paired_delta_c_is_paired]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
