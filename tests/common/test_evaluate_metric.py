"""evaluate.py 的 per_fold_metric 與 paired_delta_c(metric=...) 的測試。
可直接用 python3 tests/common/test_evaluate_metric.py 跑（用構造的小資料，不讀真實資料）。
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from evaluate import (BUCKETS, eval_a, eval_b, eval_c, paired_delta_c, per_fold_auc,  # noqa: E402
                      per_fold_metric)


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
    # 信賴區間：下界 ≤ 上界（沒有檢查是否包住點估計）；固定種子重跑相同
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


def _get(rows, group, metric):
    """從 eval_* 回傳的 dict 列表取出 (group, metric) 那一列（必須恰好一列）。"""
    hit = [r for r in rows if r["group"] == group and r["metric"] == metric]
    assert len(hit) == 1, (group, metric)
    return hit[0]


def _wrong_auc(g):
    """sklearn 獨立重算：答錯為正類 → (ROC-AUC, PR-AUC)。"""
    y, s = 1 - g["correct"].to_numpy(), 1 - g["p_correct"].to_numpy()
    return roc_auc_score(y, s), average_precision_score(y, s)


def test_eval_a_matches_sklearn_per_fold():
    old, _ = _toy()
    thr = {f: 0.40 + 0.05 * f for f in range(5)}                # 每個 fold 不同的閾值（分數 >= 閾值 → 預測答錯）
    rows = eval_a(old, thr)
    rocs, prs = zip(*[_wrong_auc(g) for _, g in old.groupby("fold")])
    f1s, accs, bases = [], [], []
    for f, g in old.groupby("fold"):
        y, pred = 1 - g["correct"].to_numpy(), ((1 - g["p_correct"].to_numpy()) >= thr[f]).astype(int)
        f1s.append(f1_score(y, pred, zero_division=0))
        accs.append(accuracy_score(y, pred))
        bases.append(accuracy_score(y, np.zeros_like(y)))         # 全部猜答對的基準
    for metric, vals in [("f1", f1s), ("accuracy", accs), ("acc_baseline_all_correct", bases)]:
        r = _get(rows, "all_rows", metric)
        assert np.isclose(r["value"], np.mean(vals), atol=1e-12), metric
        assert np.isclose(r["std"], np.std(vals, ddof=1), atol=1e-12), metric
    assert np.isclose(_get(rows, "all_rows", "threshold")["value"], np.mean(list(thr.values())), atol=1e-12)
    for metric, vals in [("roc_auc", rocs), ("pr_auc", prs)]:
        r = _get(rows, "all_rows", metric)
        assert np.isclose(r["value"], np.mean(vals), atol=1e-12), metric
        assert np.isclose(r["std"], np.std(vals, ddof=1), atol=1e-12), metric   # 樣本標準差（ddof=1）
        assert r["n_rows"] == len(old) and r["n_students"] == old["user_id"].nunique()
    assert np.std(rocs, ddof=1) != np.std(rocs, ddof=0)                         # 確認 ddof 真的有差


def test_eval_b_buckets_match_sklearn_and_differ_from_fold_mean():
    old, _ = _toy()
    rows = eval_b(old)
    assert [b[0] for b in BUCKETS] == ["t=1", "t=2-4", "t=5-10", "t=11+"]
    spans = {"t=1": (1, 1), "t=2-4": (2, 4), "t=5-10": (5, 10), "t=11+": (11, 99)}
    for name, (lo, hi) in spans.items():
        g = old[(old["t"] >= lo) & (old["t"] <= hi)]
        roc, pr = _wrong_auc(g)                                   # 合併 OOF 在該桶的列
        assert np.isclose(_get(rows, name, "roc_auc")["value"], roc, atol=1e-12), name
        assert np.isclose(_get(rows, name, "pr_auc")["value"], pr, atol=1e-12), name
        assert np.isclose(_get(rows, name, "wrong_rate")["value"], 1 - g["correct"].mean(), atol=1e-12)
        r = _get(rows, name, "roc_auc")
        assert r["n_rows"] == len(g) and r["n_students"] == g["user_id"].nunique(), name
        # B 是「合併後」的 AUC，不是各 fold AUC 的平均：兩者不相等（觀念：合併 ≠ 平均）
        fold_mean = np.mean([_wrong_auc(gf)[0] for _, gf in g.groupby("fold")])
        assert not np.isclose(roc, fold_mean, atol=1e-9), (name, roc, fold_mean)
    # 全部列也是一樣：合併 AUC ≠ 各 fold AUC 平均
    assert not np.isclose(_wrong_auc(old)[0], np.mean([_wrong_auc(g)[0] for _, g in old.groupby("fold")]), atol=1e-9)


def test_eval_c_point_estimates_and_counts_match_sklearn():
    old, _ = _toy()
    rows = eval_c(old, cut_points=(6, 11), n_boot=100, seed=7)
    assert {r["group"] for r in rows} == {"t=6", "t=11"} and len(rows) == 6
    for k in (6, 11):
        g = old[old["t"] == k]                                    # 切點 = 第 k 題那一列（每位學生 1 列）
        roc, pr = _wrong_auc(g)
        for metric, exp in [("roc_auc", roc), ("pr_auc", pr)]:
            r = _get(rows, f"t={k}", metric)
            assert np.isclose(r["value"], exp, atol=1e-12), (k, metric)
            assert r["n_rows"] == len(g) == g["user_id"].nunique() == r["n_students"], (k, metric)
            assert r["ci_low"] <= r["ci_high"]
        assert np.isclose(_get(rows, f"t={k}", "wrong_rate")["value"], 1 - g["correct"].mean(), atol=1e-12)


if __name__ == "__main__":
    tests = [test_per_fold_metric_matches_sklearn, test_paired_delta_c_metric_option, test_paired_delta_c_is_paired,
             test_eval_a_matches_sklearn_per_fold, test_eval_b_buckets_match_sklearn_and_differ_from_fold_mean,
             test_eval_c_point_estimates_and_counts_match_sklearn]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
