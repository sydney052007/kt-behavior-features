"""可重用的評估：A 全部列（每個 fold）、B 位置分桶、C 固定切點（cluster bootstrap 信賴區間）。

輸入的 OOF 表需要這些欄位：
    user_id, fold, t, correct（真實標籤，1 = 答對）, p_correct（模型預測的 P(答對)）

評估一律把「答錯」當正類：y = 1 - correct，分數 = 1 - p_correct。
F1 與 Accuracy 的閾值由呼叫端用「該 fold 的訓練列」的預測選好（select_threshold），
以 {fold: 閾值} 傳進來；這個檔案不會、也沒辦法用驗證列去選閾值。

R0、R1、跨學校實驗都呼叫同一個 evaluate()。
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, roc_auc_score)

CUT_POINTS = (6, 11)   # 前 5 題 → 第 6 題、前 10 題 → 第 11 題（可由參數覆寫）
N_BOOT = 2000
BOOT_SEED = 42
# 位置分桶：(名稱, 下界, 上界)，上界 None = 無上限
BUCKETS = [("t=1", 1, 1), ("t=2-4", 2, 4), ("t=5-10", 5, 10), ("t=11+", 11, None)]


def to_wrong(correct, p_correct):
    """換成「答錯為正類」的標籤與分數。"""
    return 1 - np.asarray(correct), 1 - np.asarray(p_correct)


def select_threshold(correct_train, p_correct_train) -> float:
    """用『訓練列』的預測選閾值，使 F1（答錯為正類）最大。

    函式簽名只收訓練資料，所以不可能偷看驗證列。
    分數 >= 閾值 就預測答錯。並列時取最小的閾值，讓結果確定。
    """
    y, score = to_wrong(correct_train, p_correct_train)
    precision, recall, thr = precision_recall_curve(y, score)
    precision, recall = precision[:-1], recall[:-1]          # 與 thr 對齊
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    return float(thr[int(np.argmax(f1))])                   # argmax 取第一個 = 最小閾值


def _auc_pair(y, score):
    """(ROC-AUC, PR-AUC)；只有單一類別時回傳 (nan, nan)。"""
    if y.min() == y.max():
        return np.nan, np.nan
    return roc_auc_score(y, score), average_precision_score(y, score)


def cluster_resample_indices(groups, rng):
    """cluster bootstrap 的一次重抽。

    groups：list，每個元素是「某位學生在這個切點的列索引」。
    重抽的單位是學生（可重複抽到）；抽到的學生會帶著他的所有列，不會被拆開。
    """
    picks = rng.integers(0, len(groups), size=len(groups))
    return np.concatenate([groups[i] for i in picks])


def _cluster_bootstrap_ci(y, score, user_ids, n_boot, seed):
    """回傳 (roc_lo, roc_hi, pr_lo, pr_hi, 略過次數)，95% 百分位數區間。"""
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(user_ids)
    order = np.argsort(codes, kind="stable")
    splits = np.flatnonzero(np.diff(codes[order])) + 1
    groups = np.split(order, splits)                         # 每位學生的列索引

    rocs, prs, skipped = [], [], 0
    for _ in range(n_boot):
        idx = cluster_resample_indices(groups, rng)
        r, p = _auc_pair(y[idx], score[idx])
        if np.isnan(r):                                      # 這次重抽只剩單一類別
            skipped += 1
            continue
        rocs.append(r)
        prs.append(p)
    lo, hi = 2.5, 97.5
    return (*np.percentile(rocs, [lo, hi]), *np.percentile(prs, [lo, hi]), skipped)


def _row(ev, group, metric, value, n_rows, n_students, std=np.nan, lo=np.nan, hi=np.nan):
    return dict(eval=ev, group=group, metric=metric, value=value, std=std,
                ci_low=lo, ci_high=hi, n_rows=n_rows, n_students=n_students)


def eval_a(oof, thresholds):
    """A：每個 fold 的驗證列各算一次，報告 5 個 fold 的平均 ± 標準差。"""
    per_fold = []
    for f, g in oof.groupby("fold"):
        y, score = to_wrong(g["correct"], g["p_correct"])
        roc, pr = _auc_pair(y, score)
        pred = (score >= thresholds[f]).astype(int)          # 閾值來自訓練列
        per_fold.append(dict(
            roc_auc=roc, pr_auc=pr,
            f1=f1_score(y, pred, zero_division=0),
            accuracy=accuracy_score(y, pred),
            acc_baseline_all_correct=accuracy_score(y, np.zeros_like(y)),
            threshold=thresholds[f]))
    pf = pd.DataFrame(per_fold)
    n_rows, n_stu = len(oof), oof["user_id"].nunique()
    return [_row("A", "all_rows", m, pf[m].mean(), n_rows, n_stu, std=pf[m].std(ddof=1))
            for m in pf.columns]


def eval_b(oof):
    """B：合併後的 OOF，依位置 t 分桶。"""
    out = []
    for name, lo, hi in BUCKETS:
        mask = oof["t"] >= lo
        if hi is not None:
            mask &= oof["t"] <= hi
        g = oof[mask]
        y, score = to_wrong(g["correct"], g["p_correct"])
        roc, pr = _auc_pair(y, score)
        n, ns = len(g), g["user_id"].nunique()
        out += [_row("B", name, "wrong_rate", y.mean(), n, ns),
                _row("B", name, "roc_auc", roc, n, ns),
                _row("B", name, "pr_auc", pr, n, ns)]
    return out


def eval_c(oof, cut_points=CUT_POINTS, n_boot=N_BOOT, seed=BOOT_SEED):
    """C：固定切點 t == k 的那一列，附 cluster bootstrap 95% 信賴區間。"""
    out = []
    for k in cut_points:
        g = oof[oof["t"] == k]
        y, score = to_wrong(g["correct"], g["p_correct"])
        roc, pr = _auc_pair(y, score)
        n, ns = len(g), g["user_id"].nunique()
        r_lo, r_hi, p_lo, p_hi, skipped = _cluster_bootstrap_ci(
            y, score, g["user_id"].to_numpy(), n_boot, seed)
        if skipped:
            print(f"[evaluate] 切點 t={k}：{skipped}/{n_boot} 次重抽只有單一類別，已略過")
        grp = f"t={k}"
        out += [_row("C", grp, "wrong_rate", y.mean(), n, ns),
                _row("C", grp, "roc_auc", roc, n, ns, lo=r_lo, hi=r_hi),
                _row("C", grp, "pr_auc", pr, n, ns, lo=p_lo, hi=p_hi)]
    return out


def evaluate(oof, thresholds, cut_points=CUT_POINTS, n_boot=N_BOOT, seed=BOOT_SEED):
    """A + B + C 全部算完，回傳只含彙總數字的長格式表（沒有學生層級的列）。"""
    rows = eval_a(oof, thresholds) + eval_b(oof) + eval_c(oof, cut_points, n_boot, seed)
    return pd.DataFrame(rows)
