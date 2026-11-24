"""test 最終評估用的評估函式（協定見 CLAUDE.md「test 最終評估協定」）。

    eval_all_rows      全部列的 ROC-AUC、PR-AUC，附以學生為單位的 cluster bootstrap 95% 信賴區間
    paired_delta_all   全部列的配對 ΔAUC（new − old），附配對 cluster bootstrap 區間
    interpret_delta    判讀規則：區間含 0 → 「無法確認」，不下結論

重抽程序與 evaluate.py 完全共用（cluster_resample_indices、固定種子），這裡不另寫一份。
C（t = 6、11）直接用 evaluate.eval_c／paired_delta_c。
"""
import numpy as np

from evaluate import (BOOT_SEED, N_BOOT, _auc_pair, _cluster_bootstrap_ci,  # noqa: F401
                      paired_delta_c, to_wrong)

CANNOT_CONFIRM = "無法確認"


def eval_all_rows(oof, n_boot=N_BOOT, seed=BOOT_SEED):
    """全部列（不分位置）的 ROC-AUC 與 PR-AUC（答錯為正類）＋ 95% cluster bootstrap 區間。
    oof 需要欄位 user_id、correct、p_correct。回傳 dict 列表（只有彙總數字）。"""
    y, score = to_wrong(oof["correct"], oof["p_correct"])
    roc, pr = _auc_pair(y, score)
    r_lo, r_hi, p_lo, p_hi, _ = _cluster_bootstrap_ci(y, score, oof["user_id"].to_numpy(), n_boot, seed)
    n, ns = len(oof), oof["user_id"].nunique()
    base = dict(eval="ALL", group="all_rows", n_rows=n, n_students=ns)
    return [dict(base, metric="roc_auc", value=roc, ci_low=r_lo, ci_high=r_hi),
            dict(base, metric="pr_auc", value=pr, ci_low=p_lo, ci_high=p_hi)]


def paired_delta_all(oof_old, oof_new, n_boot=N_BOOT, seed=BOOT_SEED):
    """全部列的配對 ΔAUC = AUC(new) − AUC(old)，附 95% 配對 cluster bootstrap 區間。

    做法：把兩份 OOF 的 t 都設成同一個值、切點取那個值，paired_delta_c 的「t == 切點」就是全部列；
    這樣重抽程序與 C 的配對區間完全相同（同一個函式）。回傳一列，group = "all_rows"。
    """
    out = paired_delta_c(oof_old.assign(t=1), oof_new.assign(t=1), cut_points=(1,), n_boot=n_boot, seed=seed)
    return out.assign(group="all_rows")


def interpret_delta(ci_low, ci_high) -> str:
    """判讀 Δ（new − old）：區間含 0（含剛好等於 0）→「無法確認」；下界 > 0 →「改善」；上界 < 0 →「變差」。
    區間缺失（NaN）也算無法確認。"""
    if np.isnan(ci_low) or np.isnan(ci_high):
        return CANNOT_CONFIRM
    if ci_low > 0:
        return "改善"
    if ci_high < 0:
        return "變差"
    return CANNOT_CONFIRM
