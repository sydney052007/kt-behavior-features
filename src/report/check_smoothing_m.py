"""item_diff 平滑參數 m 的估計依據（技能 47 → 5、技能 311 → 20 是怎麼來的）。

item_diff = (答對數 + m × p̄) / (作答數 + m)，m 是「先驗的虛擬作答數」，固定不調（CLAUDE.md「已決定」）。
這支腳本用 trainval 的題目答對率，重新估一次每個技能的 m，只用來核對、不用來改 m：

  * method of moments：各題答對率的變異數 = 二項抽樣的變異 + 題目真正難度的變異（tau2），
    把 tau2 解出來，Beta 先驗的 a+b = p̄(1-p̄)/tau2 - 1 就是 m。
  * MLE：假設每題的答對機率服從 Beta(a, b)，作答數 n、答對數 k 服從 beta-binomial，
    直接最大化對數概似，m = a + b。
  * 另外對 5 個 fold 各用「訓練列」（fold != k）重算 method of moments，看 m 穩不穩。

資料範圍：只用 role == "trainval" 的列（30,516 列）。載入後第一件事就是丟掉 test 列，
並用 assert 確認；test 的 262 位學生不參與任何統計。
輸出只有彙總（每個技能一列、每個 fold 一列），不含任何學生層級或題目層級的列。

用法（在 repo 根目錄）：
    python3 src/report/check_smoothing_m.py
輸出：印出結果，同時存成 results/01_preprocessing/smoothing_m_estimate.txt
"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import betabinom

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from features import MAIN_PATH, OUT_PATH  # noqa: E402
from features_r1 import ITEM_M  # noqa: E402

OUT_TXT = "results/01_preprocessing/smoothing_m_estimate.txt"
KEY = ["user_id", "skill_id", "order_id"]
N_TRAINVAL = 30516
SKILLS = ["47", "311"]


def assert_trainval_only(df: pd.DataFrame) -> None:
    """程式內保證：沒有任何 test 列（test 的 fold = -1）。"""
    assert (df["role"] == "trainval").all(), "含 role != trainval 的列（test 不能碰）"
    assert df["fold"].between(0, 4).all(), "fold 不在 0–4"


def load_trainval() -> pd.DataFrame:
    """讀 features_r0.csv（取 role、fold）與 main_s47_311.csv（取 problem_id），丟掉 test 列後回傳。

    注意：features_r0.csv 本身含 test 列，所以要先讀進來才能用 role 篩；篩完立刻只留 trainval，
    之後所有統計都在篩完的資料上算。"""
    f = pd.read_csv(OUT_PATH, usecols=KEY + ["correct", "role", "fold"], dtype={"skill_id": str})
    m = pd.read_csv(MAIN_PATH, usecols=KEY + ["problem_id", "correct"], dtype={"skill_id": str})
    d = f.merge(m, on=KEY, validate="one_to_one", suffixes=("", "_m"))
    assert len(d) == 36048 and (d["correct"] == d["correct_m"]).all()
    tv = d[d["role"] == "trainval"].drop(columns="correct_m").reset_index(drop=True)
    del d, f, m
    assert_trainval_only(tv)
    assert len(tv) == N_TRAINVAL and tv["problem_id"].notna().all()
    return tv


def problem_counts(g: pd.DataFrame):
    """每題的 (作答數 n, 答對數 k)。"""
    s = g.groupby("problem_id")["correct"].agg(n="size", k="sum")
    return s["n"].to_numpy(float), s["k"].to_numpy(float)


def mom(n: np.ndarray, k: np.ndarray):
    """method of moments：回傳 (p̄, tau2, m)。tau2 <= 0 時 m 無法估（回傳 NaN）。"""
    pbar = k.sum() / n.sum()
    p = k / n
    # 各題答對率圍繞 p̄ 的加權變異 − 純二項抽樣預期的變異 = 題目難度真正的變異 tau2
    tau2 = (n * (p - pbar) ** 2).sum() / n.sum() - pbar * (1 - pbar) * len(n) / n.sum()
    m = pbar * (1 - pbar) / tau2 - 1 if tau2 > 0 else np.nan
    return pbar, tau2, m


def mle(n: np.ndarray, k: np.ndarray, start):
    """beta-binomial 的最大概似（Nelder-Mead，對 log(a)、log(b) 搜尋）；回傳 ((a, b), 是否收斂)。"""
    nll = lambda th: -betabinom.logpmf(k, n, np.exp(th[0]), np.exp(th[1])).sum()
    r = minimize(nll, np.log(start), method="Nelder-Mead",
                 options=dict(xatol=1e-8, fatol=1e-10, maxiter=5000))
    return np.exp(r.x), bool(r.success)


def estimate(tv: pd.DataFrame):
    """回傳 (各技能整體估計表, 各 fold 的 method of moments 表)。傳進來的資料必須只有 trainval。"""
    assert_trainval_only(tv)
    rows = []
    for s in SKILLS:
        n, k = problem_counts(tv[tv["skill_id"] == s])
        pbar, tau2, mm = mom(n, k)
        (a, b), ok = mle(n, k, (pbar * 5, (1 - pbar) * 5))
        rows.append(dict(skill=s, n_rows=int(n.sum()), n_problems=len(n), p_bar=pbar, tau2=tau2,
                         m_moments=mm, m_MLE=a + b, a=a, b=b, mle_converged=ok,
                         m_adopted=ITEM_M[s]))
    overall = pd.DataFrame(rows)

    out = []
    for kf in range(5):
        r = dict(fold=kf)
        for s in SKILLS:
            n, k = problem_counts(tv[(tv["fold"] != kf) & (tv["skill_id"] == s)])
            _, tau2, mm = mom(n, k)
            r[f"m_{s}"], r[f"tau2_{s}"] = mm, tau2
        out.append(r)
    return overall, pd.DataFrame(out)


def main() -> None:
    tv = load_trainval()
    overall, per_fold = estimate(tv)
    lines = []
    lines.append("=== item_diff 平滑參數 m 的重新估計（只用 trainval）===")
    lines.append(f"輸入列：role == trainval，共 {len(tv):,} 列（技能 47：{(tv.skill_id == '47').sum():,}；"
                 f"技能 311：{(tv.skill_id == '311').sum():,}）；fold 0–4；已丟掉 test 列，程式內 assert。")
    lines.append("\n各技能整體（trainval 全部列）：")
    lines.append(overall.round(4).to_string(index=False))
    lines.append("\n各 fold：用訓練列（fold != k）重算 method of moments")
    lines.append(per_fold.round(4).to_string(index=False))
    lines.append(per_fold.drop(columns="fold").agg(["mean", "std", "min", "max"]).round(4).to_string())
    lines.append("\n目前採用：" + "、".join(f"技能 {s} → m = {ITEM_M[s]}" for s in SKILLS)
                 + "（固定不調；這支腳本只回報重新估計的數字，不修改 m）。")
    text = "\n".join(lines)
    print(text)
    with open(OUT_TXT, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")


if __name__ == "__main__":
    main()
