"""R0d／R1 共 13 個特徵的描述統計表（報告的「特徵清單／前處理表」）。

每個特徵一列：定義、類型、範圍（min、max）、平均、標準差、與 correct 的相關、缺失或填補規則，
以及實際有幾列是「填補值／預設值」。

統計怎麼算：和模型一樣，用 5-fold 的驗證列。每個 fold 的訓練統計量（p1、p99、題目作答時間中位數、
填補值、item_diff）都只由該 fold 的訓練列算出，套到該 fold 的驗證列；5 個 fold 的驗證列合起來就是
trainval 的 30,516 列，描述統計在這 30,516 列上算。

資料範圍：只用 role == "trainval"。load_r1_frame() 會把 features_r0.csv 的全部列（含 test）讀進來，
所以載入後第一件事就是只留 trainval 並 assert；之後 make_fold_features 也會再檢查一次。
輸出只有彙總（13 列），沒有學生層級或題目層級的資料。

用法（在 repo 根目錄）：
    python3 src/report_features_r1.py
輸出：results/01_preprocessing/r1_feature_stats.csv（並印出簡要摘要）
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import features_r1 as fr  # noqa: E402
from features_r1 import R1_FEATURES, load_r1_frame, make_fold_features  # noqa: E402

OUT_CSV = "results/01_preprocessing/r1_feature_stats.csv"
N_TRAINVAL = 30516

# 特徵說明：(群組, 類型, 定義, 缺失／填補規則)。內容對照 src/features.py 與 src/features_r1.py。
SPEC = {
    "log1p_c": ("R0", "次數（log）", "log1p(同一學生同一技能前面答對的次數)", "t=1 沒有歷史 → 次數 0，值為 0（不是填補）"),
    "log1p_w": ("R0", "次數（log）", "log1p(同一學生同一技能前面答錯的次數)", "t=1 沒有歷史 → 次數 0，值為 0（不是填補）"),
    "skill_311": ("R0", "0/1", "技能 311 = 1，技能 47 = 0", "無缺失"),
    "has_history": ("R0", "0/1", "t > 1（前面至少有 1 題）", "無缺失；t=1 為 0"),
    "item_diff": ("R0d", "比率", "題目難度＝(答對數 + m·p̄)/(作答數 + m)；m：技能 47 = 5、311 = 20；p̄ = 訓練集該技能答對率",
                  "訓練列用 inner 5-fold（學生分組）、驗證列用整個訓練集；驗證列的題目沒在訓練集出現過 → p̄"),
    "last_correct": ("R0d", "0/1（填補後為比率）", "上一題是否答對", "t=1 → 訓練集平均答對率"),
    "hint_rate": ("R1", "比率", "前面各題『有要提示』（hint_count>0）的比例", "t=1 → 訓練集『有要提示』比例"),
    "last_hint": ("R1", "0/1（填補後為比率）", "上一題是否有要提示", "t=1 → 訓練集『有要提示』比例"),
    "mean_log_time": ("R1", "標準化時間", "前面各題標準化作答時間的平均", "t=1 → 訓練集標準化時間平均；單題時間缺失先填同一平均"),
    "last_log_time": ("R1", "標準化時間", "上一題的標準化作答時間", "t=1 → 訓練集標準化時間平均；單題時間缺失先填同一平均"),
    "mean_log_att": ("R1", "log 次數", "前面各題 log1p(嘗試次數) 的平均", "t=1 → 訓練集 log1p(嘗試次數) 平均"),
    "last_log_att": ("R1", "log 次數", "上一題的 log1p(嘗試次數)", "t=1 → 訓練集 log1p(嘗試次數) 平均"),
    "scaffold_rate": ("R1", "比率", "前面各題『first_action = 2（要求鷹架）』的比例", "t=1 → 訓練集該比例"),
}
TIME_NOTE = ("標準化時間＝原始 ms ≤ 0 標缺失 → 夾在訓練集 [p1, p99] → log → 減去該題訓練集 log 中位數")
# t=1 時用填補值的特徵（has_history == 0 的列都是填補值）
T1_FILLED = {"last_correct", "hint_rate", "last_hint", "mean_log_time", "last_log_time",
             "mean_log_att", "last_log_att", "scaffold_rate"}


def load_trainval() -> pd.DataFrame:
    """載入後馬上只留 trainval（test 不參與任何統計），並用 assert 確認。"""
    df = load_r1_frame()
    tv = df[df["role"] == "trainval"].reset_index(drop=True)
    del df
    assert_trainval_only(tv)
    return tv


def assert_trainval_only(tv: pd.DataFrame) -> None:
    assert (tv["role"] == "trainval").all(), "含 role != trainval 的列（test 不能碰）"
    assert tv["fold"].between(0, 4).all(), "fold 不在 0–4"


def build_table(tv: pd.DataFrame) -> pd.DataFrame:
    """5 個 fold 各自由訓練列算統計量、算驗證列特徵；合併後算 13 個特徵的描述統計。"""
    assert_trainval_only(tv)
    vals, n_unseen, n_time_missing, n_clip_hi, n_clip_lo, p1s, p99s = [], 0, 0, 0, 0, [], []
    for k in range(5):
        tr, va = tv[tv["fold"] != k], tv[tv["fold"] == k]
        _, vf, st = make_fold_features(tr, va)          # 內部會再檢查 role、學生不重疊
        vals.append(vf)
        # item_diff 的填補：驗證列的 (技能, 題目) 沒在訓練集出現過
        seen = set(zip(tr["skill_id"], tr["problem_id"]))
        n_unseen += sum((s, p) not in seen for s, p in zip(va["skill_id"], va["problem_id"]))
        # 標準化時間缺失（ms ≤ 0 或題目沒有訓練集中位數）與截斷比例
        s_t = fr._std_time(va, st["p1"], st["p99"], st["item_log_median"])
        ms = va["ms_first_response"]
        n_time_missing += int(s_t.isna().sum())
        n_clip_hi += int((ms > st["p99"]).sum())
        n_clip_lo += int(((ms > 0) & (ms < st["p1"])).sum())
        p1s.append(st["p1"]); p99s.append(st["p99"])
    o = pd.concat(vals, ignore_index=True)
    assert len(o) == len(tv) == N_TRAINVAL
    n = len(o)
    n_t1 = int((o["has_history"] == 0).sum())

    rows = []
    for f in R1_FEATURES:
        grp, typ, definition, rule = SPEC[f]
        x = o[f]
        if f in T1_FILLED:
            n_filled = n_t1
        elif f == "item_diff":
            n_filled = n_unseen
        else:
            n_filled = 0
        note = ""
        if f.endswith("log_time"):
            note = (f"{TIME_NOTE}；驗證列單題時間缺失 {n_time_missing / n:.2%}、被 p99 截斷 {n_clip_hi / n:.2%}、"
                    f"被 p1 截斷 {n_clip_lo / n:.2%}；各 fold p1 = {min(p1s):.0f}–{max(p1s):.0f} ms、"
                    f"p99 = {min(p99s):.0f}–{max(p99s):.0f} ms")
        rows.append(dict(feature=f, group=grp, type=typ, definition=definition, min=x.min(), max=x.max(),
                         mean=x.mean(), std=x.std(), corr_correct=x.corr(o["correct"]), n_rows=n,
                         n_missing_after_fill=int(x.isna().sum()), n_filled_or_default=n_filled,
                         share_filled=n_filled / n, fill_rule=rule, note=note))
    return pd.DataFrame(rows)


def main() -> None:
    tv = load_trainval()
    table = build_table(tv)
    table.to_csv(OUT_CSV, index=False)
    print(f"輸入列：role == trainval，共 {len(tv):,} 列（已丟掉 test 列，程式內 assert）；"
          f"特徵 {len(table)} 個 → {OUT_CSV}")
    show = table[["feature", "group", "min", "max", "mean", "std", "corr_correct", "n_filled_or_default"]]
    print(show.round(4).to_string(index=False))
    print(f"\nt=1（has_history == 0）的列數：{int((tv['t'] == 1).sum()):,}")


if __name__ == "__main__":
    main()
