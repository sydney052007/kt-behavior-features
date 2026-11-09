"""R1r：R1 + 4 個「相對於題目平均」的行為特徵（事先決定的唯一一個特徵變體）。

動機：同一題的提示率、嘗試次數本來就和題目有關。把學生這一列的行為減掉「其他學生在這題的平均行為」，
得到『比這題的一般水準多或少』的訊號，再取歷史平均與上一列。

題目統計（只用「其他學生」的作答；平滑方式和 item_diff 相同）：
    item_hint(題目) = (Σ hint_flag + m × h̄_技能) / (n + m)         hint_flag = (hint_count > 0)
    item_att(題目)  = (Σ log1p(attempt_count) + m × ā_技能) / (n + m)
    m 沿用 ITEM_M（技能 47 → 5、311 → 20），固定不調；h̄_技能、ā_技能 = 參考集中該技能的平均；
    沒見過的題目 = 技能平均。
    參考集：訓練列 = 同一個 inner 分組中的「其他 4 組」（和 item_diff 用同一個 inner_groups）；
            驗證列 = 整個訓練集。
逐列訊號：rel_hint_row = hint_flag − item_hint(該列題目)；rel_att_row = log1p(attempt_count) − item_att(該列題目)。
4 個新特徵（只用第 1…t−1 列，沿用 _hist_mean 與 _last）：
    rel_hint_mean, rel_hint_last, rel_att_mean, rel_att_last
第 1 題（沒有歷史）填「訓練列 rel_*_row 的平均」。

目標列自己的 hint_count、attempt_count 只會出現在它『後面』的列的特徵裡；
它自己的 item_hint／item_att 也不含它自己這一組學生（inner 分組）。
features.py 與 features_r1.py 完全沒動；R1 既有的 13 個特徵直接由 make_fold_features 產生。
"""
import pandas as pd

import numpy as np

from features_r1 import (ITEM_M, INNER_SPLITS, R0D_FEATURES, R1_FEATURES, _hist_mean, _last,
                         inner_groups, make_fold_features)

REL_FEATURES = ["rel_hint_mean", "rel_hint_last", "rel_att_mean", "rel_att_last"]
R1R_FEATURES = R1_FEATURES + REL_FEATURES          # 17 個
R0D_REL_FEATURES = R0D_FEATURES + REL_FEATURES     # 10 個（診斷用）


def _hint_att(df: pd.DataFrame) -> pd.DataFrame:
    """逐列訊號：hint_flag = (hint_count > 0)；log_att = log1p(attempt_count)。"""
    return pd.DataFrame({"hint_flag": (df["hint_count"] > 0).astype(float),
                         "log_att": np.log1p(df["attempt_count"].astype(float))}, index=df.index)


def item_stats_encode(ref: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """用 ref 的作答算出 rows 每一列題目的 item_hint、item_att（平滑平均）。"""
    sig = _hint_att(ref)
    out = {}
    for col, name in [("hint_flag", "item_hint"), ("log_att", "item_att")]:
        d = pd.DataFrame({"skill_id": ref["skill_id"].to_numpy(), "problem_id": ref["problem_id"].to_numpy(),
                          "v": sig[col].to_numpy()})
        skill_mean = d.groupby("skill_id")["v"].mean()                      # h̄_技能／ā_技能（列加權）
        stat = d.groupby(["skill_id", "problem_id"])["v"].agg(s="sum", n="size")
        idx = pd.MultiIndex.from_frame(rows[["skill_id", "problem_id"]])
        s = stat["s"].reindex(idx).fillna(0).to_numpy()                     # 沒見過的題目：s = n = 0
        n = stat["n"].reindex(idx).fillna(0).to_numpy()
        m = rows["skill_id"].map(ITEM_M).to_numpy(float)
        mean = rows["skill_id"].map(skill_mean).to_numpy(float)
        out[name] = (s + m * mean) / (n + m)                                # n = 0 時剛好等於技能平均
    return pd.DataFrame(out, index=rows.index)


def item_stats_train_oof(train: pd.DataFrame) -> pd.DataFrame:
    """訓練列的 item_hint、item_att：每列只用『其他 4 組』學生（與 item_diff 相同的 inner 分組）。"""
    grp = inner_groups(train)
    parts = []
    for g in range(INNER_SPLITS):
        mask = grp == g
        parts.append(item_stats_encode(train[~mask], train[mask]))
    return pd.concat(parts).loc[train.index]


def _rel_features(df: pd.DataFrame, items: pd.DataFrame, fills: dict) -> pd.DataFrame:
    """用題目統計算 df 的 4 個新特徵（df 須已依 user_id, skill_id, order_id 排序）。"""
    sig = _hint_att(df)
    rel_hint_row = sig["hint_flag"] - items["item_hint"]
    rel_att_row = sig["log_att"] - items["item_att"]
    out = pd.DataFrame(index=df.index)
    out["rel_hint_mean"] = _hist_mean(df, rel_hint_row).fillna(fills["rel_hint"])
    out["rel_hint_last"] = _last(df, rel_hint_row).fillna(fills["rel_hint"])
    out["rel_att_mean"] = _hist_mean(df, rel_att_row).fillna(fills["rel_att"])
    out["rel_att_last"] = _last(df, rel_att_row).fillna(fills["rel_att"])
    return out


def make_fold_features_r1r(train: pd.DataFrame, val: pd.DataFrame):
    """CV 的一個 fold：R1 的 13 個特徵 + 4 個新特徵。統計量全由 train 算；test 列直接報錯。

    回傳 (train_feat, val_feat, stats)；stats 多一個 rel_fills（第 1 題的填補值）。
    """
    train_feat, val_feat, stats = make_fold_features(train, val)    # 已 assert 無 test 列、學生不重疊
    tr_items = item_stats_train_oof(train)
    va_items = item_stats_encode(train, val)

    # 第 1 題的填補值 = 訓練列 rel_*_row 的平均
    sig = _hint_att(train)
    fills = {"rel_hint": float((sig["hint_flag"] - tr_items["item_hint"]).mean()),
             "rel_att": float((sig["log_att"] - tr_items["item_att"]).mean())}

    train_feat = train_feat.join(_rel_features(train, tr_items, fills))
    val_feat = val_feat.join(_rel_features(val, va_items, fills))
    return train_feat, val_feat, {**stats, "rel_fills": fills}
