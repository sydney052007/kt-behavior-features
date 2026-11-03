"""R0d 與 R1 的特徵（設計修訂_1103.md 第 4 節）。這個檔案只做特徵，不跑模型。

    R0d = R0_FEATURES + item_diff（題目難度）+ last_correct
    R1  = R0d + 7 個行為特徵
          hint_rate, last_hint, mean_log_time, last_log_time,
          mean_log_att, last_log_att, scaffold_rate

所有「歷史」特徵只用同一學生、同一技能、第 1…t-1 列；目標列自己的 hint_count、
first_action、attempt_count、ms_first_response、overlap_time、bottom_hint 不會進它自己的特徵。

需要訓練集統計量的東西（作答時間的 p1、p99 與題目中位數、第 1 題的填補值、題目難度）
都寫成「給訓練列、算出統計量，再套到要編碼的列」的函式，在 CV 的每個 fold 各算一次
（make_fold_features）；絕對不要先對 trainval 全體算一次再切。
R0 既有的函式（features.py）完全沒動。
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from features import MAIN_PATH, OUT_PATH, R0_FEATURES, SEQ

# 題目難度的平滑參數 m（技能別，method of moments 從 trainval 估出，固定不調）
ITEM_M = {"47": 5, "311": 20}
INNER_SPLITS = 5     # 訓練列的題目難度：把訓練學生再切 5 組
INNER_SEED = 42
TIME_CLIP_LOW_PCT = 1    # 作答時間下限截斷百分位（只看原始 ms > 0 的訓練列）
TIME_CLIP_PCT = 99       # 作答時間上限截斷百分位

R0D_FEATURES = R0_FEATURES + ["item_diff", "last_correct"]
BEHAVIOR_FEATURES = ["hint_rate", "last_hint", "mean_log_time", "last_log_time",
                     "mean_log_att", "last_log_att", "scaffold_rate"]
R1_FEATURES = R0D_FEATURES + BEHAVIOR_FEATURES

# 只用在「歷史」特徵的原始行為欄位（來自 main_s47_311.csv）
RAW_BEHAVIOR_COLS = ["problem_id", "hint_count", "first_action", "attempt_count", "ms_first_response"]
ID_COLS = ["user_id", "skill_id", "order_id", "t", "correct", "fold"]


# ---------------------------------------------------------------- 載入

def load_r1_frame() -> pd.DataFrame:
    """features_r0.csv 併上原始行為欄位（用 user_id, skill_id, order_id 一對一合併）。"""
    f = pd.read_csv(OUT_PATH, dtype={"skill_id": str})
    m = pd.read_csv(MAIN_PATH, usecols=["user_id", "skill_id", "order_id"] + RAW_BEHAVIOR_COLS,
                    dtype={"skill_id": str})
    out = f.merge(m, on=["user_id", "skill_id", "order_id"], how="left", validate="one_to_one")
    assert len(out) == len(f) and out[RAW_BEHAVIOR_COLS].notna().all().all()
    return out.sort_values(SEQ + ["order_id"]).reset_index(drop=True)


def _check_sorted(df: pd.DataFrame) -> None:
    """歷史特徵靠 groupby 內的列順序，所以必須依 (user_id, skill_id, order_id) 排好，且 t 完整。"""
    keys = df[SEQ + ["order_id"]].reset_index(drop=True)
    assert keys.equals(keys.sort_values(SEQ + ["order_id"]).reset_index(drop=True)), "輸入沒排序"
    assert (df.groupby(SEQ).cumcount() + 1 == df["t"]).all(), "序列不完整（t 與列順序不符）"


# ---------------------------------------------------------------- 訓練集統計量

def _row_signals(df: pd.DataFrame) -> pd.DataFrame:
    """每列自己的行為訊號（不含任何歷史）。之後只會被 shift / 累計到「後面的列」。"""
    return pd.DataFrame({
        "hint_flag": (df["hint_count"] > 0).astype(float),      # 有要提示
        "scaffold_flag": (df["first_action"] == 2).astype(float),  # first_action=2 = 要求鷹架（推測）
        "log_att": np.log1p(df["attempt_count"]),
    }, index=df.index)


def _std_time(df: pd.DataFrame, p1: float, p99: float, item_log_median: pd.Series) -> pd.Series:
    """標準化作答時間：≤0 標缺失 → 夾在訓練集 [p1, p99] → log → 減去該題訓練集的 log 中位數。
    缺失（≤0，或題目沒在訓練集出現過所以沒有中位數）回傳 NaN，之後再填訓練集平均。"""
    ms = df["ms_first_response"].astype(float)
    valid = ms > 0
    logt = np.log(ms.where(valid).clip(lower=p1, upper=p99))
    return logt - df["problem_id"].map(item_log_median)


def fit_stats(train: pd.DataFrame) -> dict:
    """只由『訓練列』算出的統計量：p1、p99、各題 log 作答時間中位數、第 1 題的填補值。"""
    ms = train["ms_first_response"].astype(float)
    p1 = float(np.percentile(ms[ms > 0], TIME_CLIP_LOW_PCT))
    p99 = float(np.percentile(ms[ms > 0], TIME_CLIP_PCT))
    logt = np.log(ms.where(ms > 0).clip(lower=p1, upper=p99))      # 題目中位數用夾完之後的值算
    item_log_median = logt.groupby(train["problem_id"]).median()   # NaN 會自動略過

    sig = _row_signals(train)
    std_t = _std_time(train, p1, p99, item_log_median)
    fills = {                                  # 比率類／上一題類在 t=1 的填補值 = 訓練集平均
        "last_correct": float(train["correct"].mean()),
        "hint": float(sig["hint_flag"].mean()),
        "scaffold": float(sig["scaffold_flag"].mean()),
        "log_att": float(sig["log_att"].mean()),
        "log_time": float(std_t.mean()),       # 已標準化的訓練列平均（略過缺失）
    }
    return {"p1": p1, "p99": p99, "item_log_median": item_log_median, "fills": fills}


# ---------------------------------------------------------------- 題目難度

def item_diff_encode(train: pd.DataFrame, rows: pd.DataFrame) -> pd.Series:
    """用 train 的作答算出 rows 的題目難度：(答對數 + m·p̄) / (作答數 + m)。
    p̄ = train 中該技能的整體答對率（列加權）；m 依技能固定；沒出現過的題目 = p̄。"""
    p_bar = train.groupby("skill_id")["correct"].mean()
    stat = train.groupby(["skill_id", "problem_id"])["correct"].agg(k="sum", n="size")
    idx = pd.MultiIndex.from_frame(rows[["skill_id", "problem_id"]])
    k = stat["k"].reindex(idx).fillna(0).to_numpy()
    n = stat["n"].reindex(idx).fillna(0).to_numpy()
    m = rows["skill_id"].map(ITEM_M).to_numpy(float)
    p = rows["skill_id"].map(p_bar).to_numpy(float)
    return pd.Series((k + m * p) / (n + m), index=rows.index)


def inner_groups(train: pd.DataFrame) -> np.ndarray:
    """把訓練學生再切成 5 組（以 user_id 分組，shuffle + 固定種子），回傳每列的組別 0–4。"""
    gkf = GroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
    grp = np.full(len(train), -1)
    for g, (_, va_idx) in enumerate(gkf.split(train, groups=train["user_id"])):
        grp[va_idx] = g
    assert (grp >= 0).all()
    return grp


def item_diff_train_oof(train: pd.DataFrame) -> pd.Series:
    """訓練列的題目難度：每列只用「其他 4 組」學生的作答，不含自己這一組（也不用 leave-one-out）。"""
    grp = inner_groups(train)
    out = pd.Series(np.nan, index=train.index)
    for g in range(INNER_SPLITS):
        mask = grp == g
        out[mask] = item_diff_encode(train[~mask], train[mask])
    assert out.notna().all()
    return out


# ---------------------------------------------------------------- 特徵

def _hist_mean(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    """第 1…t-1 列的平均（累計和扣掉自己再除以 t-1）；t=1 沒有歷史 → NaN。"""
    cum = x.groupby([df["user_id"], df["skill_id"]]).cumsum()
    n_prev = (df["t"] - 1).astype(float)
    return ((cum - x) / n_prev.where(n_prev > 0))


def _last(df: pd.DataFrame, x: pd.Series) -> pd.Series:
    """上一列（同一學生同一技能）的值；t=1 → NaN。"""
    return x.groupby([df["user_id"], df["skill_id"]]).shift(1)


def transform(df: pd.DataFrame, stats: dict, item_diff: pd.Series) -> pd.DataFrame:
    """用已經算好的訓練集統計量，算 df 每一列的 R1 特徵（R0d 是它的子集）。

    item_diff 由呼叫端給（訓練列用 item_diff_train_oof，驗證列用 item_diff_encode）。
    沒有歷史（t=1）的缺失一律填訓練集平均。
    """
    _check_sorted(df)
    fills = stats["fills"]
    sig = _row_signals(df)
    # 缺失的標準化時間先填訓練集平均，再算歷史平均／上一列
    std_t = _std_time(df, stats["p1"], stats["p99"], stats["item_log_median"]).fillna(fills["log_time"])

    out = df[ID_COLS].copy()
    out["log1p_c"], out["log1p_w"] = df["log1p_c"], df["log1p_w"]       # R0 既有特徵（次數類，t=1 時是 0）
    out["skill_311"], out["has_history"] = df["skill_311"], df["has_history"]
    out["item_diff"] = item_diff
    out["last_correct"] = _last(df, df["correct"].astype(float)).fillna(fills["last_correct"])

    out["hint_rate"] = _hist_mean(df, sig["hint_flag"]).fillna(fills["hint"])
    out["last_hint"] = _last(df, sig["hint_flag"]).fillna(fills["hint"])
    out["mean_log_time"] = _hist_mean(df, std_t).fillna(fills["log_time"])
    out["last_log_time"] = _last(df, std_t).fillna(fills["log_time"])
    out["mean_log_att"] = _hist_mean(df, sig["log_att"]).fillna(fills["log_att"])
    out["last_log_att"] = _last(df, sig["log_att"]).fillna(fills["log_att"])
    out["scaffold_rate"] = _hist_mean(df, sig["scaffold_flag"]).fillna(fills["scaffold"])
    return out


def make_fold_features(train: pd.DataFrame, val: pd.DataFrame):
    """CV 的一個 fold：訓練列、驗證列各自的特徵。統計量全由 train 算。

    回傳 (train_feat, val_feat, stats)。test 學生這裡完全不碰：任何 role != trainval 的列都會報錯。
    """
    for name, d in [("train", train), ("val", val)]:
        assert (d["role"] == "trainval").all(), f"{name} 含 role != trainval 的列（test 不能碰）"
    assert set(train["user_id"]).isdisjoint(set(val["user_id"])), "訓練與驗證學生有交集"

    stats = fit_stats(train)
    train_feat = transform(train, stats, item_diff_train_oof(train))
    val_feat = transform(val, stats, item_diff_encode(train, val))
    return train_feat, val_feat, stats
