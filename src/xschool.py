"""跨學校 5-fold 的分組與特徵（規則見 CLAUDE.md「跨學校實驗規則」）。

    分組：在 trainval 上，共用同一個 student_class_id 或 teacher_id 的學校併成同一組（union-find），
          再用 GroupKFold(n_splits=5)（不 shuffle，確定性）依組別切 fold。
          「學校 → 組別 → fold」存在 data/split_groups.csv（untracked；存好之後不重新產生）。
    特徵：R0d／R1 沿用 features_r1.py（不改）；唯一不同是 item_diff 訓練列用的 inner 5-fold
          改成依『組別』分組（原本依 user_id），所以同一組的列不會互相提供題目難度。
test 學生這裡完全不碰：只讀 role == trainval 的列。
"""
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

import features_r1 as fr
from features_r1 import INNER_SEED, INNER_SPLITS, item_diff_encode, load_r1_frame

GROUPS_PATH = "data/split_groups.csv"          # 欄位只有 school_id, group_id, fold
ORG_COLS = ["school_id", "teacher_id", "student_class_id"]
KEY = ["user_id", "skill_id", "order_id"]
N_FOLDS = 5
MAX_GROUP_SHARE = 0.40                         # 最大組超過 trainval 列數的 40% → 停下來回報
MIN_GROUPS = 5


def load_trainval_with_org() -> pd.DataFrame:
    """trainval 列（test 完全不讀進來）+ school_id、teacher_id、student_class_id。"""
    df = load_r1_frame()
    tv = df[df["role"] == "trainval"].reset_index(drop=True)
    assert (tv["role"] == "trainval").all() and len(tv) == 30516
    org = pd.read_csv(fr.MAIN_PATH, usecols=KEY + ORG_COLS, dtype={"skill_id": str})
    out = tv.merge(org, on=KEY, how="left", validate="one_to_one")
    assert len(out) == len(tv) and out[ORG_COLS].notna().all().all()
    return out


def build_groups(tv: pd.DataFrame) -> pd.Series:
    """union-find：共用同一個 class 或 teacher 的學校併成同一組。回傳 school_id → group_id。

    group_id 依『組內最小的 school_id』由小到大編號 0…，所以編號是確定的。
    """
    schools = sorted(tv["school_id"].unique())
    parent = {s: s for s in schools}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for col in ["student_class_id", "teacher_id"]:
        for _, g in tv.groupby(col)["school_id"]:
            s = g.unique()
            for other in s[1:]:
                ra, rb = find(s[0]), find(other)
                if ra != rb:
                    parent[max(ra, rb)] = min(ra, rb)
    roots = sorted({find(s) for s in schools})
    gid = {r: i for i, r in enumerate(roots)}
    return pd.Series({s: gid[find(s)] for s in schools}, name="group_id")


def assign_group_folds(tv: pd.DataFrame, school_to_group: pd.Series) -> pd.Series:
    """GroupKFold(n_splits=5)（不 shuffle）依組別切；回傳 group_id → fold。"""
    grp = tv["school_id"].map(school_to_group)
    fold = {}
    for k, (_, va) in enumerate(GroupKFold(n_splits=N_FOLDS).split(tv, groups=grp)):
        for g in grp.iloc[va].unique():
            fold[g] = k
    return pd.Series(fold, name="fold")


def load_xschool_frame() -> pd.DataFrame:
    """trainval 列 + 組別 grp + 跨學校 fold。

    欄位 fold 換成跨學校 fold（讓 run_cv 直接用）；原本依學生的 fold 另存成 student_fold。
    """
    tv = load_trainval_with_org()
    groups = pd.read_csv(GROUPS_PATH)
    assert list(groups.columns) == ["school_id", "group_id", "fold"]
    out = tv.merge(groups, on="school_id", how="left", validate="many_to_one", suffixes=("", "_x"))
    assert out["group_id"].notna().all()
    out = out.rename(columns={"fold": "student_fold"})
    out["fold"] = out["fold_x"].astype(int)
    out["grp"] = out["group_id"].astype(int)
    return out.drop(columns=["fold_x", "group_id"])


# ---------------------------------------------------------------- 特徵：inner fold 依組別

def inner_groups_by_group(train: pd.DataFrame) -> np.ndarray:
    """訓練列再切成 5 個 inner 組（以『組別』分組、shuffle + 固定種子）；同一組不會跨 inner fold。"""
    gkf = GroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=INNER_SEED)
    inner = np.full(len(train), -1)
    for g, (_, va_idx) in enumerate(gkf.split(train, groups=train["grp"])):
        inner[va_idx] = g
    assert (inner >= 0).all()
    return inner


def item_diff_train_oof_groups(train: pd.DataFrame, inner: np.ndarray) -> pd.Series:
    """訓練列的題目難度：每列只用 inner 分組中『其他 4 組』的作答（與 features_r1 相同的算法，只是分組不同）。"""
    out = pd.Series(np.nan, index=train.index)
    for g in range(INNER_SPLITS):
        mask = inner == g
        out[mask] = item_diff_encode(train[~mask], train[mask])
    assert out.notna().all()
    return out


def make_fold_features_xschool(train: pd.DataFrame, val: pd.DataFrame):
    """跨學校的一個 fold：R1 的 13 個特徵（R0d 是其子集）。統計量全由 train 算；test 列直接報錯。

    和 features_r1.make_fold_features 唯一的差別是 inner 分組依組別。回傳 (train_feat, val_feat, stats)。
    """
    for name, d in [("train", train), ("val", val)]:
        assert (d["role"] == "trainval").all(), f"{name} 含 role != trainval 的列（test 不能碰）"
    assert set(train["grp"]).isdisjoint(set(val["grp"])), "訓練與驗證的組別有交集"
    assert set(train["user_id"]).isdisjoint(set(val["user_id"])), "訓練與驗證學生有交集"

    stats = fr.fit_stats(train)
    train_feat = fr.transform(train, stats, item_diff_train_oof_groups(train, inner_groups_by_group(train)))
    val_feat = fr.transform(val, stats, item_diff_encode(train, val))     # 驗證列：整個訓練集；沒見過 = p̄
    return train_feat, val_feat, stats
