"""5-fold 交叉驗證（依 split_users.csv 的 fold，不重新切）。可給 R0、R1 共用。

只允許 role == "trainval" 的列進來；任何 test 列都會讓程式直接報錯。
"""
from collections import namedtuple

import numpy as np
import pandas as pd

from evaluate import select_threshold

N_FOLDS = 5
# oof：驗證列的預測（學生層級資料，只能存 data/interim）
# thresholds：{fold: 閾值}，由該 fold 的訓練列選出
# fold_users：{fold: (訓練學生集合, 驗證學生集合)}，給測試檢查用，不輸出
CVResult = namedtuple("CVResult", ["oof", "thresholds", "fold_users"])


def run_cv(df, features, make_model) -> CVResult:
    """對每個 fold f：用 fold != f 訓練、預測 fold == f，最後合併成 OOF。

    make_model：沒有參數、回傳新的 sklearn 分類器（每個 fold 都要新模型）。
    """
    # 不碰 test：進來的資料就不能有 test 列，fold 也只能是 0–4
    assert (df["role"] == "trainval").all(), "資料裡有 role != trainval 的列（test 不能碰）"
    assert set(df["fold"].unique()) == set(range(N_FOLDS)), "fold 應該剛好是 0–4"

    oof_parts, thresholds, fold_users = [], {}, {}
    for f in range(N_FOLDS):
        train = df[df["fold"] != f]
        val = df[df["fold"] == f]

        # 同一位學生不能同時在訓練與驗證（兩個技能的列都在同一個 fold）
        tr_users, va_users = set(train["user_id"]), set(val["user_id"])
        assert tr_users.isdisjoint(va_users), f"fold {f}：訓練與驗證學生有交集"
        fold_users[f] = (tr_users, va_users)

        model = make_model()
        model.fit(train[features], train["correct"])
        classes = list(model.classes_)
        col = classes.index(1)                                # P(答對) 那一欄

        # 訓練列的預測只用來選閾值（F1 最大），不存檔
        p_train = model.predict_proba(train[features])[:, col]
        thresholds[f] = select_threshold(train["correct"].to_numpy(), p_train)

        part = val[["user_id", "skill_id", "order_id", "t", "fold", "correct"]].copy()
        part["p_correct"] = model.predict_proba(val[features])[:, col]
        oof_parts.append(part)

    oof = pd.concat(oof_parts, ignore_index=True)
    # OOF 必須剛好覆蓋每個列一次
    assert len(oof) == len(df) and oof["order_id"].is_unique
    assert set(oof["order_id"]) == set(df["order_id"])
    return CVResult(oof, thresholds, fold_users)
