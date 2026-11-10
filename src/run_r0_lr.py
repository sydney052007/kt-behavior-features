"""R0 + LR 的 5-fold 結果（W2）。只用 trainval 學生；test 完全不碰。

    python3 src/run_r0_lr.py
輸出：
    data/interim/oof_r0_lr.csv   OOF 預測（學生層級，不進 git）
    results/02_analysis/r0_lr_cv.csv         彙總表（只有彙總數字）
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.dirname(__file__))
from cv import run_cv  # noqa: E402
from evaluate import evaluate  # noqa: E402
from features import OUT_PATH, R0_FEATURES  # noqa: E402

OOF_PATH = "data/interim/oof_r0_lr.csv"
RESULT_PATH = "results/02_analysis/r0_lr_cv.csv"


def make_model():
    # R0 的 LR 不加正則化（CLAUDE.md）：C=np.inf 等同 penalty=None，但不會觸發 sklearn 1.8 之後的棄用警告
    return LogisticRegression(C=np.inf, max_iter=1000)


def main() -> None:
    df = pd.read_csv(OUT_PATH, dtype={"skill_id": str})
    n_test = int((df["role"] == "test").sum())
    df = df[df["role"] == "trainval"].reset_index(drop=True)   # 明確丟掉 test 列
    print(f"[r0_lr] 丟掉 test 列 {n_test:,}；使用 trainval {len(df):,} 列")
    assert (df["role"] == "trainval").all() and df["fold"].between(0, 4).all()

    res = run_cv(df, R0_FEATURES, make_model)
    res.oof.to_csv(OOF_PATH, index=False)
    print(f"[r0_lr] OOF 已存 {OOF_PATH}（{len(res.oof):,} 列）")

    table = evaluate(res.oof, res.thresholds)
    table.to_csv(RESULT_PATH, index=False)

    pd.set_option("display.width", 200)
    print("\n" + table.round(4).to_string(index=False))
    print(f"\n[r0_lr] 彙總表已存 {RESULT_PATH}")


if __name__ == "__main__":
    main()
