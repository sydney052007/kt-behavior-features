"""補 LR（R0／R0d／R1）在 A 的各 fold 絕對 ROC-AUC 與 PR-AUC（學生分組的同一個 split）。

    python3 src/make_lr_fold_scores.py
流程與 run_r1_lr.py 完全相同（StandardScaler + LogisticRegression(C=np.inf)、每個 fold 呼叫 make_fold_features、
同一個 split_users.csv），只是把每個 fold 的分數存下來；不存 OOF 預測，也不覆蓋 data/interim 內的任何檔案。
輸出 results/02_analysis/r1_lr_folds.csv（長格式：model, fold, n_rows, roc_auc, pr_auc），
並印出 5 個 fold 的平均、標準差與 results/02_analysis/r1_lr_cv.csv 的比對（差 < 1e-9 才算通過）。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cv import run_cv  # noqa: E402
from evaluate import per_fold_metric  # noqa: E402
from features_r1 import load_r1_frame, make_fold_features  # noqa: E402
from run_r1_lr import FEATURE_SETS, make_model  # noqa: E402

OUT_PATH = "results/02_analysis/r1_lr_folds.csv"
REF_PATH = "results/02_analysis/r1_lr_cv.csv"


def main() -> None:
    df = load_r1_frame()
    tv = df[df["role"] == "trainval"].reset_index(drop=True)        # 只用 trainval；test 完全不碰
    assert (tv["role"] == "trainval").all() and tv["fold"].between(0, 4).all()
    rows = []
    for name, feats in FEATURE_SETS.items():
        res = run_cv(tv, feats, make_model, feature_builder=make_fold_features)
        roc, pr = per_fold_metric(res.oof), per_fold_metric(res.oof, "pr_auc")
        n = res.oof.groupby("fold").size()
        rows += [dict(model=name, fold=int(f), n_rows=int(n[f]), roc_auc=roc[f], pr_auc=pr[f]) for f in roc.index]
    out = pd.DataFrame(rows)
    out.to_csv(OUT_PATH, index=False)

    ref = pd.read_csv(REF_PATH)
    ref = ref[ref["eval"] == "A"].set_index(["model", "metric"])
    print("=== 各 fold 的絕對分數 ===")
    print(out.pivot(index="fold", columns="model", values=["roc_auc", "pr_auc"]).round(4).to_string())
    print("\n=== 與 r1_lr_cv.csv 的比對（5 個 fold 的平均與標準差）===")
    worst = 0.0
    for name in FEATURE_SETS:
        for m in ["roc_auc", "pr_auc"]:
            g = out[out.model == name][m]
            d_mean = abs(g.mean() - ref.at[(name, m), "value"])
            d_std = abs(g.std(ddof=1) - ref.at[(name, m), "std"])
            worst = max(worst, d_mean, d_std)
            print(f"  {name:4s} {m:8s} 平均 {g.mean():.10f} vs csv {ref.at[(name, m), 'value']:.10f} (差 {d_mean:.1e})"
                  f"   標準差 {g.std(ddof=1):.10f} vs {ref.at[(name, m), 'std']:.10f} (差 {d_std:.1e})")
    assert worst < 1e-9, f"與 r1_lr_cv.csv 不一致（最大差 {worst:.2e}）"
    print(f"\n比對通過：最大差 {worst:.1e} < 1e-9；已存 {OUT_PATH}")


if __name__ == "__main__":
    main()
