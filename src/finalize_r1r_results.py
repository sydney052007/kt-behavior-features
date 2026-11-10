"""把 run_r1r.py 的原始輸出整理成最後存檔用的彙總 csv（不重跑任何實驗，只切分與計算差值）。

    python3 src/run_r1r.py > results/03_process/r1r_run.log 2>&1     # 先跑實驗（原始輸出）
    python3 src/finalize_r1r_results.py                    # 再整理

讀：results/02_analysis/r1r_paired.csv（原始，含 own_best／same_cfg_R1best／todo_pr 三種分析）、
    results/01_preprocessing/r1r_new_features.csv、results/02_analysis/r1r_a_summary.csv
寫：
    r1r_paired.csv              只留 own_best：R1r−R1、R0d_rel−R0d（LR 與 RF；A 逐 fold、C 的 t=6／t=11）
    r1r_paired_same_config.csv  補充：RF 兩邊都用 R1 的最佳設定
    r1_minus_r0d_pr_auc.csv     R1−R0d 的 ΔPR-AUC（LR 與 RF）
    r1r_feature_stats.csv       4 個新特徵的平均、標準差、最小、最大、與 correct 的相關
    r1r_vs_r0d_rel.csv          R1r 與 R0d_rel 的 A 比較（ROC-AUC、PR-AUC 與差值）
全部只有彙總數字。
"""
import os

import pandas as pd

P01, P02 = "results/01_preprocessing/", "results/02_analysis/"      # 前處理與描述／最終分析表


def main() -> None:
    paired = pd.read_csv(P02 + "r1r_paired.csv")
    assert {"own_best", "same_cfg_R1best", "todo_pr"} <= set(paired["analysis"]), \
        "r1r_paired.csv 已經整理過了，請先重跑 run_r1r.py 再執行這個腳本"

    paired[paired["analysis"] == "own_best"].to_csv(P02 + "r1r_paired.csv", index=False)
    paired[paired["analysis"] == "same_cfg_R1best"].to_csv(P02 + "r1r_paired_same_config.csv", index=False)
    paired[paired["analysis"] == "todo_pr"].to_csv(P02 + "r1_minus_r0d_pr_auc.csv", index=False)

    # 新特徵彙總：只是改檔名
    pd.read_csv(P01 + "r1r_new_features.csv").to_csv(P01 + "r1r_feature_stats.csv", index=False)
    os.remove(P01 + "r1r_new_features.csv")

    # R1r 與 R0d_rel 的 A 比較：差值 = R1r − R0d_rel（兩邊都有 4 個相對特徵，所以差值是 R1 比 R0d 多的
    # 7 個行為特徵在『已有相對特徵』時的額外貢獻）。這裡只有各 fold 平均的差；沒有配對信賴區間。
    s = pd.read_csv(P02 + "r1r_a_summary.csv")
    s = s[s["features"].isin(["R1r", "R0d_rel"])]
    rows = []
    for mt in ["LR", "RF"]:
        a = s[(s.model_type == mt) & (s.features == "R1r")].iloc[0]
        b = s[(s.model_type == mt) & (s.features == "R0d_rel")].iloc[0]
        for metric in ["roc_auc", "pr_auc"]:
            rows.append(dict(model_type=mt, metric=metric,
                             r1r_setting=a["setting"], r0d_rel_setting=b["setting"],
                             r1r_mean=a[f"{metric}_mean"], r1r_std=a[f"{metric}_std"],
                             r0d_rel_mean=b[f"{metric}_mean"], r0d_rel_std=b[f"{metric}_std"],
                             diff_r1r_minus_r0d_rel=a[f"{metric}_mean"] - b[f"{metric}_mean"]))
    pd.DataFrame(rows).to_csv(P02 + "r1r_vs_r0d_rel.csv", index=False)
    print("已寫出 r1r_paired.csv、r1r_paired_same_config.csv、r1_minus_r0d_pr_auc.csv、"
          "r1r_feature_stats.csv、r1r_vs_r0d_rel.csv")


if __name__ == "__main__":
    main()
