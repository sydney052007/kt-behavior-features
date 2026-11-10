"""跨學校補充實驗：LR 的 R0（沒有 item_diff、last_correct）在學校群分組下，檢查 R0 → R0d 的增益能不能搬到新學校。

    python3 src/run_xschool_r0.py > results/03_process/xschool_r0_run.log 2>&1
規則在 CLAUDE.md「跨學校補充實驗」，跑之前登記，不依結果修改：
  * 分組沿用 data/split_groups.csv（35 組、5 fold），不重新產生；test 學生完全不碰。
  * R0 特徵 = R0_FEATURES；LR = StandardScaler + LogisticRegression(C=np.inf)。
  * R0d 的各 fold ROC-AUC 沿用 results/02_analysis/xschool_a_folds.csv。C 的配對 bootstrap 需要 R0d 的逐列預測，
    而那份預測當時沒有存，所以這裡把 LR 的 R0d 在同一組分組下重算一次（LR 是確定的），並確認各 fold AUC
    與 xschool_a_folds.csv 相同才繼續。
  * 判讀規則（A 的未加權 5 fold 平均 ΔAUC = R0d − R0）：≥ +0.06 可搬移；< +0.04 有相當比例是學校／班級效應；其間不下結論。
輸出（results/0*/xschool_r0_*.csv，只有彙總數字，沒有 user_id、school_id）。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from cv import run_cv  # noqa: E402
from evaluate import paired_delta_c, per_fold_metric  # noqa: E402
from features import R0_FEATURES  # noqa: E402
from features_r1 import R0D_FEATURES  # noqa: E402
from run_r1_lr import make_model as make_lr  # noqa: E402
from run_xschool import by_cluster, caching_builder  # noqa: E402
from xschool import load_xschool_frame  # noqa: E402

FOLD_COLS = [f"fold{k}" for k in range(5)]
HIGH, LOW = 0.06, 0.04                    # 判讀規則的門檻（跑之前寫死）


def classify(delta_mean: float) -> str:
    """依 A 的未加權 5 fold 平均 ΔAUC 套用事先登記的判讀規則。"""
    if delta_mean >= HIGH:
        return "item_diff 的增益大部分可搬移到新學校"
    if delta_mean < LOW:
        return "item_diff 有相當比例是學校／班級效應"
    return "介於兩者：照實報告，不下結論"


def student_split_reference():
    """學生分組下的對照值：從既有 csv 重算（不使用四捨五入值）。"""
    cv = pd.read_csv("results/02_analysis/r1_lr_cv.csv")
    cv = cv[(cv["eval"] == "A") & (cv["metric"] == "roc_auc")].set_index("model")
    p = pd.read_csv("results/02_analysis/r1_lr_paired.csv")
    p = p[(p["eval"] == "A") & (p["comparison"] == "R0d - R0")].iloc[0]
    ref = dict(r0=cv.at["R0", "value"], r0d=cv.at["R0d", "value"],
               diff_folds=p[FOLD_COLS].to_numpy(float), diff_mean=p["delta_auc"], diff_std=p["std"],
               n_better=int(p["n_folds_better"]))
    assert abs((ref["r0d"] - ref["r0"]) - ref["diff_mean"]) < 1e-9          # 兩個 csv 彼此一致
    return ref


def main() -> None:
    tv = load_xschool_frame()                                    # 只有 trainval；fold = 跨學校 fold
    assert (tv["role"] == "trainval").all() and len(tv) == 30516
    assert "item_diff" not in R0_FEATURES and "last_correct" not in R0_FEATURES
    grp_of = tv.set_index("order_id")["grp"]
    print(f"[xschool_r0] trainval {len(tv):,} 列；{tv['grp'].nunique()} 組", flush=True)

    r0 = run_cv(tv, R0_FEATURES, make_lr)                        # R0 特徵是現成的因果特徵，不需要訓練集統計量
    r0d = run_cv(tv, R0D_FEATURES, make_lr, feature_builder=caching_builder())
    r0_oof = r0.oof.assign(grp=r0.oof["order_id"].map(grp_of).to_numpy())
    r0d_oof = r0d.oof.assign(grp=r0d.oof["order_id"].map(grp_of).to_numpy())

    # R0d 的各 fold ROC-AUC 沿用 xschool_a_folds.csv；重算的必須與它相同
    old = pd.read_csv("results/02_analysis/xschool_a_folds.csv")
    old = old[(old.model_type == "LR") & (old.features == "R0d")].set_index("fold").sort_index()
    r0d_auc = old["roc_auc"]
    assert np.abs(per_fold_metric(r0d_oof).to_numpy() - r0d_auc.to_numpy()).max() < 1e-9, "重算的 R0d 與 xschool_a_folds.csv 不同"
    rows_per_fold = old["n_rows"].astype(float)

    roc, pr = per_fold_metric(r0_oof), per_fold_metric(r0_oof, "pr_auc")
    d = r0d_auc - roc
    weighted = float(np.average(d.to_numpy(), weights=rows_per_fold.to_numpy()))
    verdict = classify(d.mean())

    fold_tab = pd.DataFrame({"fold": roc.index, "n_rows": rows_per_fold.to_numpy().astype(int), "r0_roc_auc": roc.to_numpy(),
                             "r0_pr_auc": pr.to_numpy(), "r0d_roc_auc": r0d_auc.to_numpy(), "diff_roc_auc": d.to_numpy()})
    fold_tab.to_csv("results/02_analysis/xschool_r0_a_folds.csv", index=False)
    summary = pd.DataFrame([
        dict(item="R0 ROC-AUC", mean=roc.mean(), std=roc.std(ddof=1)),
        dict(item="R0 PR-AUC", mean=pr.mean(), std=pr.std(ddof=1)),
        dict(item="R0d ROC-AUC（沿用 xschool_a_folds.csv）", mean=r0d_auc.mean(), std=r0d_auc.std(ddof=1)),
        dict(item="R0d − R0 ROC-AUC（未加權平均）", mean=d.mean(), std=d.std(ddof=1), n_folds_better=int((d > 0).sum()),
             weighted_mean=weighted, verdict=verdict)])
    summary.to_csv("results/02_analysis/xschool_r0_a_summary.csv", index=False)

    # ---- 學生分組對照並排
    ref = student_split_reference()
    side = pd.DataFrame([
        dict(split="學生分組（既有 csv）", r0_roc_auc=ref["r0"], r0d_roc_auc=ref["r0d"], diff_mean=ref["diff_mean"],
             diff_std=ref["diff_std"], n_folds_better=ref["n_better"], **{f"diff_fold{k}": v for k, v in enumerate(ref["diff_folds"])}),
        dict(split="學校群分組（本實驗）", r0_roc_auc=roc.mean(), r0d_roc_auc=r0d_auc.mean(), diff_mean=d.mean(),
             diff_std=d.std(ddof=1), n_folds_better=int((d > 0).sum()), **{f"diff_fold{int(k)}": d[k] for k in d.index})])
    side.to_csv("results/02_analysis/xschool_r0_vs_student_split.csv", index=False)

    big_fold = int(pd.read_csv("results/01_preprocessing/xschool_folds.csv").query("contains_largest_group").iloc[0]["fold"])
    f0 = pd.DataFrame([dict(fold=big_fold, n_rows=int(rows_per_fold[big_fold]), r0_roc_auc=roc[big_fold],
                            r0d_roc_auc=r0d_auc[big_fold], diff_roc_auc=d[big_fold])])
    f0.to_csv("results/02_analysis/xschool_r0_fold0.csv", index=False)

    c = paired_delta_c(by_cluster(r0_oof), by_cluster(r0d_oof))   # Δ = R0d − R0；重抽單位 = 組別
    c = c.rename(columns={"delta_auc": "delta_roc_auc"})
    c.to_csv("results/02_analysis/xschool_r0_paired_c.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n=== 判讀規則（事先登記）===")
    print(f"  ≥ {HIGH:+.2f}：item_diff 的增益大部分可搬移到新學校；< {LOW:+.2f}：有相當比例是學校／班級效應；其間：照實報告，不下結論")
    print(f"\n=== 學生分組的對照值（由 csv 重算）===\n  R0 {ref['r0']:.6f}  R0d {ref['r0d']:.6f}  差 {ref['diff_mean']:+.6f}")
    print("\n=== A：R0 各 fold（學校群分組）===")
    print(fold_tab.round(4).to_string(index=False))
    print(f"  R0 ROC-AUC {roc.mean():.4f} ± {roc.std(ddof=1):.4f}；PR-AUC {pr.mean():.4f} ± {pr.std(ddof=1):.4f}")
    print(f"\n=== R0d − R0（學校群分組；ΔAUC）===\n  各 fold [" + ", ".join(f"{x:+.4f}" for x in d) + f"]\n"
          f"  未加權平均 {d.mean():+.4f}  標準差 {d.std(ddof=1):.4f}  較好 {int((d > 0).sum())}/5  列數加權平均 {weighted:+.4f}")
    print(f"  判讀（依登記規則）：{verdict}")
    print("\n=== 並排：學生分組 vs 學校群分組 ===")
    print(side[["split", "r0_roc_auc", "r0d_roc_auc", "diff_mean", "diff_std", "n_folds_better"]].round(4).to_string(index=False))
    print(f"\n=== 最大組所在的 fold {big_fold} ===")
    print(f0.round(4).to_string(index=False))
    print("\n=== C：R0d − R0 的配對 ΔAUC（95% CI，組別為單位的 cluster bootstrap，2000 次、種子 42）===")
    for _, r in c.iterrows():
        print(f"  {r['group']:5s} {r['delta_roc_auc']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  (序列數 {int(r['n_rows'])})")


if __name__ == "__main__":
    main()
