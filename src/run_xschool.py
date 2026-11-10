"""跨學校 5-fold：R0d vs R1，LR 與 RF（規則在 CLAUDE.md「跨學校實驗規則」，跑之前登記，不依結果修改）。

    python3 src/make_xschool_groups.py      # 步驟 0：分組（只需做一次）
    python3 src/run_xschool.py > results/03_process/xschool_run.log 2>&1
模型（固定，不搜尋）：LR = StandardScaler + LogisticRegression(C=np.inf)；
RF = 300 棵、sqrt、種子 42、n_jobs=-1；R0d 用 max_depth=8、min_samples_leaf=5；R1 用 max_depth=None、min_samples_leaf=50。
評估：A（每個 fold）、B 位置分桶、C 固定切點 t=6／11（C 的 cluster bootstrap 以『組別』為單位）。

輸出（results/0*/xschool_*.csv，只有彙總數字，沒有 user_id、school_id）：
    xschool_cv.csv  xschool_a_folds.csv  xschool_paired.csv  xschool_fold0.csv  xschool_unseen_problems.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from cv import run_cv  # noqa: E402
from evaluate import eval_a, eval_b, eval_c, paired_delta_c, per_fold_metric, to_wrong  # noqa: E402
from features_r1 import R0D_FEATURES, R1_FEATURES  # noqa: E402
from run_r1_lr import make_model as make_lr  # noqa: E402
from run_rf import make_rf  # noqa: E402
from xschool import load_xschool_frame, make_fold_features_xschool  # noqa: E402

FOLD_COLS = [f"fold{k}" for k in range(5)]
FEATURES = {"R0d": R0D_FEATURES, "R1": R1_FEATURES}
RF_SETTING = {"R0d": (8, 5), "R1": (None, 50)}          # (max_depth, min_samples_leaf)；不重新搜尋


def caching_builder():
    """同一個 fold 的特徵只算一次（以驗證 fold 編號當 key），四個模型共用。"""
    cache = {}

    def build(train, val):
        key = tuple(sorted(val["fold"].unique()))
        if key not in cache:
            cache[key] = make_fold_features_xschool(train, val)
        return cache[key]
    return build


def fold_table(res):
    """每個 fold 的 A 指標（閾值用該 fold 訓練列選好的 res.thresholds）。"""
    rows = []
    for f, g in res.oof.groupby("fold"):
        y, score = to_wrong(g["correct"], g["p_correct"])
        pred = (score >= res.thresholds[f]).astype(int)
        rows.append(dict(fold=f, n_rows=len(g), roc_auc=per_fold_metric(g)[f],
                         pr_auc=per_fold_metric(g, "pr_auc")[f], f1=f1_score(y, pred, zero_division=0),
                         accuracy=accuracy_score(y, pred), threshold=res.thresholds[f]))
    return pd.DataFrame(rows)


def by_cluster(oof):
    """C 的 bootstrap 以『組別』為單位：把 user_id 欄換成組別（evaluate 的重抽單位就是這一欄）。"""
    return oof.assign(user_id=oof["grp"])


def pair_rows(model_type, new, ref, new_oof, ref_oof, weights):
    d = per_fold_metric(new_oof) - per_fold_metric(ref_oof)
    base = dict(model_type=model_type, comparison=f"{new} - {ref}")
    rows = [dict(base, eval="A", group="all_rows", delta=d.mean(), std=d.std(ddof=1), n_folds_better=int((d > 0).sum()),
                 weighted_mean=float(np.average(d.to_numpy(), weights=weights.loc[d.index].to_numpy())),
                 **{f"fold{f}": d[f] for f in d.index})]
    for _, r in paired_delta_c(by_cluster(ref_oof), by_cluster(new_oof)).iterrows():
        rows.append(dict(base, eval="C", group=r["group"], delta=r["delta_auc"], ci_low=r["ci_low"], ci_high=r["ci_high"]))
    return rows


def main() -> None:
    t0 = time.time()
    tv = load_xschool_frame()                                    # 只有 trainval；fold = 跨學校 fold
    assert (tv["role"] == "trainval").all() and len(tv) == 30516
    print(f"[xschool] trainval {len(tv):,} 列；{tv['grp'].nunique()} 組；跨學校 fold 的列數 "
          f"{tv.groupby('fold').size().tolist()}", flush=True)
    builder = caching_builder()
    grp_of = tv.set_index("order_id")["grp"]

    runs = {}
    for mt in ["LR", "RF"]:
        for name, feats in FEATURES.items():
            maker = make_lr if mt == "LR" else make_rf(*RF_SETTING[name])
            res = run_cv(tv, feats, maker, feature_builder=builder)
            res = res._replace(models=None, oof=res.oof.assign(grp=res.oof["order_id"].map(grp_of).to_numpy()))
            runs[(mt, name)] = res
            a = per_fold_metric(res.oof)
            print(f"[run] {mt} {name:4s} A ROC-AUC {a.mean():.4f} ± {a.std(ddof=1):.4f}   (累計 {time.time() - t0:.0f}s)", flush=True)

    # ---- A／B／C
    cv_rows, fold_rows = [], []
    for (mt, name), res in runs.items():
        meta = dict(model_type=mt, features=name,
                    setting="StandardScaler+LR" if mt == "LR" else f"depth={RF_SETTING[name][0]}, leaf={RF_SETTING[name][1]}")
        for r in eval_a(res.oof, res.thresholds) + eval_b(res.oof):
            cv_rows.append(dict(meta, **r))
        for r in eval_c(by_cluster(res.oof)):                    # CI：以組別為單位重抽
            r = dict(r)
            r["n_clusters"] = r.pop("n_students")
            cut = int(r["group"].split("=")[1])
            r["n_students"] = res.oof.loc[res.oof["t"] == cut, "user_id"].nunique()
            cv_rows.append(dict(meta, **r))
        ft = fold_table(res)
        for _, r in ft.iterrows():
            fold_rows.append(dict(meta, **r.to_dict()))
    cv = pd.DataFrame(cv_rows)
    cv.to_csv("results/02_analysis/xschool_cv.csv", index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv("results/02_analysis/xschool_a_folds.csv", index=False)

    pd.set_option("display.width", 220)
    print("\n=== A 全部列（5 個 fold 平均 ± 標準差）===")
    a_tab = cv[cv["eval"] == "A"]
    for (mt, name) in runs:
        t = a_tab[(a_tab.model_type == mt) & (a_tab.features == name)].set_index("metric")
        print(f"  {mt} {name:4s} " + "   ".join(f"{m} {t.at[m, 'value']:.4f} ± {t.at[m, 'std']:.4f}"
                                                for m in ["roc_auc", "pr_auc", "f1", "accuracy"]) +
              f"   (基準 Accuracy {t.at['acc_baseline_all_correct', 'value']:.3f})")
    print("\n=== 各 fold 的 A ROC-AUC／PR-AUC ===")
    pv = folds.pivot_table(index="fold", columns=["model_type", "features"], values=["roc_auc", "pr_auc"])
    pv.insert(0, "rows", folds[(folds.model_type == "LR") & (folds.features == "R0d")].set_index("fold")["n_rows"])
    print(pv.round(4).to_string())
    print("\n=== B 位置分桶（合併 OOF；ROC-AUC）===")
    b = cv[(cv["eval"] == "B") & (cv.metric == "roc_auc")].pivot_table(index="group", columns=["model_type", "features"], values="value", sort=False)
    print(b.round(4).to_string())
    print("\n=== C 固定切點（95% CI，以組別為單位的 cluster bootstrap）===")
    c = cv[(cv["eval"] == "C") & cv.metric.isin(["roc_auc", "pr_auc"])]
    for g in c["group"].unique():
        n = c[(c.group == g)].iloc[0]
        print(f"  {g}（序列數 {int(n.n_rows)}，學生數 {int(n.n_students)}，組數 {int(n.n_clusters)}）")
        for m in ["roc_auc", "pr_auc"]:
            print(f"    {m:8s}" + "   ".join(
                f"{mt} {nm}: {r['value']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
                for (mt, nm) in runs for r in [c[(c.group == g) & (c.model_type == mt) & (c.features == nm) & (c.metric == m)].iloc[0]]))

    # ---- 配對比較 R1 − R0d
    weights = folds[(folds.model_type == "LR") & (folds.features == "R0d")].set_index("fold")["n_rows"].astype(float)
    paired = []
    for mt in ["LR", "RF"]:
        paired += pair_rows(mt, "R1", "R0d", runs[(mt, "R1")].oof, runs[(mt, "R0d")].oof, weights)
    paired = pd.DataFrame(paired)
    paired.to_csv("results/02_analysis/xschool_paired.csv", index=False)
    print("\n=== 配對差異 R1 − R0d（ΔAUC）===")
    for _, r in paired[paired["eval"] == "A"].iterrows():
        print(f"  {r.model_type} A 各 fold [" + ", ".join(f"{r[f]:+.4f}" for f in FOLD_COLS) +
              f"]  未加權平均 {r.delta:+.4f}  標準差 {r['std']:.4f}  較好 {int(r.n_folds_better)}/5  列數加權平均 {r.weighted_mean:+.4f}")
    for _, r in paired[paired["eval"] == "C"].iterrows():
        print(f"  {r.model_type} C {r.group:5s} {r.delta:+.4f}  [{r.ci_low:+.4f}, {r.ci_high:+.4f}]")

    # ---- 最大組所在 fold
    big_fold = int(pd.read_csv("results/01_preprocessing/xschool_folds.csv").query("contains_largest_group").iloc[0]["fold"])
    f0 = []
    for mt in ["LR", "RF"]:
        r0 = folds[(folds.model_type == mt) & (folds.features == "R0d") & (folds.fold == big_fold)].iloc[0]
        r1 = folds[(folds.model_type == mt) & (folds.features == "R1") & (folds.fold == big_fold)].iloc[0]
        f0.append(dict(model_type=mt, fold=big_fold, n_rows=int(r0.n_rows), roc_auc_R0d=r0.roc_auc, roc_auc_R1=r1.roc_auc,
                       delta_roc_auc=r1.roc_auc - r0.roc_auc, pr_auc_R0d=r0.pr_auc, pr_auc_R1=r1.pr_auc,
                       delta_pr_auc=r1.pr_auc - r0.pr_auc))
    f0 = pd.DataFrame(f0)
    f0.to_csv("results/02_analysis/xschool_fold0.csv", index=False)
    print(f"\n=== 最大組所在的 fold（fold {big_fold}）===")
    print(f0.round(4).to_string(index=False))

    # ---- 驗證列中訓練集沒出現過的 problem_id 的比例（跨學校 vs 原本的學生分組）
    rows = []
    for k in range(5):
        row = dict(fold=k)
        for tag, col in [("xschool", "fold"), ("student_split", "student_fold")]:
            tr, va = tv[tv[col] != k], tv[tv[col] == k]
            unseen = ~va["problem_id"].isin(set(tr["problem_id"]))
            row[f"{tag}_val_rows"], row[f"{tag}_unseen_rows"], row[f"{tag}_unseen_frac"] = len(va), int(unseen.sum()), unseen.mean()
        rows.append(row)
    un = pd.DataFrame(rows)
    un.to_csv("results/01_preprocessing/xschool_unseen_problems.csv", index=False)
    print("\n=== 驗證列中『訓練集沒出現過的 problem_id』的比例 ===")
    print(un[["fold", "xschool_val_rows", "xschool_unseen_rows", "xschool_unseen_frac",
              "student_split_val_rows", "student_split_unseen_rows", "student_split_unseen_frac"]].round(4).to_string(index=False))
    print(f"\n[xschool] 完成，共 {(time.time() - t0) / 60:.1f} 分鐘；已存 results/0*/xschool_*.csv", flush=True)


if __name__ == "__main__":
    main()
