"""R0 / R0d / R1 的 LR 5-fold 比較（W3）。只用 trainval 學生；test 完全不碰。

    python3 src/run_r1_lr.py
三組特徵、同一個模型（StandardScaler + LogisticRegression(C=np.inf)），
評估跟 R0 完全一樣（evaluate.py 的 A／B／C）。R0d、R1 的特徵在每個 fold 內
用 make_fold_features(train, val) 各算一次（統計量只來自該 fold 的訓練列）。

輸出（都只有彙總數字）：
    results/r1_lr_cv.csv       三組 × A／B／C（長格式，多一欄 model）
    results/r1_lr_paired.csv   配對差異（A 的逐 fold ΔAUC、C 的配對 bootstrap ΔAUC）
    results/r1_lr_coef.csv     R1 標準化後的 LR 係數（5 個 fold 的平均與標準差）
OOF 預測（學生層級）存 data/interim/oof_cmp_<model>.csv，不進 git。
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from cv import run_cv  # noqa: E402
from evaluate import evaluate, paired_delta_c, per_fold_auc  # noqa: E402
from features import R0_FEATURES  # noqa: E402
from features_r1 import R0D_FEATURES, R1_FEATURES, load_r1_frame, make_fold_features  # noqa: E402

FEATURE_SETS = {"R0": R0_FEATURES, "R0d": R0D_FEATURES, "R1": R1_FEATURES}
R0_REFERENCE = "results/r0_lr_cv.csv"      # 之前（沒有 StandardScaler）的 R0 結果，用來確認數字不變


def make_model():
    # 不加正則化（C=np.inf）；StandardScaler 只在每個 fold 的訓練列上 fit（Pipeline 保證）
    return Pipeline([("scale", StandardScaler()),
                     ("lr", LogisticRegression(C=np.inf, max_iter=1000))])


def run_one(tv, name):
    """跑一組特徵的 5-fold；回傳 (CVResult, 收斂警告次數)。"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_cv(tv, FEATURE_SETS[name], make_model, feature_builder=make_fold_features)
    n_conv = sum(issubclass(w.category, ConvergenceWarning) for w in caught)
    res.oof.to_csv(f"data/interim/oof_cmp_{name}.csv", index=False)
    return res, n_conv


def check_r0_unchanged(table):
    """R0 加了 StandardScaler 後，A 的各指標要和之前一樣（小數點三位）。"""
    ref = pd.read_csv(R0_REFERENCE)
    ref = ref[ref["eval"] == "A"].set_index("metric")
    new = table[table["eval"] == "A"].set_index("metric")
    bad = []
    for m in ["roc_auc", "pr_auc", "f1", "accuracy"]:
        for col in ["value", "std"]:
            if round(ref.at[m, col], 3) != round(new.at[m, col], 3):
                bad.append(f"{m}.{col}: 之前 {ref.at[m, col]:.4f} → 現在 {new.at[m, col]:.4f}")
    return bad


def main() -> None:
    df = load_r1_frame()
    n_test = int((df["role"] == "test").sum())
    tv = df[df["role"] == "trainval"].reset_index(drop=True)      # 明確丟掉 test 列
    assert (tv["role"] == "trainval").all() and tv["fold"].between(0, 4).all()
    print(f"[r1_lr] 丟掉 test 列 {n_test:,}；使用 trainval {len(tv):,} 列")

    results, tables, n_warn = {}, [], {}
    for name in FEATURE_SETS:
        results[name], n_warn[name] = run_one(tv, name)
        t = evaluate(results[name].oof, results[name].thresholds)
        t.insert(0, "model", name)
        tables.append(t)
        if name == "R0":
            bad = check_r0_unchanged(t)
            if bad:
                print("\n[停止] R0 加 StandardScaler 後與之前的數字不同：\n  " + "\n  ".join(bad))
                sys.exit(1)
            print("[r1_lr] 檢查通過：R0 + StandardScaler 的 A 與之前一致（小數點三位）")
    table = pd.concat(tables, ignore_index=True)
    table.to_csv("results/r1_lr_cv.csv", index=False)

    # ---- 配對差異
    pairs = [("R0d", "R0"), ("R1", "R0d")]                  # (new, old)
    fold_auc = {n: per_fold_auc(results[n].oof) for n in FEATURE_SETS}
    paired = []
    for new, old in pairs:
        d = fold_auc[new] - fold_auc[old]
        row = dict(eval="A", comparison=f"{new} - {old}", group="all_rows",
                   delta_auc=d.mean(), std=d.std(ddof=1), n_folds_better=int((d > 0).sum()))
        row.update({f"fold{f}": d[f] for f in d.index})
        paired.append(row)
        c = paired_delta_c(results[old].oof, results[new].oof)
        for _, r in c.iterrows():
            paired.append(dict(eval="C", comparison=f"{new} - {old}", group=r["group"],
                               delta_auc=r["delta_auc"], ci_low=r["ci_low"], ci_high=r["ci_high"]))
    paired = pd.DataFrame(paired)
    paired.to_csv("results/r1_lr_paired.csv", index=False)

    # ---- R1 係數（標準化後；每個 fold 一組）
    coef = pd.DataFrame([m.named_steps["lr"].coef_[0] for m in results["R1"].models],
                        columns=R1_FEATURES)
    coef_sum = pd.DataFrame({"mean": coef.mean(), "std": coef.std(ddof=1)})
    coef_sum.to_csv("results/r1_lr_coef.csv")

    # ---- 印出
    pd.set_option("display.width", 220)
    print("\n=== A 全部列（5 個 fold 平均 ± 標準差）===")
    a = table[table["eval"] == "A"]
    for m in ["roc_auc", "pr_auc", "f1", "accuracy", "acc_baseline_all_correct", "threshold"]:
        cells = [f"{a[(a.model == n) & (a.metric == m)]['value'].iloc[0]:.4f} ± "
                 f"{a[(a.model == n) & (a.metric == m)]['std'].iloc[0]:.4f}" for n in FEATURE_SETS]
        print(f"  {m:26s}" + "   ".join(f"{n}: {c}" for n, c in zip(FEATURE_SETS, cells)))

    print("\n=== B 位置分桶（合併 OOF）===")
    b = table[table["eval"] == "B"].pivot_table(index=["group", "metric"], columns="model",
                                                values="value", sort=False)[list(FEATURE_SETS)]
    b["n_rows"] = table[(table["eval"] == "B") & (table.model == "R0")].set_index(["group", "metric"])["n_rows"]
    print(b.round(4).to_string())

    print("\n=== C 固定切點（95% cluster bootstrap CI）===")
    c = table[(table["eval"] == "C") & table["metric"].isin(["roc_auc", "pr_auc"])]
    for g in c["group"].unique():
        n_rows = c[(c.group == g) & (c.model == "R0")]["n_rows"].iloc[0]
        n_stu = c[(c.group == g) & (c.model == "R0")]["n_students"].iloc[0]
        wr = table[(table["eval"] == "C") & (table.group == g) & (table.metric == "wrong_rate") & (table.model == "R0")]["value"].iloc[0]
        print(f"  {g}（序列數 {n_rows}，學生數 {n_stu}，答錯率 {wr:.3f}）")
        for m in ["roc_auc", "pr_auc"]:
            cells = []
            for n in FEATURE_SETS:
                r = c[(c.group == g) & (c.model == n) & (c.metric == m)].iloc[0]
                cells.append(f"{n}: {r['value']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]")
            print(f"    {m:8s}" + "   ".join(cells))

    print("\n=== 配對差異 ΔAUC ===")
    for _, r in paired[paired["eval"] == "A"].iterrows():
        folds = ", ".join(f"{r[f'fold{f}']:+.4f}" for f in range(5))
        print(f"  A {r['comparison']:9s} 各 fold: [{folds}]  平均 {r['delta_auc']:+.4f}  標準差 {r['std']:.4f}  "
              f"new 較好的 fold 數 {int(r['n_folds_better'])}/5")
    for _, r in paired[paired["eval"] == "C"].iterrows():
        print(f"  C {r['group']:5s} {r['comparison']:9s} ΔAUC {r['delta_auc']:+.4f}  95% CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")

    print("\n=== 收斂警告次數（ConvergenceWarning）===")
    print("  " + "  ".join(f"{n}: {k}" for n, k in n_warn.items()))

    print("\n=== R1 的 LR 係數（標準化後，5 個 fold 的平均 ± 標準差）===")
    for f_, r in coef_sum.iterrows():
        print(f"  {f_:14s} {r['mean']:+.4f} ± {r['std']:.4f}")
    print("\n[r1_lr] 已存 results/r1_lr_cv.csv、r1_lr_paired.csv、r1_lr_coef.csv")


if __name__ == "__main__":
    main()
