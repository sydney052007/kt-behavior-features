"""LR 的 ablation（加法／減法）。特徵、StandardScaler pipeline、評估都跟 run_r1_lr.py 一樣。

    python3 src/run_ablation_lr.py
每個 fold 的特徵只算一次（make_fold_features，統計量只來自該 fold 訓練列），
所有特徵組共用，所以各組使用完全相同的 fold 與驗證列。
所有 Δ 都是「該組 − 參考組」：減法 ablation 的 Δ 為負，代表拿掉那組特徵會變差。

輸出（只有彙總數字）：
    results/02_analysis/ablation_lr_cv.csv      每組的 A／B／C（長格式，多一欄 model）
    results/02_analysis/ablation_lr_paired.csv  每個 (組, 參考組) 的配對 ΔAUC（A 逐 fold、C 配對 bootstrap）
    results/02_analysis/ablation_lr_steps.csv   R0 → R0d → R1 每一步的 ΔAUC
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

sys.path.insert(0, os.path.dirname(__file__))
from cv import run_cv  # noqa: E402
from evaluate import evaluate, paired_delta_c, per_fold_auc  # noqa: E402
from features import R0_FEATURES  # noqa: E402
from features_r1 import R0D_FEATURES, R1_FEATURES, load_r1_frame, make_fold_features  # noqa: E402
from run_r1_lr import make_model  # noqa: E402  (StandardScaler + LogisticRegression(C=np.inf))

HINT = ["hint_rate", "last_hint"]
TIME = ["mean_log_time", "last_log_time"]
ATT = ["mean_log_att", "last_log_att"]
SCAFFOLD = ["scaffold_rate"]


def _minus(cols, drop):
    return [c for c in cols if c not in drop]


# 名稱 -> 特徵欄位。前三個是參考組（R0、R0d、R1）。
FEATURE_SETS = {
    "R0": R0_FEATURES,
    "R0d": R0D_FEATURES,
    "R1": R1_FEATURES,
    # 加法
    "R0+item_diff": R0_FEATURES + ["item_diff"],
    "R0+last_correct": R0_FEATURES + ["last_correct"],
    "R0d+hint": R0D_FEATURES + HINT,
    "R0d+time": R0D_FEATURES + TIME,
    "R0d+att": R0D_FEATURES + ATT,
    "R0d+scaffold": R0D_FEATURES + SCAFFOLD,
    "R0d+mean_only": R0D_FEATURES + ["hint_rate", "mean_log_time", "mean_log_att", "scaffold_rate"],
    "R0d+last_only": R0D_FEATURES + ["last_hint", "last_log_time", "last_log_att"],
    # 減法（對 R1）
    "R1-hint": _minus(R1_FEATURES, HINT),
    "R1-time": _minus(R1_FEATURES, TIME),
    "R1-att": _minus(R1_FEATURES, ATT),
    "R1-scaffold": _minus(R1_FEATURES, SCAFFOLD),
}
# (該組, 參考組)
COMPARISONS = [
    ("R0+item_diff", "R0"), ("R0+item_diff", "R0d"),
    ("R0+last_correct", "R0"), ("R0+last_correct", "R0d"),
    ("R0d+hint", "R0d"), ("R0d+time", "R0d"), ("R0d+att", "R0d"), ("R0d+scaffold", "R0d"),
    ("R0d+mean_only", "R0d"), ("R0d+last_only", "R0d"),
    ("R1-hint", "R1"), ("R1-time", "R1"), ("R1-att", "R1"), ("R1-scaffold", "R1"),
]
STEPS = [("R0d", "R0"), ("R1", "R0d"), ("R1", "R0")]     # R0 → R0d → R1，最後一列是總和
PREV_RESULT = "results/02_analysis/r1_lr_cv.csv"


def caching_builder():
    """包住 make_fold_features：同一個 fold 只算一次，所有特徵組共用。
    以驗證 fold 編號當 key（訓練列就是其他 fold，所以 key 唯一對應一組 train/val）。"""
    cache = {}

    def build(train, val):
        key = tuple(sorted(val["fold"].unique()))
        if key not in cache:
            cache[key] = make_fold_features(train, val)
        return cache[key]
    return build


def run_sets(tv, names):
    """對 names 裡的每個特徵組跑 5-fold；回傳 {name: (CVResult, 收斂警告次數)}。"""
    builder = caching_builder()
    out = {}
    for name in names:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            res = run_cv(tv, FEATURE_SETS[name], make_model, feature_builder=builder)
        out[name] = (res, sum(issubclass(w.category, ConvergenceWarning) for w in caught))
    return out


def compare(results, new, ref):
    """(該組, 參考組) 的配對差異：A 逐 fold 差 + C 配對 cluster bootstrap。回傳 list of dict。"""
    a_new, a_ref = per_fold_auc(results[new][0].oof), per_fold_auc(results[ref][0].oof)
    d = a_new - a_ref
    rows = [dict(model=new, reference=ref, eval="A", group="all_rows", delta_auc=d.mean(),
                 std=d.std(ddof=1), n_folds_better=int((d > 0).sum()),
                 **{f"fold{f}": d[f] for f in d.index})]
    c = paired_delta_c(results[ref][0].oof, results[new][0].oof)
    for _, r in c.iterrows():
        rows.append(dict(model=new, reference=ref, eval="C", group=r["group"], delta_auc=r["delta_auc"],
                         ci_low=r["ci_low"], ci_high=r["ci_high"]))
    return rows


def main() -> None:
    df = load_r1_frame()
    n_test = int((df["role"] == "test").sum())
    tv = df[df["role"] == "trainval"].reset_index(drop=True)       # 明確丟掉 test 列
    assert (tv["role"] == "trainval").all() and tv["fold"].between(0, 4).all()
    print(f"[ablation] 丟掉 test 列 {n_test:,}；使用 trainval {len(tv):,} 列；{len(FEATURE_SETS)} 組特徵")

    results = run_sets(tv, list(FEATURE_SETS))

    # 所有組的驗證列必須和 R0 完全相同（同 fold、同列、同順序）
    base = results["R0"][0].oof[["order_id", "fold"]]
    for name, (res, _) in results.items():
        assert res.oof[["order_id", "fold"]].equals(base), f"{name} 的 fold／驗證列與其他組不同"

    tables = []
    for name, (res, _) in results.items():
        t = evaluate(res.oof, res.thresholds)
        t.insert(0, "model", name)
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv("results/02_analysis/ablation_lr_cv.csv", index=False)

    # 與上一輪 run_r1_lr.py 的 R0／R0d／R1 結果核對（A 的 AUC 到小數點四位）
    prev = pd.read_csv(PREV_RESULT)
    for name in ["R0", "R0d", "R1"]:
        for m in ["roc_auc", "pr_auc"]:
            now = table[(table.model == name) & (table["eval"] == "A") & (table.metric == m)]["value"].iloc[0]
            old = prev[(prev.model == name) & (prev["eval"] == "A") & (prev.metric == m)]["value"].iloc[0]
            assert round(now, 4) == round(old, 4), f"{name} {m} 與 {PREV_RESULT} 不同：{old:.4f} → {now:.4f}"
    print("[ablation] 核對通過：R0／R0d／R1 的 A 與 run_r1_lr.py 的結果一致（小數點四位）")

    paired = pd.DataFrame([r for new, ref in COMPARISONS for r in compare(results, new, ref)])
    paired.to_csv("results/02_analysis/ablation_lr_paired.csv", index=False)
    steps = pd.DataFrame([dict(step=f"{ref} -> {new}", **{k: v for k, v in r.items() if k not in ("model", "reference")})
                          for new, ref in STEPS for r in compare(results, new, ref)])
    steps.to_csv("results/02_analysis/ablation_lr_steps.csv", index=False)

    # ---- 印出
    pd.set_option("display.width", 220)
    print("\n=== A：各組 ROC-AUC／PR-AUC（5 個 fold 平均 ± 標準差）===")
    a = table[table["eval"] == "A"].set_index(["model", "metric"])
    for name in FEATURE_SETS:
        r, p = a.loc[(name, "roc_auc")], a.loc[(name, "pr_auc")]
        print(f"  {name:16s} ROC {r['value']:.4f} ± {r['std']:.4f}   PR {p['value']:.4f} ± {p['std']:.4f}"
              f"   ({len(FEATURE_SETS[name])} 個特徵)")

    print("\n=== A：配對 ΔAUC（該組 − 參考組；5 個 fold 的差值）===")
    for _, r in paired[paired["eval"] == "A"].iterrows():
        folds = ", ".join(f"{r[f'fold{f}']:+.4f}" for f in range(5))
        print(f"  {r['model']:16s} vs {r['reference']:4s} [{folds}]  平均 {r['delta_auc']:+.4f}  "
              f"標準差 {r['std']:.4f}  較好 {int(r['n_folds_better'])}/5")

    print("\n=== C：配對 ΔAUC 與 95% cluster bootstrap CI ===")
    for _, r in paired[paired["eval"] == "C"].iterrows():
        print(f"  {r['model']:16s} vs {r['reference']:4s} {r['group']:5s} {r['delta_auc']:+.4f}  "
              f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")

    print("\n=== 總表：R0 → R0d → R1 每一步 ΔAUC ===")
    for step in steps["step"].unique():
        s = steps[steps.step == step]
        ra = s[s["eval"] == "A"].iloc[0]
        line = f"  {step:11s} A {ra['delta_auc']:+.4f} ± {ra['std']:.4f} ({int(ra['n_folds_better'])}/5 fold)"
        for _, rc in s[s["eval"] == "C"].iterrows():
            line += f" | {rc['group']} {rc['delta_auc']:+.4f} [{rc['ci_low']:+.4f}, {rc['ci_high']:+.4f}]"
        print(line)

    print("\n=== 收斂警告次數 ===")
    bad = {n: k for n, (_, k) in results.items() if k}
    print("  全部 0 次" if not bad else f"  {bad}")
    print("\n[ablation] 已存 results/02_analysis/ablation_lr_cv.csv、ablation_lr_paired.csv、ablation_lr_steps.csv")


if __name__ == "__main__":
    main()
