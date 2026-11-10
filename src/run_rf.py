"""RandomForest：R0／R0d／R1 × 9 組超參數（W3-B）。特徵不變，只換模型。

    python3 src/run_rf.py > results/03_process/rf_run.log 2>&1
模型：RandomForestClassifier(n_estimators=300, max_features="sqrt", random_state=42, n_jobs=-1)，不縮放。
網格（事先固定）：max_depth ∈ {8, 16, None} × min_samples_leaf ∈ {5, 20, 50}，共 9 組，每組特徵都跑全部。
選擇準則只看 5-fold 的 A ROC-AUC 平均；之後不因結果調整任何設定。

輸出（只有彙總數字，results/ 的子資料夾；索引見 results/INDEX.md）：
    rf_grid_auc.csv          3 組特徵 × 9 組超參數的 A ROC-AUC（含各 fold，標出最佳）
    rf_cv.csv                各組最佳設定的 A／B／C（長格式）
    rf_paired.csv            配對差異（各自最佳設定；以及『用 R0d 最佳設定套到全部』的穩健性檢查）
    rf_vs_lr.csv             RF 與 LR 並排
    rf_perm_importance.csv   R1 最佳設定的 permutation importance（診斷，不用來選擇）
"""
import itertools
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from cv import run_cv  # noqa: E402
from evaluate import evaluate, paired_delta_c, per_fold_auc  # noqa: E402
from features import R0_FEATURES  # noqa: E402
from features_r1 import R0D_FEATURES, R1_FEATURES, load_r1_frame  # noqa: E402
from run_ablation_lr import caching_builder  # noqa: E402  (同一個 fold 的特徵只算一次)

FEATURE_SETS = {"R0": R0_FEATURES, "R0d": R0D_FEATURES, "R1": R1_FEATURES}
GRID = list(itertools.product([8, 16, None], [5, 20, 50]))       # (max_depth, min_samples_leaf)
RF_SEED = 42
PERM_REPEATS = 5
# permutation importance 的特徵群（同一群的欄位一起打亂）
PERM_GROUPS = {
    "item_diff": ["item_diff"],
    "last_correct": ["last_correct"],
    "hint": ["hint_rate", "last_hint"],
    "time": ["mean_log_time", "last_log_time"],
    "att": ["mean_log_att", "last_log_att"],
    "scaffold": ["scaffold_rate"],
}
LR_CV, LR_PAIRED = "results/02_analysis/r1_lr_cv.csv", "results/02_analysis/r1_lr_paired.csv"
FOLD_COLS = [f"fold{k}" for k in range(5)]


def make_rf(max_depth, min_samples_leaf):
    """回傳『沒有參數的建模函式』給 run_cv（每個 fold 都要新模型）。"""
    def make():
        return RandomForestClassifier(n_estimators=300, max_features="sqrt", max_depth=max_depth,
                                      min_samples_leaf=min_samples_leaf, random_state=RF_SEED, n_jobs=-1)
    return make


def cfg_name(cfg):
    return f"depth={cfg[0]}, leaf={cfg[1]}"


def run_grid(tv, builder):
    """3 組特徵 × 9 組超參數；回傳 {(特徵組, cfg): CVResult（不留模型，省記憶體）}。"""
    results, t0 = {}, time.time()
    for name, feats in FEATURE_SETS.items():
        for cfg in GRID:
            res = run_cv(tv, feats, make_rf(*cfg), feature_builder=builder)
            results[(name, cfg)] = res._replace(models=None)
            auc = per_fold_auc(res.oof)
            print(f"[grid] {name:4s} {cfg_name(cfg):22s} A ROC-AUC {auc.mean():.4f} ± {auc.std(ddof=1):.4f}"
                  f"   (累計 {time.time() - t0:.0f}s)", flush=True)
    return results


def grid_table(results):
    rows = []
    for (name, cfg), res in results.items():
        auc = per_fold_auc(res.oof)
        rows.append(dict(features=name, max_depth=str(cfg[0]), min_samples_leaf=cfg[1], roc_auc_mean=auc.mean(),
                         roc_auc_std=auc.std(ddof=1), **{f"fold{f}": auc[f] for f in auc.index}))
    t = pd.DataFrame(rows)
    # 每組特徵的最佳設定 = A ROC-AUC 平均最高者
    t["is_best"] = t.groupby("features")["roc_auc_mean"].transform("max") == t["roc_auc_mean"]
    return t


def best_cfgs(results):
    out = {}
    for name in FEATURE_SETS:
        scores = {cfg: per_fold_auc(results[(name, cfg)].oof).mean() for cfg in GRID}
        out[name] = max(GRID, key=lambda c: scores[c])        # 並列時取網格中較前面的
    return out


def paired_rows(analysis, new, ref, res_new, res_ref):
    """A 的逐 fold 差 + C（t=6、t=11）配對 cluster bootstrap。"""
    d = per_fold_auc(res_new.oof) - per_fold_auc(res_ref.oof)
    rows = [dict(analysis=analysis, comparison=f"{new} - {ref}", eval="A", group="all_rows", delta_auc=d.mean(),
                 std=d.std(ddof=1), n_folds_better=int((d > 0).sum()), **{f"fold{f}": d[f] for f in d.index})]
    for _, r in paired_delta_c(res_ref.oof, res_new.oof).iterrows():
        rows.append(dict(analysis=analysis, comparison=f"{new} - {ref}", eval="C", group=r["group"],
                         delta_auc=r["delta_auc"], ci_low=r["ci_low"], ci_high=r["ci_high"]))
    return rows


def permutation_importance(tv, builder, cfg):
    """R1 最佳設定：每個 fold 的驗證列，一群特徵一起打亂（列順序同步）後 AUC 的下降量。"""
    res = run_cv(tv, R1_FEATURES, make_rf(*cfg), feature_builder=builder)    # 重跑一次取得各 fold 的模型
    rows = {g: [] for g in PERM_GROUPS}
    for f in range(5):
        train, val = tv[tv["fold"] != f], tv[tv["fold"] == f]
        _, val_x, _ = builder(train, val)                      # 快取：與 CV 時同一份驗證特徵
        X, y = val_x[R1_FEATURES].reset_index(drop=True), val["correct"].to_numpy()
        model = res.models[f]
        col = list(model.classes_).index(1)
        base = roc_auc_score(y, model.predict_proba(X)[:, col])
        rng = np.random.default_rng(RF_SEED + f)
        for g, cols in PERM_GROUPS.items():
            drops = []
            for _ in range(PERM_REPEATS):
                Xp = X.copy()
                Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]       # 同一群欄位用同一個列排列
                drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, col]))
            rows[g].append(float(np.mean(drops)))
    out = pd.DataFrame(rows).T
    out.columns = FOLD_COLS
    out.insert(0, "mean", out[FOLD_COLS].mean(axis=1))
    out.insert(1, "std", out[FOLD_COLS].std(axis=1, ddof=1))
    out.index.name = "group"
    return out.reset_index(), res


def main() -> None:
    t0 = time.time()
    df = load_r1_frame()
    n_test = int((df["role"] == "test").sum())
    tv = df[df["role"] == "trainval"].reset_index(drop=True)          # 明確丟掉 test 列
    assert (tv["role"] == "trainval").all() and tv["fold"].between(0, 4).all()
    assert len(GRID) == 9 and len(set(GRID)) == 9
    print(f"[rf] 丟掉 test 列 {n_test:,}；使用 trainval {len(tv):,} 列；網格 {len(GRID)} 組 × {len(FEATURE_SETS)} 組特徵", flush=True)

    builder = caching_builder()
    results = run_grid(tv, builder)

    # ---- 2. 完整網格 + 最佳設定
    grid = grid_table(results)
    grid.to_csv("results/03_process/rf_grid_auc.csv", index=False)
    best = best_cfgs(results)
    print("\n=== 3 組特徵 × 9 組超參數：A ROC-AUC 平均（* = 該組最佳）===")
    pivot = grid.assign(cfg=lambda d: "depth=" + d.max_depth + ", leaf=" + d.min_samples_leaf.astype(str))
    for name in FEATURE_SETS:
        print(f"  {name}")
        for _, r in pivot[pivot.features == name].iterrows():
            print(f"    {r['cfg']:22s} {r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f}{'  *' if r.is_best else ''}")
    print("  最佳設定：" + "；".join(f"{n}: {cfg_name(c)}" for n, c in best.items()), flush=True)

    # ---- 3. 各組最佳設定的 A／B／C
    tables = []
    for name, cfg in best.items():
        t = evaluate(results[(name, cfg)].oof, results[(name, cfg)].thresholds)
        t.insert(0, "model", name)
        t.insert(1, "max_depth", str(cfg[0]))
        t.insert(2, "min_samples_leaf", cfg[1])
        tables.append(t)
    table = pd.concat(tables, ignore_index=True)
    table.to_csv("results/02_analysis/rf_cv.csv", index=False)
    pd.set_option("display.width", 220)
    a = table[table["eval"] == "A"]
    print("\n=== A 全部列（各組最佳設定；5 個 fold 平均 ± 標準差）===")
    for m in ["roc_auc", "pr_auc", "f1", "accuracy", "acc_baseline_all_correct", "threshold"]:
        print(f"  {m:26s}" + "   ".join(
            f"{n}: {a[(a.model == n) & (a.metric == m)]['value'].iloc[0]:.4f} ± {a[(a.model == n) & (a.metric == m)]['std'].iloc[0]:.4f}"
            for n in FEATURE_SETS))
    print("\n=== B 位置分桶（合併 OOF）===")
    b = table[table["eval"] == "B"].pivot_table(index=["group", "metric"], columns="model", values="value", sort=False)[list(FEATURE_SETS)]
    b["n_rows"] = table[(table["eval"] == "B") & (table.model == "R0")].set_index(["group", "metric"])["n_rows"]
    print(b.round(4).to_string())
    print("\n=== C 固定切點（95% cluster bootstrap CI）===")
    c = table[(table["eval"] == "C") & table["metric"].isin(["roc_auc", "pr_auc"])]
    for g in c["group"].unique():
        print(f"  {g}")
        for m in ["roc_auc", "pr_auc"]:
            print(f"    {m:8s}" + "   ".join(
                f"{n}: {r['value']:.4f} [{r['ci_low']:.4f}, {r['ci_high']:.4f}]"
                for n in FEATURE_SETS for r in [c[(c.group == g) & (c.model == n) & (c.metric == m)].iloc[0]]))

    # ---- 4. 配對差異（各自最佳設定）+ 5. 穩健性（R0d 的最佳設定套到全部）
    paired = []
    for new, ref in [("R0d", "R0"), ("R1", "R0d")]:
        paired += paired_rows("own_best", new, ref, results[(new, best[new])], results[(ref, best[ref])])
    cfg_d = best["R0d"]
    for new, ref in [("R0d", "R0"), ("R1", "R0d")]:
        paired += paired_rows("R0d_best_cfg_for_all", new, ref, results[(new, cfg_d)], results[(ref, cfg_d)])
    paired = pd.DataFrame(paired)
    paired.to_csv("results/02_analysis/rf_paired.csv", index=False)
    print("\n=== 配對 ΔAUC ===")
    for _, r in paired[paired["eval"] == "A"].iterrows():
        print(f"  [{r['analysis']}] A {r['comparison']:9s} 各 fold [" + ", ".join(f"{r[f]:+.4f}" for f in FOLD_COLS) +
              f"]  平均 {r['delta_auc']:+.4f}  標準差 {r['std']:.4f}  較好 {int(r['n_folds_better'])}/5")
    for _, r in paired[paired["eval"] == "C"].iterrows():
        print(f"  [{r['analysis']}] C {r['group']:5s} {r['comparison']:9s} {r['delta_auc']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]")
    print(f"  （穩健性所用的設定 = R0d 的最佳設定：{cfg_name(cfg_d)}）", flush=True)

    # ---- 6. 與 LR 並排
    lr = pd.read_csv(LR_CV)
    lr = lr[lr["eval"] == "A"].set_index(["model", "metric"])
    lrp = pd.read_csv(LR_PAIRED)
    lrp = lrp[(lrp["eval"] == "A") & (lrp["comparison"] == "R1 - R0d")].iloc[0]
    side = []
    for name in ["R0d", "R1"]:
        rf_a = a[(a.model == name) & (a.metric == "roc_auc")].iloc[0]
        side.append(dict(item=f"{name} A ROC-AUC", lr_mean=lr.loc[(name, "roc_auc"), "value"], lr_std=lr.loc[(name, "roc_auc"), "std"],
                         rf_mean=rf_a["value"], rf_std=rf_a["std"], rf_setting=cfg_name(best[name])))
    rf_d = paired[(paired.analysis == "own_best") & (paired.comparison == "R1 - R0d") & (paired["eval"] == "A")].iloc[0]
    rf_d2 = paired[(paired.analysis == "R0d_best_cfg_for_all") & (paired.comparison == "R1 - R0d") & (paired["eval"] == "A")].iloc[0]
    side.append(dict(item="R1 - R0d A ΔAUC", lr_mean=lrp["delta_auc"], lr_std=lrp["std"], rf_mean=rf_d["delta_auc"],
                     rf_std=rf_d["std"], rf_setting="各自最佳設定",
                     lr_folds_better=int(lrp["n_folds_better"]), rf_folds_better=int(rf_d["n_folds_better"]),
                     **{f"lr_{f}": lrp[f] for f in FOLD_COLS}, **{f"rf_{f}": rf_d[f] for f in FOLD_COLS}))
    side.append(dict(item="R1 - R0d A ΔAUC", lr_mean=lrp["delta_auc"], lr_std=lrp["std"], rf_mean=rf_d2["delta_auc"],
                     rf_std=rf_d2["std"], rf_setting=f"R0d 最佳設定（{cfg_name(cfg_d)}）",
                     lr_folds_better=int(lrp["n_folds_better"]), rf_folds_better=int(rf_d2["n_folds_better"]),
                     **{f"lr_{f}": lrp[f] for f in FOLD_COLS}, **{f"rf_{f}": rf_d2[f] for f in FOLD_COLS}))
    side = pd.DataFrame(side)
    side.to_csv("results/02_analysis/rf_vs_lr.csv", index=False)
    print("\n=== RF 與 LR 並排（A）===")
    for _, r in side.iterrows():
        print(f"  {r['item']:18s} LR {r.lr_mean:+.4f} ± {r.lr_std:.4f} | RF {r.rf_mean:+.4f} ± {r.rf_std:.4f}  ({r.rf_setting})")

    # ---- 7. permutation importance（診斷）
    perm, res_best = permutation_importance(tv, builder, best["R1"])
    # 重跑的 R1 最佳設定必須與網格中存的結果完全相同（可重現）
    # （n_jobs=-1 的多執行緒加總有約 1e-16 的浮點誤差，所以比到 1e-12，不要求逐位元組相同）
    assert np.abs(res_best.oof["p_correct"].to_numpy() - results[("R1", best["R1"])].oof["p_correct"].to_numpy()).max() < 1e-12, \
        "R1 最佳設定重跑結果與網格不同"
    perm.to_csv("results/02_analysis/rf_perm_importance.csv", index=False)
    print(f"\n=== R1 最佳設定（{cfg_name(best['R1'])}）的 permutation importance：AUC 下降量（5 個 fold 平均 ± 標準差；診斷用）===")
    for _, r in perm.iterrows():
        print(f"  {r['group']:13s} {r['mean']:+.4f} ± {r['std']:.4f}")

    print(f"\n[rf] 完成，共 {(time.time() - t0) / 60:.1f} 分鐘；已存 results/0*/rf_*.csv", flush=True)


if __name__ == "__main__":
    main()
