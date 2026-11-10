"""R1r（R1 + 4 個相對於題目平均的行為特徵）與 R0d_rel（R0d + 同樣 4 個）的 LR 與 RF 評估。

    python3 src/run_r1r.py > results/03_process/r1r_run.log 2>&1
事先決定、不依結果修改：
  * 模型：LR（StandardScaler + LogisticRegression(C=np.inf)）；RF（300 棵、sqrt、種子 42、n_jobs=-1，
    網格 max_depth ∈ {8,16,None} × min_samples_leaf ∈ {5,20,50}，共 9 組），選擇只看 5-fold 的 A ROC-AUC 平均。
  * R1、R0d 的數字沿用既有結果：LR 讀 data/interim/oof_cmp_*.csv；RF 只重跑各自最佳的那一個設定
    （R1：rf_grid_auc.csv 裡 R1 的最佳；R0d：R0d 的最佳），並確認與先前的各 fold AUC 相同。
  * 判準：以 RF 的 A 為準，R1r − R1（各自最佳設定）的平均 ΔAUC ≥ +0.002 且至少 4/5 個 fold 較好
    → 『R1 的改良版』；否則 → 『沒有明顯改善的變體』。

輸出（results/ 的子資料夾，只有彙總數字；索引見 results/INDEX.md）：
    r1r_new_features.csv  r1r_a_summary.csv  r1r_rf_grid.csv  r1r_cv.csv
    r1r_paired.csv        r1r_verdict.csv    r1r_perm_importance.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from cv import run_cv  # noqa: E402
from evaluate import evaluate, paired_delta_c, per_fold_metric  # noqa: E402
from features_r1 import R0D_FEATURES, R1_FEATURES, load_r1_frame  # noqa: E402
from features_r1r import R0D_REL_FEATURES, R1R_FEATURES, REL_FEATURES, make_fold_features_r1r  # noqa: E402
from run_r1_lr import make_model as make_lr  # noqa: E402
from run_rf import GRID, PERM_REPEATS, RF_SEED, cfg_name, make_rf  # noqa: E402

FOLD_COLS = [f"fold{k}" for k in range(5)]
VARIANTS = {"R1r": R1R_FEATURES, "R0d_rel": R0D_REL_FEATURES}
PERM_GROUPS = {
    "item_diff": ["item_diff"], "last_correct": ["last_correct"],
    "hint": ["hint_rate", "last_hint"], "att": ["mean_log_att", "last_log_att"],
    "time": ["mean_log_time", "last_log_time"], "scaffold": ["scaffold_rate"],
    "rel_hint": ["rel_hint_mean", "rel_hint_last"], "rel_att": ["rel_att_mean", "rel_att_last"],
}
MIN_DELTA, MIN_FOLDS = 0.002, 4          # 判準（寫死）


def caching_builder():
    """同一個 fold 的特徵只算一次（以驗證 fold 編號當 key）。"""
    cache = {}

    def build(train, val):
        key = tuple(sorted(val["fold"].unique()))
        if key not in cache:
            cache[key] = make_fold_features_r1r(train, val)
        return cache[key]
    return build


def pair_rows(analysis, model_type, new, ref, new_oof, ref_oof, metric="roc_auc"):
    """A 的逐 fold 差 + C（t=6、t=11）配對 cluster bootstrap（2000 次，種子 42）。"""
    d = per_fold_metric(new_oof, metric) - per_fold_metric(ref_oof, metric)
    base = dict(analysis=analysis, model_type=model_type, comparison=f"{new} - {ref}", metric=metric)
    rows = [dict(base, eval="A", group="all_rows", delta=d.mean(), std=d.std(ddof=1),
                 n_folds_better=int((d > 0).sum()), **{f"fold{f}": d[f] for f in d.index})]
    for _, r in paired_delta_c(ref_oof, new_oof, metric=metric).iterrows():
        rows.append(dict(base, eval="C", group=r["group"], delta=r["delta_auc"], ci_low=r["ci_low"], ci_high=r["ci_high"]))
    return rows


def rerun_rf(tv, builder, feats, cfg):
    return run_cv(tv, feats, make_rf(*cfg), feature_builder=builder)


def main() -> None:
    t0 = time.time()
    df = load_r1_frame()
    n_test = int((df["role"] == "test").sum())
    tv = df[df["role"] == "trainval"].reset_index(drop=True)          # 明確丟掉 test 列
    assert (tv["role"] == "trainval").all() and tv["fold"].between(0, 4).all()
    print(f"[r1r] 丟掉 test 列 {n_test:,}；使用 trainval {len(tv):,} 列", flush=True)
    builder = caching_builder()

    # ---- 1. 4 個新特徵的彙總（5 個 fold 的驗證列合併）
    vals = [builder(tv[tv.fold != k], tv[tv.fold == k])[1] for k in range(5)]
    o = pd.concat(vals)
    nf = pd.DataFrame({"mean": o[REL_FEATURES].mean(), "std": o[REL_FEATURES].std(), "min": o[REL_FEATURES].min(),
                       "max": o[REL_FEATURES].max(), "corr_correct": o[REL_FEATURES].corrwith(o["correct"])})
    nf.index.name = "feature"
    nf.reset_index().to_csv("results/01_preprocessing/r1r_new_features.csv", index=False)
    print("\n=== 1. 4 個新特徵（5 個 fold 的驗證列合併 = trainval 30,516 列）===")
    print(nf.round(4).to_string(), flush=True)

    # ---- 2. LR（R1r、R0d_rel）
    lr = {}
    for name, feats in VARIANTS.items():
        lr[name] = run_cv(tv, feats, make_lr, feature_builder=builder)
    # ---- 2. RF 網格（R1r、R0d_rel）
    grid_rows, rf_res = [], {}
    for name, feats in VARIANTS.items():
        for cfg in GRID:
            res = rerun_rf(tv, builder, feats, cfg)
            rf_res[(name, cfg)] = res._replace(models=None)
            auc = per_fold_metric(res.oof)
            grid_rows.append(dict(features=name, max_depth=str(cfg[0]), min_samples_leaf=cfg[1], roc_auc_mean=auc.mean(),
                                  roc_auc_std=auc.std(ddof=1), **{f"fold{f}": auc[f] for f in auc.index}))
            print(f"[grid] {name:8s} {cfg_name(cfg):22s} A ROC-AUC {auc.mean():.4f} ± {auc.std(ddof=1):.4f}"
                  f"   (累計 {time.time() - t0:.0f}s)", flush=True)
    grid = pd.DataFrame(grid_rows)
    grid["is_best"] = grid.groupby("features")["roc_auc_mean"].transform("max") == grid["roc_auc_mean"]
    grid.to_csv("results/03_process/r1r_rf_grid.csv", index=False)
    best = {n: max(GRID, key=lambda c: per_fold_metric(rf_res[(n, c)].oof).mean()) for n in VARIANTS}

    # ---- 參考組 R1、R0d（沿用既有結果；RF 只重跑各自最佳設定並確認可重現）
    # max_depth 欄有字串 "None"，不能讓 pandas 把它當缺失值
    old_grid = pd.read_csv("results/02_analysis/rf_grid_auc.csv", dtype={"max_depth": str}, keep_default_na=False)
    ref_cfg = {}
    for n in ["R1", "R0d"]:
        r = old_grid[(old_grid.features == n) & old_grid.is_best].iloc[0]
        ref_cfg[n] = (None if r.max_depth == "None" else int(r.max_depth), int(r.min_samples_leaf))
    rf_ref = {}
    for n, feats in [("R1", R1_FEATURES), ("R0d", R0D_FEATURES)]:
        res = rerun_rf(tv, builder, feats, ref_cfg[n])
        r = old_grid[(old_grid.features == n) & old_grid.is_best].iloc[0]
        now = per_fold_metric(res.oof)
        diff = np.abs(now.to_numpy() - r[FOLD_COLS].to_numpy(float)).max()
        assert diff < 1e-9, f"{n} RF 重跑與先前結果不同（最大差 {diff:.2e}）"
        rf_ref[n] = res
        print(f"[reproduce] RF {n} {cfg_name(ref_cfg[n])}：各 fold AUC 與先前結果最大差 {diff:.1e}  OK", flush=True)
    lr_ref = {n: pd.read_csv(f"data/interim/oof_cmp_{n}.csv") for n in ["R1", "R0d"]}
    prev = pd.read_csv("results/02_analysis/r1_lr_cv.csv")
    for n in ["R1", "R0d"]:
        old = prev[(prev.model == n) & (prev["eval"] == "A") & (prev.metric == "roc_auc")]["value"].iloc[0]
        assert abs(per_fold_metric(lr_ref[n]).mean() - old) < 1e-9
    print("[reproduce] LR R1／R0d 的 OOF 與 results/02_analysis/r1_lr_cv.csv 一致  OK", flush=True)

    # ---- 2. A 的彙總 + A／B／C 全部報告
    models = {("LR", n): (lr[n], VARIANTS[n], "StandardScaler+LR") for n in VARIANTS}
    models.update({("RF", n): (rf_res[(n, best[n])], VARIANTS[n], cfg_name(best[n])) for n in VARIANTS})
    summary, tables = [], []
    for (mt, n), (res, feats, cfg) in models.items():
        a, p = per_fold_metric(res.oof), per_fold_metric(res.oof, "pr_auc")
        summary.append(dict(model_type=mt, features=n, n_features=len(feats), setting=cfg, roc_auc_mean=a.mean(),
                            roc_auc_std=a.std(ddof=1), pr_auc_mean=p.mean(), pr_auc_std=p.std(ddof=1)))
        t = evaluate(res.oof, res.thresholds)
        t.insert(0, "model_type", mt); t.insert(1, "features", n); t.insert(2, "setting", cfg)
        tables.append(t)
    for mt, n, oof, cfg in [("LR", "R1", lr_ref["R1"], "StandardScaler+LR"), ("LR", "R0d", lr_ref["R0d"], "StandardScaler+LR"),
                            ("RF", "R1", rf_ref["R1"].oof, cfg_name(ref_cfg["R1"])), ("RF", "R0d", rf_ref["R0d"].oof, cfg_name(ref_cfg["R0d"]))]:
        oof = oof.oof if hasattr(oof, "oof") else oof
        a, p = per_fold_metric(oof), per_fold_metric(oof, "pr_auc")
        summary.append(dict(model_type=mt, features=n + "（參考）", n_features=len(R1_FEATURES if n == "R1" else R0D_FEATURES),
                            setting=cfg, roc_auc_mean=a.mean(), roc_auc_std=a.std(ddof=1), pr_auc_mean=p.mean(), pr_auc_std=p.std(ddof=1)))
    summary = pd.DataFrame(summary)
    summary.to_csv("results/02_analysis/r1r_a_summary.csv", index=False)
    pd.concat(tables, ignore_index=True).to_csv("results/02_analysis/r1r_cv.csv", index=False)
    print("\n=== 2. A ROC-AUC／PR-AUC（5 個 fold 平均 ± 標準差；RF 為各自最佳設定）===")
    for _, r in summary.iterrows():
        print(f"  {r.model_type} {r.features:10s} ROC {r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f}   "
              f"PR {r.pr_auc_mean:.4f} ± {r.pr_auc_std:.4f}   ({r.n_features} 個特徵；{r.setting})")
    print("\n  RF 完整網格（A ROC-AUC 平均 ± 標準差，* = 最佳）")
    for n in VARIANTS:
        print(f"    {n}")
        for _, r in grid[grid.features == n].iterrows():
            print(f"      depth={r.max_depth}, leaf={r.min_samples_leaf:<3d} {r.roc_auc_mean:.4f} ± {r.roc_auc_std:.4f}{'  *' if r.is_best else ''}")
    a_tab = pd.concat(tables)
    print("\n  F1／Accuracy／B／C 全部寫在 results/02_analysis/r1r_cv.csv（A 的 F1、Accuracy：）")
    for (mt, n), _ in models.items():
        t = a_tab[(a_tab.model_type == mt) & (a_tab.features == n) & (a_tab["eval"] == "A")].set_index("metric")
        print(f"    {mt} {n:8s} F1 {t.at['f1', 'value']:.4f} ± {t.at['f1', 'std']:.4f}   Accuracy {t.at['accuracy', 'value']:.4f} ± {t.at['accuracy', 'std']:.4f}")

    # ---- 3. 配對差異
    paired = []
    rf_best_oof = {n: rf_res[(n, best[n])].oof for n in VARIANTS}
    paired += pair_rows("own_best", "LR", "R1r", "R1", lr["R1r"].oof, lr_ref["R1"])
    paired += pair_rows("own_best", "RF", "R1r", "R1", rf_best_oof["R1r"], rf_ref["R1"].oof)
    paired += pair_rows("own_best", "LR", "R0d_rel", "R0d", lr["R0d_rel"].oof, lr_ref["R0d"])
    paired += pair_rows("own_best", "RF", "R0d_rel", "R0d", rf_best_oof["R0d_rel"], rf_ref["R0d"].oof)
    # 補充（不當判準）：RF 兩邊都用 R1 的最佳設定
    paired += pair_rows("same_cfg_R1best", "RF", "R1r", "R1", rf_res[("R1r", ref_cfg["R1"])].oof, rf_ref["R1"].oof)
    # ---- 4. 之前的待辦：R1 − R0d 的 ΔPR-AUC
    paired += pair_rows("todo_pr", "LR", "R1", "R0d", lr_ref["R1"], lr_ref["R0d"], "pr_auc")
    paired += pair_rows("todo_pr", "RF", "R1", "R0d", rf_ref["R1"].oof, rf_ref["R0d"].oof, "pr_auc")
    paired = pd.DataFrame(paired)
    paired.to_csv("results/02_analysis/r1r_paired.csv", index=False)
    print("\n=== 3./4. 配對差異（Δ = 左 − 右）===")
    for _, r in paired[paired["eval"] == "A"].iterrows():
        print(f"  [{r.analysis}] {r.model_type} A {r.metric:7s} {r.comparison:14s} [" + ", ".join(f"{r[f]:+.4f}" for f in FOLD_COLS) +
              f"]  平均 {r.delta:+.4f}  標準差 {r['std']:.4f}  較好 {int(r.n_folds_better)}/5")
    for _, r in paired[paired["eval"] == "C"].iterrows():
        print(f"  [{r.analysis}] {r.model_type} C {r.metric:7s} {r.comparison:14s} {r.group:5s} {r.delta:+.4f}  [{r.ci_low:+.4f}, {r.ci_high:+.4f}]")

    # ---- 判準（寫死）
    v = paired[(paired.analysis == "own_best") & (paired.model_type == "RF") & (paired.comparison == "R1r - R1") & (paired["eval"] == "A")].iloc[0]
    ok = (v.delta >= MIN_DELTA) and (v.n_folds_better >= MIN_FOLDS)
    verdict = "R1 的改良版" if ok else "沒有明顯改善的變體"
    pd.DataFrame([dict(rule=f"RF 的 A：R1r − R1 平均 ΔAUC ≥ {MIN_DELTA:+.3f} 且至少 {MIN_FOLDS}/5 fold 較好", delta_auc=v.delta,
                       n_folds_better=int(v.n_folds_better), verdict=verdict)]).to_csv("results/02_analysis/r1r_verdict.csv", index=False)
    print(f"\n=== 判準：RF 的 A，R1r − R1 平均 ΔAUC = {v.delta:+.4f}（門檻 ≥ {MIN_DELTA:+.3f}），較好 {int(v.n_folds_better)}/5（門檻 ≥ {MIN_FOLDS}）"
          f" → 【{verdict}】", flush=True)

    # ---- 5. R1r 在 RF 最佳設定下的 permutation importance
    res = rerun_rf(tv, builder, R1R_FEATURES, best["R1r"])
    assert np.abs(res.oof["p_correct"].to_numpy() - rf_best_oof["R1r"]["p_correct"].to_numpy()).max() < 1e-12
    rows = {g: [] for g in PERM_GROUPS}
    for f in range(5):
        train, val = tv[tv.fold != f], tv[tv.fold == f]
        X, y = builder(train, val)[1][R1R_FEATURES].reset_index(drop=True), val["correct"].to_numpy()
        model = res.models[f]
        col = list(model.classes_).index(1)
        base = roc_auc_score(y, model.predict_proba(X)[:, col])
        rng = np.random.default_rng(RF_SEED + f)
        for g, cols in PERM_GROUPS.items():
            drops = []
            for _ in range(PERM_REPEATS):
                Xp = X.copy()
                Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
                drops.append(base - roc_auc_score(y, model.predict_proba(Xp)[:, col]))
            rows[g].append(float(np.mean(drops)))
    perm = pd.DataFrame(rows).T
    perm.columns = FOLD_COLS
    perm.insert(0, "mean", perm[FOLD_COLS].mean(axis=1)); perm.insert(1, "std", perm[FOLD_COLS].std(axis=1, ddof=1))
    perm.index.name = "group"
    perm.reset_index().to_csv("results/02_analysis/r1r_perm_importance.csv", index=False)
    print(f"\n=== 5. R1r 的 permutation importance（RF {cfg_name(best['R1r'])}；AUC 下降量，5 個 fold 平均 ± 標準差；診斷用）===")
    for g, r in perm.iterrows():
        print(f"  {g:13s} {r['mean']:+.4f} ± {r['std']:.4f}")
    print(f"\n[r1r] 完成，共 {(time.time() - t0) / 60:.1f} 分鐘；已存 results/0*/r1r_*.csv", flush=True)


if __name__ == "__main__":
    main()
