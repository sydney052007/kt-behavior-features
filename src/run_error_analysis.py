"""錯誤分析（規則見 CLAUDE.md「錯誤分析規則」，跑之前登記，不依結果修改）。

    python3 src/run_error_analysis.py > results/03_process/error_analysis_run.log 2>&1
只用 5-fold 驗證預測（OOF）；test 的 262 位學生完全不碰。OOF 預測只放記憶體，不存檔；
輸出只有彙總（results/02_analysis/error_*.csv，長格式，沒有 user_id、school_id）：
    error_reproduction.csv       A 的 5 fold 平均與既有結果的重現檢查
    error_cutpoints.csv          item_diff 三等分的切點
    error_segments.csv           分段表現（R0d、R1 的 AUC，R1 − R0d 的 ΔAUC 與 95% CI）
    error_confusion.csv          混淆矩陣與 slip／guess
    error_calibration.csv        10 個等寬桶的校準表
    error_calibration_ece.csv    ECE
    error_high_conf_groups.csv   高信心錯誤兩群的列數與占比
    error_high_conf_features.csv 兩群在 7 個特徵的平均 vs 全部列的平均
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))
from cv import run_cv  # noqa: E402
from evaluate import cluster_resample_indices, per_fold_metric  # noqa: E402
from features_r1 import R0D_FEATURES, R1_FEATURES, load_r1_frame, make_fold_features  # noqa: E402
from run_ablation_lr import caching_builder  # noqa: E402
from run_r1_lr import make_model as make_lr  # noqa: E402
from run_rf import make_rf  # noqa: E402

OUT = "results/02_analysis/"
N_BOOT, SEED = 2000, 42
RF_SETTING = {"R0d": (8, 5), "R1": (None, 50)}
FEATURES = {"R0d": R0D_FEATURES, "R1": R1_FEATURES}
POSITION_BINS = [("1", 1, 1), ("2-4", 2, 4), ("5-10", 5, 10), (">=11", 11, None)]
Q_LOW, Q_HIGH = 0.33, 0.67                       # item_diff 三等分的分位數
HI, LO = 0.8, 0.2                                # 高信心錯誤的門檻
N_BINS = 10
PROFILE_FEATURES = [("position", "t"), ("item_diff", "item_diff"), ("hint_rate", "hint_rate"), ("last_correct", "last_correct"),
                    ("log1p_c", "log1p_c"), ("log1p_w", "log1p_w"), ("has_history", "has_history")]
MERGE_COLS = ["item_diff", "hint_rate", "last_correct", "log1p_c", "log1p_w", "has_history"]


# ---------------------------------------------------------------- OOF（只放記憶體）
def build_runs(tv=None):
    """LR、RF × R0d、R1 的 5-fold OOF（含該 fold 訓練列選出的閾值），並把驗證列的特徵併進來。"""
    if tv is None:
        df = load_r1_frame()
        tv = df[df["role"] == "trainval"].reset_index(drop=True)                 # 明確只用 trainval
    assert (tv["role"] == "trainval").all() and tv["fold"].between(0, 4).all()
    builder = caching_builder()
    runs = {}
    for mt in ["LR", "RF"]:
        for fs in ["R0d", "R1"]:
            maker = make_lr if mt == "LR" else make_rf(*RF_SETTING[fs])
            res = run_cv(tv, FEATURES[fs], maker, feature_builder=builder)
            feats = pd.concat([builder(tv[tv.fold != k], tv[tv.fold == k])[1] for k in range(5)])
            oof = res.oof.merge(feats[["order_id"] + MERGE_COLS], on="order_id", how="left", validate="one_to_one")
            assert len(oof) == len(tv) and oof[MERGE_COLS].notna().all().all()
            runs[(mt, fs)] = dict(oof=oof, thresholds=res.thresholds)
    return runs


def reproduction_check(runs):
    """A 的 5 fold 平均（ROC-AUC、PR-AUC）要與既有結果一致（差 < 1e-9）。"""
    lr = pd.read_csv("results/02_analysis/r1_lr_cv.csv")
    rf = pd.read_csv("results/02_analysis/rf_cv.csv")
    rows = []
    for (mt, fs), r in runs.items():
        ref = (lr if mt == "LR" else rf)
        ref = ref[(ref["model"] == fs) & (ref["eval"] == "A")].set_index("metric")
        for m in ["roc_auc", "pr_auc"]:
            now = per_fold_metric(r["oof"], m).mean()
            rows.append(dict(model_type=mt, model=fs, metric=m, rerun_mean=now, existing_mean=ref.at[m, "value"],
                             abs_diff=abs(now - ref.at[m, "value"])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 分段
def item_diff_cuts(oof):
    return float(np.quantile(oof["item_diff"], Q_LOW)), float(np.quantile(oof["item_diff"], Q_HIGH))


def segments(oof, cuts):
    """回傳 [(分段類型, 分段名稱, 布林遮罩)]；每種分段類型的遮罩互斥且涵蓋全部列。"""
    q1, q2 = cuts
    out = []
    for name, lo, hi in POSITION_BINS:
        m = oof["t"] >= lo
        if hi is not None:
            m &= oof["t"] <= hi
        out.append(("position", name, m.to_numpy()))
    for sk in ["47", "311"]:
        out.append(("skill", sk, (oof["skill_id"].astype(str) == sk).to_numpy()))
    d = oof["item_diff"]
    out += [("item_diff", "low (<=q33)", (d <= q1).to_numpy()), ("item_diff", "mid (q33-q67]", ((d > q1) & (d <= q2)).to_numpy()),
            ("item_diff", "high (>q67)", (d > q2).to_numpy())]
    has = ((oof["has_history"] == 1) & (oof["hint_rate"] > 0)).to_numpy()       # t=1 的 hint_rate 是填補值，歸入 = 0
    out += [("hint_history", "hint_rate>0", has), ("hint_history", "hint_rate=0 (incl. t=1)", ~has)]
    return out


def roc(y, s):
    return np.nan if y.min() == y.max() else roc_auc_score(y, s)


def pr(y, s):
    from sklearn.metrics import average_precision_score
    return np.nan if y.min() == y.max() else average_precision_score(y, s)


def paired_delta_ci(y, s_old, s_new, users, n_boot=N_BOOT, seed=SEED):
    """ΔAUC = AUC(new) − AUC(old) 與 95% CI：以學生為單位重抽，同一批重抽同時算兩個模型。"""
    codes, _ = pd.factorize(users)
    order = np.argsort(codes, kind="stable")
    groups = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n_boot):
        idx = cluster_resample_indices(groups, rng)
        a, b = roc(y[idx], s_old[idx]), roc(y[idx], s_new[idx])
        if not (np.isnan(a) or np.isnan(b)):
            deltas.append(b - a)
    point = roc(y, s_new) - roc(y, s_old)
    if len(deltas) < n_boot // 2:
        return point, np.nan, np.nan
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return point, lo, hi


def segment_table(runs):
    rows = []
    cuts = item_diff_cuts(runs[("LR", "R1")]["oof"])
    for mt in ["LR", "RF"]:
        old, new = runs[(mt, "R0d")]["oof"], runs[(mt, "R1")]["oof"]
        assert old["order_id"].equals(new["order_id"])
        for stype, name, m in segments(new, cuts):
            y = (1 - new.loc[m, "correct"]).to_numpy()
            s_old, s_new = (1 - old.loc[m, "p_correct"]).to_numpy(), (1 - new.loc[m, "p_correct"]).to_numpy()
            n, ns, wr = int(m.sum()), int(new.loc[m, "user_id"].nunique()), float(y.mean())
            base = dict(model_type=mt, segment_type=stype, segment=name, n_rows=n, n_students=ns, wrong_rate=wr)
            single = bool(y.min() == y.max())
            note = "single class in this segment: AUC missing" if single else ""
            for fs, s in [("R0d", s_old), ("R1", s_new)]:
                rows.append(dict(base, model=fs, metric="roc_auc", value=roc(y, s), note=note))
                rows.append(dict(base, model=fs, metric="pr_auc", value=pr(y, s), note=note))
            if single:
                rows.append(dict(base, model="R1 - R0d", metric="delta_roc_auc", value=np.nan, note=note))
            else:
                d, lo, hi = paired_delta_ci(y, s_old, s_new, new.loc[m, "user_id"].to_numpy())
                rows.append(dict(base, model="R1 - R0d", metric="delta_roc_auc", value=d, ci_low=lo, ci_high=hi, note=""))
            print(f"[seg] {mt} {stype:12s} {name:24s} n={n:6d} wrong={wr:.3f}", flush=True)
    return pd.DataFrame(rows), cuts


# ---------------------------------------------------------------- 混淆矩陣
def confusion_table(runs):
    rows = []
    for (mt, fs), r in runs.items():
        o = r["oof"]
        thr = o["fold"].map(r["thresholds"]).to_numpy()
        pred_wrong = (1 - o["p_correct"].to_numpy()) >= thr                   # 閾值來自該 fold 的訓練列
        wrong = (o["correct"].to_numpy() == 0)
        c = dict(TP=int((pred_wrong & wrong).sum()), FP=int((pred_wrong & ~wrong).sum()),
                 FN=int((~pred_wrong & wrong).sum()), TN=int((~pred_wrong & ~wrong).sum()))
        n = len(o)
        assert sum(c.values()) == n
        for k, v in c.items():
            rows.append(dict(model_type=mt, model=fs, metric=k, count=v, share_of_all_rows=v / n, share_within_actual_class=np.nan))
        rows.append(dict(model_type=mt, model=fs, metric="slip (pred correct, actual wrong)", count=c["FN"], share_of_all_rows=c["FN"] / n,
                         share_within_actual_class=c["FN"] / (c["FN"] + c["TP"])))
        rows.append(dict(model_type=mt, model=fs, metric="guess (pred wrong, actual correct)", count=c["FP"], share_of_all_rows=c["FP"] / n,
                         share_within_actual_class=c["FP"] / (c["FP"] + c["TN"])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 校準
def calibration_tables(runs):
    rows, ece = [], []
    edges = np.linspace(0, 1, N_BINS + 1)
    for (mt, fs), r in runs.items():
        p, y = r["oof"]["p_correct"].to_numpy(), r["oof"]["correct"].to_numpy()
        b = np.minimum(np.digitize(p, edges[1:-1], right=False), N_BINS - 1)      # [0,0.1)…[0.9,1.0]（最後一桶含 1.0）
        total, e = len(p), 0.0
        for k in range(N_BINS):
            m = b == k
            n = int(m.sum())
            mp, ob = (float(p[m].mean()), float(y[m].mean())) if n else (np.nan, np.nan)
            if n:
                e += n / total * abs(ob - mp)
            rows.append(dict(model_type=mt, model=fs, bin=k + 1, bin_low=edges[k], bin_high=edges[k + 1], n_rows=n, mean_pred=mp, obs_correct_rate=ob))
        ece.append(dict(model_type=mt, model=fs, ece=e, n_rows=total))
    return pd.DataFrame(rows), pd.DataFrame(ece)


# ---------------------------------------------------------------- 高信心錯誤
def high_confidence_tables(runs):
    groups, feats = [], []
    for (mt, fs), r in runs.items():
        o = r["oof"]
        p, wrong = o["p_correct"], (o["correct"] == 0)
        n = len(o)
        defs = [("pred_correct>=0.8 but wrong", (p >= HI), (p >= HI) & wrong), ("pred_correct<=0.2 but correct", (p <= LO), (p <= LO) & ~wrong)]
        for gname, region, grp in defs:
            groups.append(dict(model_type=mt, model=fs, group=gname, n_rows=int(grp.sum()), share_of_all_rows=grp.sum() / n,
                               region_rows=int(region.sum()), share_of_region=grp.sum() / region.sum() if region.sum() else np.nan))
            for label, col in PROFILE_FEATURES:
                gm, am = float(o.loc[grp, col].mean()), float(o[col].mean())
                feats.append(dict(model_type=mt, model=fs, group=gname, feature=label, group_mean=gm, all_rows_mean=am, diff=gm - am,
                                  n_rows=int(grp.sum())))
    return pd.DataFrame(groups), pd.DataFrame(feats)


def main() -> None:
    t0 = time.time()
    runs = build_runs()
    n = len(runs[("LR", "R1")]["oof"])
    print(f"[error] trainval OOF {n:,} 列（只在記憶體；test 完全不碰）；{len(runs)} 組 OOF 已產生（{time.time() - t0:.0f}s）", flush=True)

    rep = reproduction_check(runs)
    rep.to_csv(OUT + "error_reproduction.csv", index=False)
    pd.set_option("display.width", 200)
    print("\n=== 重現檢查：A 的 5 fold 平均 vs 既有結果 ===")
    print(rep.to_string(index=False, float_format=lambda x: f"{x:.10f}" if x > 1e-3 else f"{x:.1e}"))
    assert rep["abs_diff"].max() < 1e-9, f"與既有結果不一致（最大差 {rep['abs_diff'].max():.2e}）"
    print(f"重現檢查通過：最大差 {rep['abs_diff'].max():.1e} < 1e-9", flush=True)

    seg, cuts = segment_table(runs)
    seg.to_csv(OUT + "error_segments.csv", index=False)
    pd.DataFrame([dict(cut="q33", quantile=Q_LOW, item_diff=cuts[0]), dict(cut="q67", quantile=Q_HIGH, item_diff=cuts[1])]).to_csv(OUT + "error_cutpoints.csv", index=False)
    conf = confusion_table(runs)
    conf.to_csv(OUT + "error_confusion.csv", index=False)
    cal, ece = calibration_tables(runs)
    cal.to_csv(OUT + "error_calibration.csv", index=False)
    ece.to_csv(OUT + "error_calibration_ece.csv", index=False)
    hg, hf = high_confidence_tables(runs)
    hg.to_csv(OUT + "error_high_conf_groups.csv", index=False)
    hf.to_csv(OUT + "error_high_conf_features.csv", index=False)

    print(f"\nitem_diff 三等分切點：q33 = {cuts[0]:.4f}，q67 = {cuts[1]:.4f}")
    for mt in ["LR", "RF"]:
        print(f"\n=== 分段表現（{mt}）===")
        w = seg[seg.model_type == mt]
        tab = w.pivot_table(index=["segment_type", "segment"], columns=["model", "metric"], values="value", sort=False)
        info = w[w.model == "R0d"].drop_duplicates(["segment_type", "segment"]).set_index(["segment_type", "segment"])[["n_rows", "wrong_rate"]]
        ci = w[w.model == "R1 - R0d"].set_index(["segment_type", "segment"])[["ci_low", "ci_high"]]
        out = pd.concat([info, tab[[("R0d", "roc_auc"), ("R1", "roc_auc"), ("R0d", "pr_auc"), ("R1", "pr_auc"), ("R1 - R0d", "delta_roc_auc")]], ci], axis=1)
        out.columns = ["n_rows", "wrong_rate", "AUC_R0d", "AUC_R1", "PR_R0d", "PR_R1", "dAUC", "ci_low", "ci_high"]
        print(out.round(4).to_string())
    print("\n=== 混淆矩陣（5 個 fold 合併；答錯為正類）===")
    print(conf.pivot_table(index=["model_type", "model"], columns="metric", values="count", sort=False).to_string())
    print(conf[conf.metric.str.startswith(("slip", "guess"))].round(4).to_string(index=False))
    print("\n=== 校準（ECE）===")
    print(ece.round(4).to_string(index=False))
    for (mt, fs) in runs:
        print(f"\n校準表 {mt} {fs}")
        print(cal[(cal.model_type == mt) & (cal.model == fs)][["bin", "bin_low", "bin_high", "n_rows", "mean_pred", "obs_correct_rate"]].round(4).to_string(index=False))
    print("\n=== 高信心錯誤 ===")
    print(hg.round(4).to_string(index=False))
    print(hf.pivot_table(index=["model_type", "model", "group"], columns="feature", values="group_mean", sort=False).round(3).to_string())
    print("（全部列的平均）", hf[(hf.model_type == "LR") & (hf.model == "R1") & (hf.group == hf.group.iloc[0])].set_index("feature")["all_rows_mean"].round(3).to_dict())
    print(f"\n[error] 完成，共 {(time.time() - t0) / 60:.1f} 分鐘；已存 {OUT}error_*.csv", flush=True)


if __name__ == "__main__":
    main()
