"""報告用寬表：從 results/02_analysis 的長格式檔案整理成寬格式，輸出到 results/05_report_tables/。

    python3 src/make_report_tables.py
原則：
  * 所有數字都直接取自長格式檔案，不重算任何統計量（只做篩選、並排、改欄名）。
  * 列 = 比較對象（模型 × 特徵組…），欄 = 指標；每個指標有獨立的數值欄（mean、sd，或 value、ci_low、ci_high）。
  * 另外每個指標有一個『格式化字串欄』（結尾 _str，4 位小數，CI 寫成 "0.7547 [0.7280, 0.7806]"）。
    數值欄一律是數字，字串欄只放給人看的格式，兩者分開。
  * A、B、C 各自一張表，不合併（c_ablation 依規格同時有 A 與 C 的欄，沒有 B）。
  * 每張表的來源檔案記在 TABLES[表名]["sources"]，tests/test_report_tables.py 會檢查表內每個數字都能在來源檔案找到。
"""
import os

import numpy as np
import pandas as pd

R2 = "results/02_analysis/"
OUT = "results/05_report_tables/"
FEATS = ["R0", "R0d", "R1"]
FOLDS = [f"fold{k}" for k in range(5)]
# 文字欄位（識別用，不是數值）；其餘欄位若不是 *_str 就必須是數值（或布林）
ID_COLS = {"model_type", "features", "setting", "analysis", "comparison", "group", "reference", "split", "bucket",
           "max_depth", "setting_R0", "setting_R0d", "setting_R1", "segment_type", "segment", "model", "note"}


def rd(name):
    """讀長格式檔案；max_depth 有字串 "None"，不能讓 pandas 當缺失值。"""
    return pd.read_csv(R2 + name, dtype={"max_depth": str}, keep_default_na=False, na_values=[""])


def one(df, **kw):
    sel = df
    for k, v in kw.items():
        sel = sel[sel[k] == v]
    assert len(sel) == 1, (kw, len(sel))
    return sel.iloc[0]


# ---- 格式化字串（4 位小數）
def f4(x): return f"{x:.4f}"
def pm(m, s): return f"{m:.4f} ± {s:.4f}"
def ci(v, lo, hi): return f"{v:.4f} [{lo:.4f}, {hi:.4f}]"
def spm(m, s): return f"{m:+.4f} ± {s:.4f}"
def sci(v, lo, hi): return f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def rf_setting(rf, model):
    r = rf[rf["model"] == model].iloc[0]
    return f"depth={r['max_depth']}, leaf={r['min_samples_leaf']}"


# ====================================================================== (a) 主結果表：A 與 C
def table_a_main():
    lr, rf = rd("r1_lr_cv.csv"), rd("rf_cv.csv")
    A, C = [], []
    for mt, df in [("LR", lr), ("RF", rf)]:
        for fs in FEATS:
            setting = "StandardScaler+LR" if mt == "LR" else rf_setting(rf, fs)
            a = df[(df["model"] == fs) & (df["eval"] == "A")].set_index("metric")
            row = dict(model_type=mt, features=fs, setting=setting, n_rows=int(a.at["roc_auc", "n_rows"]),
                       n_students=int(a.at["roc_auc", "n_students"]))
            for m in ["roc_auc", "pr_auc", "f1", "accuracy"]:
                row[f"{m}_mean"], row[f"{m}_sd"] = a.at[m, "value"], a.at[m, "std"]
            row["acc_baseline_all_correct"] = a.at["acc_baseline_all_correct", "value"]
            for m in ["roc_auc", "pr_auc", "f1", "accuracy"]:
                row[f"{m}_str"] = pm(row[f"{m}_mean"], row[f"{m}_sd"])
            A.append(row)
            row = dict(model_type=mt, features=fs, setting=setting)
            for g, tag in [("t=6", "t6"), ("t=11", "t11")]:
                c = df[(df["model"] == fs) & (df["eval"] == "C") & (df["group"] == g)].set_index("metric")
                row[f"{tag}_n_rows"], row[f"{tag}_n_students"] = int(c.at["roc_auc", "n_rows"]), int(c.at["roc_auc", "n_students"])
                row[f"{tag}_wrong_rate"] = c.at["wrong_rate", "value"]
                for m in ["roc_auc", "pr_auc"]:
                    row[f"{tag}_{m}"], row[f"{tag}_{m}_ci_low"], row[f"{tag}_{m}_ci_high"] = \
                        c.at[m, "value"], c.at[m, "ci_low"], c.at[m, "ci_high"]
                for m in ["roc_auc", "pr_auc"]:
                    row[f"{tag}_{m}_str"] = ci(row[f"{tag}_{m}"], row[f"{tag}_{m}_ci_low"], row[f"{tag}_{m}_ci_high"])
            C.append(row)
    return pd.DataFrame(A), pd.DataFrame(C)


# ====================================================================== (b) 配對差異：A 與 C
def table_b_paired():
    lr, rf = rd("r1_lr_paired.csv"), rd("rf_paired.csv")
    A, C = [], []
    specs = [("LR", "main", lr, "comparison", None)]
    for an in ["own_best", "R0d_best_cfg_for_all"]:
        specs.append(("RF", an, rf[rf["analysis"] == an], "comparison", an))
    for mt, an, df, _, _ in specs:
        for comp in ["R0d - R0", "R1 - R0d"]:
            a = one(df, comparison=comp, eval="A")
            row = dict(model_type=mt, analysis=an, comparison=comp, delta_mean=a["delta_auc"], delta_sd=a["std"],
                       n_folds_better=int(a["n_folds_better"]), **{k: a[k] for k in FOLDS})
            row["delta_str"] = spm(row["delta_mean"], row["delta_sd"])
            A.append(row)
            row = dict(model_type=mt, analysis=an, comparison=comp)
            for g, tag in [("t=6", "t6"), ("t=11", "t11")]:
                c = one(df, comparison=comp, eval="C", group=g)
                row[f"{tag}_delta"], row[f"{tag}_ci_low"], row[f"{tag}_ci_high"] = c["delta_auc"], c["ci_low"], c["ci_high"]
                row[f"{tag}_delta_str"] = sci(c["delta_auc"], c["ci_low"], c["ci_high"])
            C.append(row)
    return pd.DataFrame(A), pd.DataFrame(C)


# ====================================================================== (c) ablation
def table_c_ablation():
    cv, pr = rd("ablation_lr_cv.csv"), rd("ablation_lr_paired.csv")
    rows = []
    for _, a in pr[pr["eval"] == "A"].iterrows():
        g, ref = a["model"], a["reference"]
        s = cv[(cv["model"] == g) & (cv["eval"] == "A")].set_index("metric")
        row = dict(group=g, reference=ref,
                   roc_auc_mean=s.at["roc_auc", "value"], roc_auc_sd=s.at["roc_auc", "std"],
                   pr_auc_mean=s.at["pr_auc", "value"], pr_auc_sd=s.at["pr_auc", "std"],
                   delta_auc_mean=a["delta_auc"], delta_auc_sd=a["std"], n_folds_better=int(a["n_folds_better"]),
                   **{k: a[k] for k in FOLDS})
        for tag, gg in [("t6", "t=6"), ("t11", "t=11")]:
            c = one(pr, model=g, reference=ref, eval="C", group=gg)
            row[f"{tag}_delta"], row[f"{tag}_ci_low"], row[f"{tag}_ci_high"] = c["delta_auc"], c["ci_low"], c["ci_high"]
        row["roc_auc_str"], row["pr_auc_str"] = pm(row["roc_auc_mean"], row["roc_auc_sd"]), pm(row["pr_auc_mean"], row["pr_auc_sd"])
        row["delta_auc_str"] = spm(row["delta_auc_mean"], row["delta_auc_sd"])
        row["t6_delta_str"] = sci(row["t6_delta"], row["t6_ci_low"], row["t6_ci_high"])
        row["t11_delta_str"] = sci(row["t11_delta"], row["t11_ci_low"], row["t11_ci_high"])
        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================== (d) 位置分桶 B
def table_d_position():
    out = {}
    for mt, name in [("LR", "r1_lr_cv.csv"), ("RF", "rf_cv.csv")]:
        df = rd(name)
        df = df[df["eval"] == "B"]
        rows = []
        for g in ["t=1", "t=2-4", "t=5-10", "t=11+"]:
            row = dict(bucket=g)
            n = {one(df, model=fs, group=g, metric="roc_auc")["n_rows"] for fs in FEATS}
            w = {one(df, model=fs, group=g, metric="wrong_rate")["value"] for fs in FEATS}
            assert len(n) == 1 and len(w) == 1                         # 三組的列數與答錯率相同
            row["n_rows"], row["wrong_rate"] = int(n.pop()), w.pop()
            for m in ["roc_auc", "pr_auc"]:
                for fs in FEATS:
                    row[f"{m}_{fs}"] = one(df, model=fs, group=g, metric=m)["value"]
            for m in ["roc_auc", "pr_auc"]:
                for fs in FEATS:
                    row[f"{m}_{fs}_str"] = f4(row[f"{m}_{fs}"])
            if mt == "RF":
                for fs in FEATS:
                    row[f"setting_{fs}"] = rf_setting(rd("rf_cv.csv"), fs)
            rows.append(row)
        out[mt] = pd.DataFrame(rows)
    return out


# ====================================================================== (e) 跨學校並排
def table_e_cross_school():
    lr, rf = rd("r1_lr_paired.csv"), rd("rf_paired.csv")
    rf = rf[rf["analysis"] == "own_best"]
    xp, vs, sm = rd("xschool_paired.csv"), rd("xschool_r0_vs_student_split.csv"), rd("xschool_r0_a_summary.csv")
    rows = []

    def add(comp, mt, split, delta, sd, better, folds, weighted=np.nan):
        row = dict(comparison=comp, model_type=mt, split=split, delta_mean=delta, delta_sd=sd,
                   n_folds_better=int(better), weighted_mean=weighted, **dict(zip(FOLDS, folds)))
        row["delta_str"] = spm(delta, sd)
        rows.append(row)
    for comp in ["R0d - R0", "R1 - R0d"]:
        for mt, src in [("LR", lr), ("RF", rf)]:
            a = one(src, comparison=comp, eval="A")
            add(comp, mt, "student_split", a["delta_auc"], a["std"], a["n_folds_better"], [a[k] for k in FOLDS])
            if comp == "R1 - R0d":
                x = one(xp, model_type=mt, comparison=comp, eval="A")
                add(comp, mt, "school_group", x["delta"], x["std"], x["n_folds_better"], [x[k] for k in FOLDS], x["weighted_mean"])
            elif mt == "LR":                                              # R0d − R0 的學校群分組只有 LR
                v = vs[vs["split"] == "學校群分組（本實驗）"].iloc[0]
                w = one(sm, item="R0d − R0 ROC-AUC（未加權平均）")["weighted_mean"]
                add(comp, mt, "school_group", v["diff_mean"], v["diff_std"], v["n_folds_better"],
                    [v[f"diff_fold{k}"] for k in range(5)], w)
    df = pd.DataFrame(rows)                                            # 列已依 比較 × 模型 × 分組 的順序加入
    return df.reset_index(drop=True)


# ====================================================================== (f) RF 網格
def table_f_rf_grid():
    """R0／R0d／R1 各一張（來源 02_analysis/rf_grid_auc.csv）。R1r 與 R0d_rel 的網格在 03_process，不納入。"""
    df = rd("rf_grid_auc.csv")
    out = {}
    for fs in FEATS:
        g = df[df["features"] == fs]
        rows = []
        for depth in ["8", "16", "None"]:
            row = dict(max_depth=depth)
            for leaf in [5, 20, 50]:
                r = one(g, max_depth=depth, min_samples_leaf=leaf)
                row[f"leaf{leaf}_mean"], row[f"leaf{leaf}_sd"] = r["roc_auc_mean"], r["roc_auc_std"]
                row[f"leaf{leaf}_is_best"] = bool(r["is_best"])
            for leaf in [5, 20, 50]:
                row[f"leaf{leaf}_str"] = pm(row[f"leaf{leaf}_mean"], row[f"leaf{leaf}_sd"]) + (" *" if row[f"leaf{leaf}_is_best"] else "")
            rows.append(row)
        out[fs] = pd.DataFrame(rows)
    return out


# ====================================================================== (g) 各 fold 的絕對 AUC（資訊階梯圖）
def table_g_fold_auc():
    lrf, grid = pd.read_csv(R2 + "r1_lr_folds.csv"), rd("rf_grid_auc.csv")
    rows = []
    for mt in ["LR", "RF"]:
        for fs in FEATS:
            if mt == "LR":
                g = lrf[lrf["model"] == fs].sort_values("fold")
                roc, prv, setting = g["roc_auc"].to_numpy(), g["pr_auc"].to_numpy(), "StandardScaler+LR"
            else:
                b = one(grid[grid["features"] == fs], is_best=True)
                roc, prv, setting = np.array([b[k] for k in FOLDS]), np.full(5, np.nan), f"depth={b['max_depth']}, leaf={b['min_samples_leaf']}"
            row = dict(model_type=mt, features=fs, setting=setting,
                       **{f"roc_{k}": v for k, v in zip(FOLDS, roc)}, **{f"pr_{k}": v for k, v in zip(FOLDS, prv)})
            rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================== (h) 錯誤分析：分段表現
def na(x, fn):
    return "n/a" if pd.isna(x) else fn(x)


def table_h_error_segments():
    seg = rd("error_segments.csv")
    out = {}
    for mt in ["LR", "RF"]:
        d = seg[seg["model_type"] == mt]
        rows = []
        for stype, name in d[["segment_type", "segment"]].drop_duplicates().itertuples(index=False):
            sub = d[(d["segment_type"] == stype) & (d["segment"] == name)]
            g = lambda model, metric: one(sub, model=model, metric=metric)
            base = g("R0d", "roc_auc")
            dl = g("R1 - R0d", "delta_roc_auc")
            row = dict(segment_type=stype, segment=name, n_rows=int(base["n_rows"]), n_students=int(base["n_students"]), wrong_rate=base["wrong_rate"],
                       roc_auc_R0d=base["value"], roc_auc_R1=g("R1", "roc_auc")["value"],
                       pr_auc_R0d=g("R0d", "pr_auc")["value"], pr_auc_R1=g("R1", "pr_auc")["value"],
                       delta_roc_auc=dl["value"], delta_ci_low=dl["ci_low"], delta_ci_high=dl["ci_high"],
                       note=base["note"] if isinstance(base["note"], str) else np.nan)      # 沒有註記就留空（csv 讀回來是缺失值）
            for c in ["roc_auc_R0d", "roc_auc_R1", "pr_auc_R0d", "pr_auc_R1"]:
                row[f"{c}_str"] = na(row[c], f4)
            row["delta_str"] = "n/a" if pd.isna(dl["value"]) or pd.isna(dl["ci_low"]) else sci(dl["value"], dl["ci_low"], dl["ci_high"])
            rows.append(row)
        out[mt] = pd.DataFrame(rows)
    return out


# ====================================================================== (i) 錯誤分析：混淆矩陣與 slip／guess
def table_i_error_confusion():
    c = rd("error_confusion.csv")
    rows = []
    for (mt, fs), g in c.groupby(["model_type", "model"], sort=False):
        v = lambda m: one(g, metric=m)
        slip, guess = v("slip (pred correct, actual wrong)"), v("guess (pred wrong, actual correct)")
        row = dict(model_type=mt, model=fs, TP=int(v("TP")["count"]), FP=int(v("FP")["count"]), FN=int(v("FN")["count"]), TN=int(v("TN")["count"]),
                   slip_count=int(slip["count"]), slip_share_of_all_rows=slip["share_of_all_rows"], slip_share_of_actual_wrong=slip["share_within_actual_class"],
                   guess_count=int(guess["count"]), guess_share_of_all_rows=guess["share_of_all_rows"], guess_share_of_actual_correct=guess["share_within_actual_class"])
        row["slip_str"] = f"{row['slip_share_of_all_rows']:.4f} of all rows ({row['slip_share_of_actual_wrong']:.4f} of actual wrong)"
        row["guess_str"] = f"{row['guess_share_of_all_rows']:.4f} of all rows ({row['guess_share_of_actual_correct']:.4f} of actual correct)"
        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================== (j) 錯誤分析：校準
def table_j_calibration():
    cal, ece = rd("error_calibration.csv"), rd("error_calibration_ece.csv")
    out = {}
    for mt in ["LR", "RF"]:
        d = cal[cal["model_type"] == mt]
        rows = []
        for b in sorted(d["bin"].unique()):
            row = dict(bin=int(b))
            first = one(d[d["model"] == "R0d"], bin=b)
            row["bin_low"], row["bin_high"] = first["bin_low"], first["bin_high"]
            for fs in ["R0d", "R1"]:
                r = one(d[d["model"] == fs], bin=b)
                row[f"n_rows_{fs}"], row[f"mean_pred_{fs}"], row[f"obs_correct_rate_{fs}"] = int(r["n_rows"]), r["mean_pred"], r["obs_correct_rate"]
            for fs in ["R0d", "R1"]:
                row[f"calibration_{fs}_str"] = "n/a" if row[f"n_rows_{fs}"] == 0 else f"{row[f'mean_pred_{fs}']:.4f} predicted vs {row[f'obs_correct_rate_{fs}']:.4f} observed"
            rows.append(row)
        out[mt] = pd.DataFrame(rows)
    e = ece.copy()
    e["ece_str"] = e["ece"].map(f4)
    out["ece"] = e[["model_type", "model", "ece", "n_rows", "ece_str"]]
    return out


# ====================================================================== (k) 錯誤分析：高信心錯誤
def table_k_high_confidence():
    g, f = rd("error_high_conf_groups.csv"), rd("error_high_conf_features.csv")
    rows = []
    for _, r in g.iterrows():
        sub = f[(f["model_type"] == r["model_type"]) & (f["model"] == r["model"]) & (f["group"] == r["group"])]
        row = dict(model_type=r["model_type"], model=r["model"], group=r["group"], n_rows=int(r["n_rows"]), share_of_all_rows=r["share_of_all_rows"],
                   region_rows=int(r["region_rows"]), share_of_region=r["share_of_region"])
        for feat in ["position", "item_diff", "hint_rate", "last_correct", "log1p_c", "log1p_w", "has_history"]:
            x = one(sub, feature=feat)
            row[f"group_mean_{feat}"], row[f"all_rows_mean_{feat}"] = x["group_mean"], x["all_rows_mean"]
        row["share_str"] = f"{row['share_of_all_rows']:.4f} of all rows ({row['share_of_region']:.4f} of region)"
        for feat in ["position", "item_diff", "hint_rate", "last_correct", "log1p_c", "log1p_w", "has_history"]:
            row[f"{feat}_str"] = f"{row[f'group_mean_{feat}']:.4f} vs {row[f'all_rows_mean_{feat}']:.4f}"
        rows.append(row)
    return pd.DataFrame(rows)


# ====================================================================== 建立所有表
def build_all():
    a_A, a_C = table_a_main()
    b_A, b_C = table_b_paired()
    d = table_d_position()
    f = table_f_rf_grid()
    T = {
        "a_main_A": (a_A, ["r1_lr_cv.csv", "rf_cv.csv"], "主結果表（A）：LR／RF × R0／R0d／R1 的 ROC-AUC、PR-AUC、F1、Accuracy", "4. 主結果表"),
        "a_main_C": (a_C, ["r1_lr_cv.csv", "rf_cv.csv"], "主結果表（C）：t=6、t=11 的 ROC-AUC、PR-AUC 與 95% CI", "4. 主結果表"),
        "b_paired_A": (b_A, ["r1_lr_paired.csv", "rf_paired.csv"], "配對差異（A）：R0d−R0、R1−R0d；平均、標準差、各 fold 差值、較好 fold 數", "1. 資訊階梯圖（增量）；4. 主結果表"),
        "b_paired_C": (b_C, ["r1_lr_paired.csv", "rf_paired.csv"], "配對差異（C）：t=6、t=11 的 ΔAUC 與 95% CI", "1. 資訊階梯圖（增量）；4. 主結果表"),
        "c_ablation": (table_c_ablation(), ["ablation_lr_cv.csv", "ablation_lr_paired.csv"], "LR ablation：每組一列（對一個參考組）；A 的 AUC、ΔAUC、較好 fold 數、各 fold 差值，以及 C t=6／t=11 的 ΔAUC 與 CI", "2. 行為特徵群邊際貢獻圖"),
        "d_position_bins_LR": (d["LR"], ["r1_lr_cv.csv"], "位置分桶 B（LR）：桶為列，R0／R0d／R1 的 ROC-AUC、PR-AUC 為欄", "5. 位置分桶 B 表"),
        "d_position_bins_RF": (d["RF"], ["rf_cv.csv"], "位置分桶 B（RF，各組最佳設定）", "5. 位置分桶 B 表"),
        "e_cross_school": (table_e_cross_school(), ["r1_lr_paired.csv", "rf_paired.csv", "xschool_paired.csv", "xschool_r0_vs_student_split.csv", "xschool_r0_a_summary.csv"],
                           "跨學校：學生分組與學校群分組並排的 R0d−R0、R1−R0d（LR、RF），各 fold 差值", "3. 跨學校圖"),
        "g_fold_auc": (table_g_fold_auc(), ["r1_lr_folds.csv", "rf_grid_auc.csv"], "各 fold 的絕對 AUC：LR 與 RF（最佳設定）× R0／R0d／R1", "1. 資訊階梯圖（各 fold 值）"),
    }
    for fs, df in f.items():
        T[f"f_rf_grid_{fs}"] = (df, ["rf_grid_auc.csv"], f"RF 網格（{fs}）：max_depth（列）× min_samples_leaf（欄）的 A ROC-AUC，* = 最佳", "（補充；無對應候選）")
    # 錯誤分析（來源：02_analysis/error_*.csv）
    h = table_h_error_segments()
    for mt in ["LR", "RF"]:
        T[f"h_error_segments_{mt}"] = (h[mt], ["error_segments.csv"], f"錯誤分析・分段表現（{mt}）：位置／技能／item_diff 三等分／提示歷史，R0d 與 R1 的 AUC、R1−R0d 的 ΔAUC 與 95% CI", "8. 錯誤分析")
    T["i_error_confusion"] = (table_i_error_confusion(), ["error_confusion.csv"], "錯誤分析・混淆矩陣與 slip／guess（5 個 fold 合併；答錯為正類）", "8. 錯誤分析")
    j = table_j_calibration()
    for mt in ["LR", "RF"]:
        T[f"j_calibration_{mt}"] = (j[mt], ["error_calibration.csv"], f"錯誤分析・校準（{mt}）：10 個等寬桶，R0d 與 R1 的列數、平均預測機率、實際答對率", "8. 錯誤分析")
    T["j_calibration_ece"] = (j["ece"], ["error_calibration_ece.csv"], "錯誤分析・ECE（R0d、R1 × LR、RF）", "8. 錯誤分析")
    T["k_high_confidence"] = (table_k_high_confidence(), ["error_high_conf_groups.csv", "error_high_conf_features.csv"], "錯誤分析・高信心錯誤兩群的列數、占比與特徵平均（vs 全部列）", "8. 錯誤分析")
    return {k: dict(df=v[0], sources=v[1], desc=v[2], candidate=v[3]) for k, v in T.items()}


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    tables = build_all()
    for name, t in tables.items():
        t["df"].to_csv(OUT + name + ".csv", index=False)
        print(f"{name:24s} {len(t['df']):3d} 列 × {t['df'].shape[1]:2d} 欄   來源：{', '.join(t['sources'])}")
    print(f"\n已存 {len(tables)} 張表到 {OUT}")


if __name__ == "__main__":
    main()
