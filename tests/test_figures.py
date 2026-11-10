"""三張圖草稿的測試。可直接用 python3 tests/test_figures.py 跑。

確認：每張圖用到的數值（figures/data/*.csv）都能對回來源 csv；平均與 05_report_tables 的對應表一致；
圖檔存在且格式正確（PNG 300 dpi、SVG 可解析）；磁碟上的數據表與重新產生的一致。
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import make_figures as mf  # noqa: E402

R2, R5, DATA, FIG = mf.R2, mf.R5, mf.DATA, mf.FIG
TOL = 1e-12
FOLDS = mf.FOLDS


def rd(path):
    return mf.rd(path)


def all_numbers(files):
    """來源檔案裡所有數值儲存格（排序後的 array，用來做容差比對）。"""
    vals = []
    for f in files:
        d = rd(f)
        for c in d.columns:
            if pd.api.types.is_bool_dtype(d[c]):
                continue
            vals.extend(pd.to_numeric(d[c], errors="coerce").dropna().tolist())
    return np.sort(np.array(vals, dtype=float))


def found(v, arr):
    i = np.searchsorted(arr, v - TOL)
    return i < len(arr) and abs(arr[i] - v) <= TOL


def test_files_exist_and_formats():
    for name in ["fig2_ladder", "fig3_behavior_groups", "fig4_cross_school"]:
        png, svg = f"{FIG}{name}.png", f"{FIG}{name}.svg"
        assert os.path.getsize(png) > 10_000 and os.path.getsize(svg) > 10_000, name
        dpi = Image.open(png).info.get("dpi")
        assert dpi is not None and round(dpi[0]) == 300 and round(dpi[1]) == 300, f"{name} 的 PNG dpi = {dpi}"
        assert ET.parse(svg).getroot().tag.endswith("svg")
    for f in ["fig2_ladder_levels", "fig2_ladder_increments", "fig3_behavior_groups", "fig4_cross_school"]:
        assert os.path.getsize(f"{DATA}{f}.csv") > 0


def test_consistent_encoding_constants():
    assert mf.COLORS == {"R0": "#0072B2", "R0d": "#E69F00", "R1": "#009E73"}       # Okabe-Ito（色盲友善）
    assert mf.MARKERS == {"LR": "o", "RF": "s"}


def test_data_tables_equal_fresh_build():
    lv, inc = mf.data_fig2()
    for name, fresh in [("fig2_ladder_levels", lv), ("fig2_ladder_increments", inc), ("fig3_behavior_groups", mf.data_fig3()),
                        ("fig4_cross_school", mf.data_fig4())]:
        disk = rd(f"{DATA}{name}.csv")
        pd.testing.assert_frame_equal(disk, fresh.reset_index(drop=True), check_exact=True, check_dtype=False, obj=name)


def test_fig2_values_trace_back_and_means_match_05():
    lv, inc = rd(DATA + "fig2_ladder_levels.csv"), rd(DATA + "fig2_ladder_increments.csv")
    assert len(lv) == 30 and len(inc) == 20
    # 各 fold 的絕對 AUC：LR 來自 r1_lr_folds.csv，RF 來自 rf_grid_auc.csv 的最佳設定列
    lrf, grid = rd(R2 + "r1_lr_folds.csv"), rd(R2 + "rf_grid_auc.csv")
    for _, r in lv.iterrows():
        if r["model_type"] == "LR":
            src = lrf[(lrf["model"] == r["features"]) & (lrf["fold"] == r["fold"])]["roc_auc"].iloc[0]
        else:
            b = grid[(grid["features"] == r["features"]) & (grid["is_best"])].iloc[0]
            src = b[f"fold{int(r['fold'])}"]
        assert abs(r["roc_auc"] - src) <= TOL, r.to_dict()
    # 平均與 05 的 a_main_A 一致（圖上標的數字），且等於各 fold 值的平均（一致性檢查）
    a = rd(R5 + "a_main_A.csv").set_index(["model_type", "features"])
    for (mt, fs), g in lv.groupby(["model_type", "features"]):
        assert g["mean_roc_auc"].nunique() == 1 and g["mean_roc_auc"].iloc[0] == a.at[(mt, fs), "roc_auc_mean"]
        assert abs(g["roc_auc"].mean() - g["mean_roc_auc"].iloc[0]) < 1e-9, (mt, fs)
    # 增量：與 05 的 b_paired_A 完全一致，並能回到 02 的 r1_lr_paired.csv / rf_paired.csv
    b5 = rd(R5 + "b_paired_A.csv")
    lr2, rf2 = rd(R2 + "r1_lr_paired.csv"), rd(R2 + "rf_paired.csv")
    for (mt, step), g in inc.groupby(["model_type", "step"]):
        comp = {"R0->R0d": "R0d - R0", "R0d->R1": "R1 - R0d"}[step]
        an = "main" if mt == "LR" else "own_best"
        row5 = b5[(b5["model_type"] == mt) & (b5["analysis"] == an) & (b5["comparison"] == comp)].iloc[0]
        row2 = (lr2[(lr2["eval"] == "A") & (lr2["comparison"] == comp)] if mt == "LR"
                else rf2[(rf2["analysis"] == "own_best") & (rf2["eval"] == "A") & (rf2["comparison"] == comp)]).iloc[0]
        assert g["mean_delta_auc"].iloc[0] == row5["delta_mean"] and abs(row5["delta_mean"] - row2["delta_auc"]) <= TOL
        for _, r in g.iterrows():
            k = int(r["fold"])
            assert r["delta_auc"] == row5[f"fold{k}"] and abs(r["delta_auc"] - row2[f"fold{k}"]) <= TOL


def test_fig3_values_trace_back_and_means_match_05():
    d = rd(DATA + "fig3_behavior_groups.csv")
    assert len(d) == 40 and set(d["group"]) == {"hint", "time", "att", "scaffold"}
    p2 = rd(R2 + "ablation_lr_paired.csv")
    p2 = p2[p2["eval"] == "A"].set_index(["model", "reference"])
    c5 = rd(R5 + "c_ablation.csv").set_index(["group", "reference"])
    for (panel, g), s in d.groupby(["panel", "group"]):
        prefix, ref = ("R0d+", "R0d") if panel == "a_add_to_R0d" else ("R1-", "R1")
        row2, row5 = p2.loc[(prefix + g, ref)], c5.loc[(prefix + g, ref)]
        assert s["mean_delta_auc"].iloc[0] == row2["delta_auc"] and abs(row2["delta_auc"] - row5["delta_auc_mean"]) <= TOL
        for _, r in s.iterrows():
            k = int(r["fold"])
            assert r["delta_auc"] == row2[f"fold{k}"] and abs(r["delta_auc"] - row5[f"fold{k}"]) <= TOL


def test_fig4_values_trace_back_and_means_match_05():
    d = rd(DATA + "fig4_cross_school.csv")
    assert len(d) == 35                                                           # 7 組 × 5 個 fold
    # RF 的 R0d−R0 沒有學校群分組（not run）：圖上沒有這組的資料
    assert d[(d["panel"] == "a_R0d_minus_R0") & (d["model_type"] == "RF") & (d["split"] == "school_group")].empty
    # 學校群分組的 fold 0（最大組）恰好有 3 組各 1 個點被標成特殊標記
    flagged = d[d["is_largest_group_fold"]]
    assert len(flagged) == 3 and set(flagged["fold"]) == {0} and set(flagged["split"]) == {"school_group"}
    e5 = rd(R5 + "e_cross_school.csv")
    srcs = {"student_LR": rd(R2 + "r1_lr_paired.csv"), "student_RF": rd(R2 + "rf_paired.csv"),
            "school": rd(R2 + "xschool_paired.csv"), "school_r0": rd(R2 + "xschool_r0_vs_student_split.csv")}
    for (panel, mt, split), s in d.groupby(["panel", "model_type", "split"]):
        comp = "R0d - R0" if panel == "a_R0d_minus_R0" else "R1 - R0d"
        row5 = e5[(e5["comparison"] == comp) & (e5["model_type"] == mt) & (e5["split"] == split)].iloc[0]
        assert s["mean_delta_auc"].iloc[0] == row5["delta_mean"]
        if split == "student_split":
            if mt == "LR":
                t = srcs["student_LR"]; r2 = t[(t["eval"] == "A") & (t["comparison"] == comp)].iloc[0]
            else:
                t = srcs["student_RF"]; r2 = t[(t["analysis"] == "own_best") & (t["eval"] == "A") & (t["comparison"] == comp)].iloc[0]
            mean2, folds2 = r2["delta_auc"], [r2[f"fold{k}"] for k in range(5)]
        elif comp == "R1 - R0d":
            t = srcs["school"]; r2 = t[(t["model_type"] == mt) & (t["comparison"] == comp) & (t["eval"] == "A")].iloc[0]
            mean2, folds2 = r2["delta"], [r2[f"fold{k}"] for k in range(5)]
        else:
            t = srcs["school_r0"]; r2 = t[t["split"] == "學校群分組（本實驗）"].iloc[0]
            mean2, folds2 = r2["diff_mean"], [r2[f"diff_fold{k}"] for k in range(5)]
        assert abs(row5["delta_mean"] - mean2) <= TOL
        for _, r in s.sort_values("fold").iterrows():
            k = int(r["fold"])
            assert r["delta_auc"] == row5[f"fold{k}"] and abs(r["delta_auc"] - folds2[k]) <= TOL


def test_every_plotted_number_exists_in_some_source():
    srcs = {"fig2_ladder_levels": ["r1_lr_folds.csv", "rf_grid_auc.csv", "r1_lr_cv.csv", "rf_cv.csv"],
            "fig2_ladder_increments": ["r1_lr_paired.csv", "rf_paired.csv"],
            "fig3_behavior_groups": ["ablation_lr_paired.csv"],
            "fig4_cross_school": ["r1_lr_paired.csv", "rf_paired.csv", "xschool_paired.csv", "xschool_r0_vs_student_split.csv"]}
    for name, files in srcs.items():
        arr = all_numbers([R2 + f for f in files])
        d = rd(f"{DATA}{name}.csv")
        for c in d.columns:
            if c in ("fold",) or pd.api.types.is_bool_dtype(d[c]) or not pd.api.types.is_numeric_dtype(d[c]):
                continue
            for v in d[c]:
                assert found(float(v), arr), f"{name}.{c} 的 {v} 在來源 {files} 找不到"


if __name__ == "__main__":
    tests = [test_files_exist_and_formats, test_consistent_encoding_constants, test_data_tables_equal_fresh_build,
             test_fig2_values_trace_back_and_means_match_05, test_fig3_values_trace_back_and_means_match_05,
             test_fig4_values_trace_back_and_means_match_05, test_every_plotted_number_exists_in_some_source]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
