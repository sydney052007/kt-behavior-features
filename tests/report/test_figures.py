"""三張圖草稿的測試。可直接用 python3 tests/report/test_figures.py 跑。

確認：每張圖用到的數值（figures/data/*.csv）都能對回來源 csv；平均與 05_report_tables 的對應表一致；
圖檔存在且格式正確（PNG 300 dpi、SVG 可解析）；磁碟上的數據表與重新產生的一致。
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
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
    for name in ["fig2_ladder", "fig3_behavior_groups", "fig4_cross_school", "fig5_error_analysis"]:
        png, svg = f"{FIG}{name}.png", f"{FIG}{name}.svg"
        assert os.path.getsize(png) > 10_000 and os.path.getsize(svg) > 10_000, name
        dpi = Image.open(png).info.get("dpi")
        assert dpi is not None and round(dpi[0]) == 300 and round(dpi[1]) == 300, f"{name} 的 PNG dpi = {dpi}"
        assert ET.parse(svg).getroot().tag.endswith("svg")
    for f in ["fig2_ladder_levels", "fig2_ladder_increments", "fig3_behavior_groups", "fig3_reference_lines", "fig4_cross_school",
              "fig5_a_segment_delta_auc", "fig5_b_calibration"]:
        assert os.path.getsize(f"{DATA}{f}.csv") > 0


def test_consistent_encoding_constants():
    assert mf.COLORS == {"R0": "#0072B2", "R0d": "#E69F00", "R1": "#009E73"}       # Okabe-Ito（色盲友善）
    assert mf.MARKERS == {"LR": "o", "RF": "s"}


def test_data_tables_equal_fresh_build():
    lv, inc = mf.data_fig2()
    for name, fresh in [("fig2_ladder_levels", lv), ("fig2_ladder_increments", inc), ("fig3_behavior_groups", mf.data_fig3()),
                        ("fig4_cross_school", mf.data_fig4()), ("fig3_reference_lines", mf.data_fig3_refs()),
                        ("fig5_a_segment_delta_auc", mf.data_fig5a()), ("fig5_b_calibration", mf.data_fig5b())]:
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


def test_fig3_reference_lines_come_from_05_b_paired_A():
    refs = rd(DATA + "fig3_reference_lines.csv")
    assert list(refs["panel"]) == ["a_add_to_R0d", "b_remove_from_R1"]
    b = rd(R5 + "b_paired_A.csv")
    v = b[(b["model_type"] == "LR") & (b["analysis"] == "main") & (b["comparison"] == "R1 - R0d")]["delta_mean"].iloc[0]
    assert refs["value"].iloc[0] == v and refs["value"].iloc[1] == -v          # (b) 是同一個值取負號
    assert abs(round(v, 4) - 0.0048) < 1e-12                                    # 圖上標的 ±0.0048
    lr2 = rd(R2 + "r1_lr_paired.csv")
    src = lr2[(lr2["eval"] == "A") & (lr2["comparison"] == "R1 - R0d")]["delta_auc"].iloc[0]
    assert abs(v - src) <= TOL                                                  # 也對得回 02 的來源
    assert refs["label"].iloc[0] == "all behavior groups (R1 − R0d)" and refs["label"].iloc[1] == "all behavior groups removed (R0d vs R1)"


def test_no_label_overlaps_or_out_of_frame():
    """三張圖的標註不互相覆蓋、不超出圖框／座標軸、不壓到資料點與水平線（零線、參考線、平均線）。"""
    lv, inc = mf.data_fig2()
    refs = mf.data_fig3_refs()
    for name, fig in [("fig2", mf.fig2(lv, inc)), ("fig3", mf.fig3(mf.data_fig3(), refs)), ("fig4", mf.fig4(mf.data_fig4())),
                      ("fig5", mf.fig5(mf.data_fig5a(), mf.data_fig5b()))]:
        probs = mf.check_layout(fig)
        mf.plt.close(fig)
        assert not probs, f"{name}: {probs}"


def test_all_figure_data_csvs_are_tracked_by_git():
    import subprocess
    tracked = set(subprocess.run(["git", "ls-files", "figures/data"], capture_output=True, text=True, check=True).stdout.split())
    on_disk = {f"figures/data/{f}" for f in os.listdir(DATA) if f.endswith(".csv")}
    assert on_disk and on_disk <= tracked, f"尚未追蹤：{sorted(on_disk - tracked)}"


def test_fig5_data_matches_error_csvs():
    a, b = rd(DATA + "fig5_a_segment_delta_auc.csv"), rd(DATA + "fig5_b_calibration.csv")
    assert len(a) == 22 and len(b) == 20                                          # 11 分段 × 2 模型；10 bin × 2 模型
    # ---- (a) 分段 ΔAUC：每個點的 value、CI、n_rows 都與 error_segments.csv 完全一致，且來源恰好就是這 22 列
    seg = rd(R2 + "error_segments.csv")
    src = seg[(seg["model"] == "R1 - R0d") & (seg["metric"] == "delta_roc_auc")]
    assert len(src) == 22
    key = ["model_type", "segment_type", "segment"]
    assert set(map(tuple, a[key].to_numpy())) == set(map(tuple, src[key].to_numpy()))
    m = a.merge(src, on=key, suffixes=("", "_src"), validate="one_to_one")
    assert (m["delta_roc_auc"] == m["value"]).all() and (m["ci_low"] == m["ci_low_src"]).all()
    assert (m["ci_high"] == m["ci_high_src"]).all() and (m["n_rows"] == m["n_rows_src"]).all()
    assert (a["ci_low"] <= a["delta_roc_auc"]).all() and (a["delta_roc_auc"] <= a["ci_high"]).all()
    # 11 個分段（位置 4、技能 2、item_diff 3、提示歷史 2），LR 與 RF 各一個點，列數相同
    per = a.groupby(["segment_type", "segment"])
    assert len(per) == 11 and (per["model_type"].nunique() == 2).all() and (per["n_rows"].nunique() == 1).all()
    assert a.groupby("segment_type")["segment"].nunique().to_dict() == {"position": 4, "skill": 2, "item_diff": 3, "hint_history": 2}
    assert list(a.drop_duplicates("row_order")["segment"]) == ["1", "2-4", "5-10", ">=11", "47", "311", "low (<=q33)", "mid (q33-q67]",
                                                               "high (>q67)", "hint_rate=0 (incl. t=1)", "hint_rate>0"]
    # ---- (b) 校準：R1 的 10 個 bin；列數、平均預測機率、實際答對率、ECE 都與 error_calibration*.csv 一致
    cal, ece = rd(R2 + "error_calibration.csv"), rd(R2 + "error_calibration_ece.csv")
    c = cal[cal["model"] == "R1"]
    assert len(c) == 20 and set(b["model"]) == {"R1"}
    mb = b.merge(c, on=["model_type", "bin"], suffixes=("", "_src"), validate="one_to_one")
    for col in ["bin_low", "bin_high", "n_rows", "mean_pred", "obs_correct_rate"]:
        assert (mb[col] == mb[col + "_src"]).all(), col
    for mt in ["LR", "RF"]:
        assert b[b["model_type"] == mt]["n_rows"].sum() == 30516
        e = ece[(ece["model_type"] == mt) & (ece["model"] == "R1")]["ece"].iloc[0]
        assert (b[b["model_type"] == mt]["ece"] == e).all()


def test_gitignore_tracks_figures_data_but_not_data():
    lines = [l.strip() for l in open(".gitignore", encoding="utf-8")]
    assert lines.count("data/") == 1 and lines.count("!figures/data/") == 1
    import subprocess
    tracked = subprocess.run(["git", "ls-files", "data"], capture_output=True, text=True, check=True).stdout.split()
    assert tracked == [], f"data/ 底下不該有被追蹤的檔案：{tracked}"


def test_fig5_presentation_rf_translucent_lr_on_top_and_no_ticks_on_group_rows():
    a, b = mf.data_fig5a(), mf.data_fig5b()
    fig = mf.fig5(a, b)
    ax, bx = fig.axes[0], fig.axes[1]
    # 只看有 10 個資料點的集合（右下角「bin 大小」圖例用的空 scatter 不算）；形狀不同：LR 圓、RF 方
    data_cols = [c for c in bx.collections if len(c.get_offsets()) == 10]
    assert len(data_cols) == 2
    cols = {"RF" if c.get_alpha() is not None and c.get_alpha() < 1 else "LR": c for c in data_cols}
    assert set(cols) == {"RF", "LR"}
    assert abs(cols["RF"].get_alpha() - 0.6) < 1e-9 and cols["LR"].get_alpha() in (None, 1.0)      # RF 半透明
    assert cols["LR"].get_zorder() > cols["RF"].get_zorder()                                      # LR 在上層
    ax.figure.canvas.draw()
    labels = [tl.get_text() for tl in ax.get_yticklabels()]
    ticks = ax.yaxis.get_major_ticks()
    groups = {"Position", "Skill", "Item difficulty (tertile)", "Hint history"}
    for tl, tick in zip(ax.get_yticklabels(), ticks):
        assert tick.tick1line.get_visible() == (tl.get_text() not in groups), tl.get_text()       # 只有組名列沒有刻度短線
    assert groups <= set(labels)
    mf.plt.close(fig)


if __name__ == "__main__":
    tests = [test_files_exist_and_formats, test_consistent_encoding_constants, test_data_tables_equal_fresh_build,
             test_fig2_values_trace_back_and_means_match_05, test_fig3_values_trace_back_and_means_match_05,
             test_fig4_values_trace_back_and_means_match_05, test_every_plotted_number_exists_in_some_source,
             test_fig3_reference_lines_come_from_05_b_paired_A, test_no_label_overlaps_or_out_of_frame,
             test_all_figure_data_csvs_are_tracked_by_git, test_fig5_data_matches_error_csvs,
             test_gitignore_tracks_figures_data_but_not_data,
             test_fig5_presentation_rf_translucent_lr_on_top_and_no_ticks_on_group_rows]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
