"""錯誤分析的測試。可直接用 python3 tests/test_error_analysis.py 跑。

(a) A 的重現檢查（重新產生 OOF，與既有結果的差 < 1e-9）；(b) 各分段列數加總 = 總列數，且每種分段互斥且涵蓋全部列；
(c) 校準桶的列數加總 = 總列數；(d) 所有 error_*.csv（與 05 的錯誤分析寬表）沒有 user_id、school_id 欄位，也不是逐列資料。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import run_error_analysis as ea  # noqa: E402

OUT = ea.OUT
N = 30516
_cache = {}


def _runs():
    if "runs" not in _cache:
        _cache["runs"] = ea.build_runs()
    return _cache["runs"]


def _csv(name):
    return pd.read_csv(OUT + name, float_precision="round_trip")


def test_a_reproduces_existing_A_results():
    runs = _runs()
    assert set(runs) == {("LR", "R0d"), ("LR", "R1"), ("RF", "R0d"), ("RF", "R1")}
    rep = ea.reproduction_check(runs)
    assert len(rep) == 8 and rep["abs_diff"].max() < 1e-9, rep
    # 存好的重現檢查檔也要通過
    saved = _csv("error_reproduction.csv")
    assert len(saved) == 8 and saved["abs_diff"].max() < 1e-9
    # 只用 trainval：OOF 的列數與 fold 結構
    for r in runs.values():
        o = r["oof"]
        assert len(o) == N and o["order_id"].is_unique and set(o["fold"]) == set(range(5))


def test_b_segments_partition_all_rows_and_counts_sum_to_total():
    runs = _runs()
    oof = runs[("LR", "R1")]["oof"]
    cuts = ea.item_diff_cuts(oof)
    assert abs(cuts[0] - np.quantile(oof["item_diff"], 0.33)) < 1e-15 and abs(cuts[1] - np.quantile(oof["item_diff"], 0.67)) < 1e-15
    segs = ea.segments(oof, cuts)
    by_type = {}
    for stype, name, m in segs:
        by_type.setdefault(stype, []).append(m)
    assert set(by_type) == {"position", "skill", "item_diff", "hint_history"}
    for stype, masks in by_type.items():
        assert (np.sum(masks, axis=0) == 1).all(), f"{stype}：各分段必須互斥且涵蓋全部列"      # 每一列恰好在一段
        assert sum(int(m.sum()) for m in masks) == N
    # t=1 的列一律歸入 hint_rate = 0
    has = [m for st, nm, m in segs if st == "hint_history" and nm.startswith("hint_rate>0")][0]
    assert not has[(oof["t"] == 1).to_numpy()].any()
    # 存好的分段表：每個模型類型、每種分段類型的 n_rows 加總 = 總列數
    seg = _csv("error_segments.csv")
    base = seg[(seg["model"] == "R0d") & (seg["metric"] == "roc_auc")]
    for mt in ["LR", "RF"]:
        for stype in by_type:
            part = base[(base["model_type"] == mt) & (base["segment_type"] == stype)]
            assert part["n_rows"].sum() == N, (mt, stype)
    # R0d 與 R1 在同一分段的列數、答錯率相同；ΔAUC 的 CI 包住點估計
    d = seg[seg["model"] == "R1 - R0d"]
    assert len(d) == 22 and (d["ci_low"] <= d["value"]).all() and (d["value"] <= d["ci_high"]).all()


def test_c_calibration_bins_sum_to_total_and_ece():
    runs = _runs()
    cal, ece = _csv("error_calibration.csv"), _csv("error_calibration_ece.csv")
    assert len(cal) == 40 and len(ece) == 4
    for (mt, fs), g in cal.groupby(["model_type", "model"]):
        assert g["n_rows"].sum() == N and len(g) == 10, (mt, fs)
        # 由校準表自己算 ECE，要等於 ece 檔
        gg = g[g["n_rows"] > 0]
        e = float((gg["n_rows"] / N * (gg["obs_correct_rate"] - gg["mean_pred"]).abs()).sum())
        assert abs(e - ece[(ece["model_type"] == mt) & (ece["model"] == fs)]["ece"].iloc[0]) < 1e-12
    # 與重新產生的 OOF 重算一致
    fresh_cal, fresh_ece = ea.calibration_tables(runs)
    assert (fresh_cal["n_rows"].to_numpy() == cal["n_rows"].to_numpy()).all()
    assert np.allclose(fresh_ece["ece"].to_numpy(), ece["ece"].to_numpy(), atol=1e-12)


def test_confusion_and_high_confidence_counts():
    conf, hg = _csv("error_confusion.csv"), _csv("error_high_conf_groups.csv")
    for (mt, fs), g in conf.groupby(["model_type", "model"]):
        c = g.set_index("metric")["count"]
        assert c["TP"] + c["FP"] + c["FN"] + c["TN"] == N
        assert c["slip (pred correct, actual wrong)"] == c["FN"] and c["guess (pred wrong, actual correct)"] == c["FP"]
    assert (hg["n_rows"] <= hg["region_rows"]).all() and (hg["share_of_all_rows"] < 0.1).all()
    runs = _runs()
    fresh = ea.confusion_table(runs)
    assert (fresh["count"].to_numpy() == conf["count"].to_numpy()).all()


def test_d_no_student_level_columns_and_no_row_level_data():
    files = sorted(glob.glob(OUT + "error_*.csv")) + sorted(glob.glob("results/05_report_tables/*error*.csv")) + \
        sorted(glob.glob("results/05_report_tables/*calibration*.csv")) + sorted(glob.glob("results/05_report_tables/k_high_confidence.csv"))
    assert len(files) >= 8 + 7, files
    for f in files:
        d = pd.read_csv(f)
        cols = [c.lower() for c in d.columns]
        assert not any(c in ("user_id", "school_id", "order_id", "teacher_id", "student_class_id", "problem_id") for c in cols), f
        assert len(d) < 500, f"{f} 有 {len(d)} 列，不像彙總表"                      # 彙總表很小；逐列資料會有幾萬列
    # OOF 預測不存進 repo：results/ 與 figures/ 底下沒有任何一個 csv 有 p_correct 欄
    for f in glob.glob("results/**/*.csv", recursive=True) + glob.glob("figures/**/*.csv", recursive=True):
        assert "p_correct" not in pd.read_csv(f, nrows=0).columns, f


if __name__ == "__main__":
    tests = [test_a_reproduces_existing_A_results, test_b_segments_partition_all_rows_and_counts_sum_to_total,
             test_c_calibration_bins_sum_to_total_and_ece, test_confusion_and_high_confidence_counts,
             test_d_no_student_level_columns_and_no_row_level_data]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
