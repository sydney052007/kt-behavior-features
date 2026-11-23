"""錯誤分析的測試。可直接用 python3 tests/analysis/test_error_analysis.py 跑。

(a) A 的重現檢查（重新產生 OOF，與既有結果的差 < 1e-9）；(b) 各分段列數加總 = 總列數，且每種分段互斥且涵蓋全部列；
(c) 校準桶的列數加總 = 總列數；(d) 所有 error_*.csv（與 05 的錯誤分析寬表）沒有 user_id、school_id 欄位，也不是逐列資料。
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
import run_error_analysis as ea  # noqa: E402
from evaluate import paired_delta_c  # noqa: E402

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
    # 高信心表：用重新產生的 OOF 重算，n_rows、region_rows 逐列與存好的 csv 完全相等
    fresh_hg = ea.high_confidence_tables(runs)[0]
    key = ["model_type", "model", "group"]
    assert len(fresh_hg) == len(hg) == 8 and not hg.duplicated(key).any()
    a, b = fresh_hg.set_index(key), hg.set_index(key)
    assert set(a.index) == set(b.index)
    for col in ["n_rows", "region_rows"]:
        assert (a[col] == b.loc[a.index, col]).all(), f"{col} 與存好的 error_high_conf_groups.csv 不同"


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


def test_e_hint_excl_t1_supplement_is_consistent_with_original_segments():
    sup, seg = _csv("error_hint_excl_t1.csv"), _csv("error_segments.csv")
    assert (sup["note"].str.contains("descriptive supplement")).all()
    base = seg[(seg["model"] == "R0d") & (seg["metric"] == "roc_auc")]
    for mt in ["LR", "RF"]:
        b = base[base["model_type"] == mt].set_index(["segment_type", "segment"])
        n_t1 = int(b.loc[("position", "1"), "n_rows"])
        s = sup[(sup["model_type"] == mt) & (sup["model"] == "R0d") & (sup["metric"] == "roc_auc")].set_index("segment")
        # 排除 t=1：'= 0' 少了 t=1 的列，'> 0' 完全不變；兩段加總 = 總列數 − t=1 列數
        assert s.loc["hint_rate=0 (excl. t=1)", "n_rows"] == b.loc[("hint_history", "hint_rate=0 (incl. t=1)"), "n_rows"] - n_t1
        assert s.loc["hint_rate>0", "n_rows"] == b.loc[("hint_history", "hint_rate>0"), "n_rows"]
        assert s["n_rows"].sum() == N - n_t1 and n_t1 == 1841
    # '> 0' 的列與 bootstrap 設定都沒變（同樣的列、同一個種子）→ ΔAUC 與 CI 必須與原本完全相同
    d_sup = sup[(sup["segment"] == "hint_rate>0") & (sup["model"] == "R1 - R0d")].set_index("model_type")
    d_old = seg[(seg["segment"] == "hint_rate>0") & (seg["model"] == "R1 - R0d")].set_index("model_type")
    for mt in ["LR", "RF"]:
        for col in ["value", "ci_low", "ci_high"]:
            assert abs(d_sup.loc[mt, col] - d_old.loc[mt, col]) < 1e-12, (mt, col)
    dd = sup[sup["model"] == "R1 - R0d"]
    assert len(dd) == 4 and (dd["ci_low"] <= dd["value"]).all() and (dd["value"] <= dd["ci_high"]).all()


def _fake_scores(n_users=60, per_user=12, seed=0):
    """每位學生多列的假資料：回傳 (y 答錯為正類, 舊模型分數, 新模型分數, user_id, t)。"""
    rng = np.random.default_rng(seed)
    uid = np.repeat(np.arange(n_users), per_user)
    t = np.tile(np.arange(1, per_user + 1), n_users)
    y = rng.integers(0, 2, len(uid))
    s_old = np.clip(0.35 + 0.2 * y + rng.random(len(uid)) * 0.5, 0.01, 0.99)
    s_new = np.clip(0.25 + 0.4 * y + rng.random(len(uid)) * 0.4, 0.01, 0.99)
    return y, s_old, s_new, uid, t


def test_f_paired_delta_ci_zero_for_identical_models_and_reproducible():
    y, s_old, s_new, uid, _ = _fake_scores()
    # 兩個模型分數相同 → 點估計與整個 CI 都是 0（若不是配對重抽，CI 不會是 0）
    pt, lo, hi = ea.paired_delta_ci(y, s_old, s_old.copy(), uid, n_boot=200, seed=42)
    assert pt == 0 and lo == 0 and hi == 0
    # 兩個不同的模型：固定種子重跑完全相同，換種子 CI 會不同（確認種子真的有作用）
    r1 = ea.paired_delta_ci(y, s_old, s_new, uid, n_boot=200, seed=42)
    r2 = ea.paired_delta_ci(y, s_old, s_new, uid, n_boot=200, seed=42)
    r3 = ea.paired_delta_ci(y, s_old, s_new, uid, n_boot=200, seed=43)
    assert r1 == r2 and r1[0] == r3[0] and r1[1:] != r3[1:]
    assert r1[1] < r1[2]                                      # 區間不是退化的


def test_g_paired_delta_ci_equals_evaluate_paired_delta_c():
    """paired_delta_ci 與 evaluate.paired_delta_c 的重抽程序相同（種子、依學生分組的順序、cluster_resample_indices、
    略過單一類別的規則），所以同一批列（t == 切點）、同一個種子與 n_boot 下，點估計與 CI 都應相等。"""
    y, s_old, s_new, uid, t = _fake_scores()
    base = pd.DataFrame({"user_id": uid, "order_id": np.arange(len(uid)), "t": t, "correct": 1 - y})
    old, new = base.assign(p_correct=1 - s_old), base.assign(p_correct=1 - s_new)
    for k in (6, 11):
        out = paired_delta_c(old, new, cut_points=(k,), n_boot=200, seed=42).iloc[0]
        m = t == k                                            # 每位學生恰 1 列；列順序 = order_id 順序
        pt, lo, hi = ea.paired_delta_ci(y[m], s_old[m], s_new[m], uid[m], n_boot=200, seed=42)
        assert abs(pt - out["delta_auc"]) < 1e-12 and abs(lo - out["ci_low"]) < 1e-12 \
            and abs(hi - out["ci_high"]) < 1e-12, (k, (pt, lo, hi), out.to_dict())
        assert lo < hi


if __name__ == "__main__":
    tests = [test_a_reproduces_existing_A_results, test_b_segments_partition_all_rows_and_counts_sum_to_total,
             test_c_calibration_bins_sum_to_total_and_ece, test_confusion_and_high_confidence_counts,
             test_d_no_student_level_columns_and_no_row_level_data,
             test_e_hint_excl_t1_supplement_is_consistent_with_original_segments,
             test_f_paired_delta_ci_zero_for_identical_models_and_reproducible,
             test_g_paired_delta_ci_equals_evaluate_paired_delta_c]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
