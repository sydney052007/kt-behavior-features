"""報告用寬表的測試。可直接用 python3 tests/test_report_tables.py 跑。

重點：寬表裡的每個數字都必須能在它的來源長格式檔案中找到相同的值（寬表不重算任何東西）；
數值欄是數字、字串欄只放格式化文字；磁碟上的表與重新產生的結果完全相同（沒有過期）。
"""
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from make_report_tables import FEATS, ID_COLS, OUT, R2, build_all, rd  # noqa: E402

_cache = {}


def _tables():
    if "t" not in _cache:
        _cache["t"] = build_all()
    return _cache["t"]


def _read(name):
    """讀磁碟上的寬表；*_str 欄位明確讀成文字（csv 沒有型別，內容像 "0.4834" 的字串會被當成數字）。"""
    header = pd.read_csv(OUT + name + ".csv", nrows=0).columns
    dtype = {c: str for c in header if c.endswith("_str") or c == "max_depth"}
    return pd.read_csv(OUT + name + ".csv", dtype=dtype, keep_default_na=False, na_values=[""])


def _numeric_cols(df):
    return [c for c in df.columns if c not in ID_COLS and not c.endswith("_str")]


def _source_numbers(files):
    """來源檔案裡所有數值儲存格（排除布林與 NaN），用 float 表示。"""
    vals = set()
    for f in files:
        d = pd.read_csv(R2 + f, dtype={"max_depth": str}, keep_default_na=False, na_values=[""])
        for c in d.columns:
            col = d[c]
            if pd.api.types.is_bool_dtype(col):
                continue
            num = pd.to_numeric(col, errors="coerce").dropna()
            vals.update(float(v) for v in num)
    return vals


def test_all_expected_tables_exist_and_match_fresh_build():
    t = _tables()
    expected = {"a_main_A", "a_main_C", "b_paired_A", "b_paired_C", "c_ablation", "d_position_bins_LR", "d_position_bins_RF",
                "e_cross_school", "g_fold_auc", "f_rf_grid_R0", "f_rf_grid_R0d", "f_rf_grid_R1",
                "h_error_segments_LR", "h_error_segments_RF", "i_error_confusion", "j_calibration_LR", "j_calibration_RF",
                "j_calibration_ece", "k_high_confidence"}
    assert set(t) == expected
    for name, info in t.items():
        disk, fresh = _read(name), info["df"]
        assert list(disk.columns) == list(fresh.columns), name
        pd.testing.assert_frame_equal(disk, fresh.reset_index(drop=True), check_exact=True, check_dtype=False,
                                      obj=f"{name}（磁碟 vs 重新產生）")


def test_every_number_is_found_in_source_files():
    for name, info in _tables().items():
        src = _source_numbers(info["sources"])
        df = info["df"]
        n = 0
        for c in _numeric_cols(df):
            col = df[c]
            if pd.api.types.is_bool_dtype(col):
                continue
            assert pd.api.types.is_numeric_dtype(col), f"{name}.{c} 應該是數值欄，卻是 {col.dtype}"
            for v in col.dropna():
                assert float(v) in src, f"{name}.{c} 的 {v} 在來源 {info['sources']} 找不到"
                n += 1
        assert n > 0, name


def test_numeric_columns_are_not_strings_and_string_columns_are_text():
    for name, info in _tables().items():
        df = _read(name)
        for c in df.columns:
            if c.endswith("_str"):
                assert df[c].map(lambda x: isinstance(x, str)).all(), f"{name}.{c} 應該是字串"
            elif c in ID_COLS:
                continue
            else:
                assert pd.api.types.is_numeric_dtype(df[c]) or pd.api.types.is_bool_dtype(df[c]), f"{name}.{c} 不是數值欄（{df[c].dtype}）"


def test_formatted_strings_agree_with_numeric_columns():
    tok = re.compile(r"[+-]?\d+\.\d{4}")
    for name, info in _tables().items():
        df = info["df"]
        num = [c for c in _numeric_cols(df) if not pd.api.types.is_bool_dtype(df[c])]
        for c in [c for c in df.columns if c.endswith("_str")]:
            for i, s in df[c].items():
                tokens = tok.findall(s)
                assert tokens, (name, c, s)
                row_vals = [float(v) for v in df.loc[i, num].dropna()]
                for t in tokens:
                    assert any(abs(float(t) - round(v, 4)) < 1e-9 for v in row_vals), f"{name}.{c}[{i}] = {s!r} 的 {t} 對不到同一列的數值欄"
                assert all(len(t.split(".")[1]) == 4 for t in tokens)                  # 4 位小數


def test_a_b_c_are_separate_tables_with_expected_columns():
    t = _tables()
    a, c = t["a_main_A"]["df"], t["a_main_C"]["df"]
    for m in ["roc_auc", "pr_auc", "f1", "accuracy"]:
        assert f"{m}_mean" in a.columns and f"{m}_sd" in a.columns and f"{m}_str" in a.columns
    assert not any(col.startswith(("t6_", "t11_")) for col in a.columns)             # A 表沒有 C 的欄
    for tag in ["t6", "t11"]:
        for m in ["roc_auc", "pr_auc"]:
            assert {f"{tag}_{m}", f"{tag}_{m}_ci_low", f"{tag}_{m}_ci_high", f"{tag}_{m}_str"} <= set(c.columns)
    assert not any(col.endswith("_sd") for col in c.columns)                          # C 表沒有 A 的 sd 欄
    assert len(a) == len(c) == 6 and set(a["model_type"]) == {"LR", "RF"} and set(a["features"]) == set(FEATS)
    # B 表：LR 與 RF 各一張，桶為列
    for nm in ["d_position_bins_LR", "d_position_bins_RF"]:
        d = t[nm]["df"]
        assert list(d["bucket"]) == ["t=1", "t=2-4", "t=5-10", "t=11+"]
        for m in ["roc_auc", "pr_auc"]:
            assert all(f"{m}_{fs}" in d.columns for fs in FEATS)
    # (c) ablation：14 組比較，同時有 A 與 C 的欄（規格如此）；(f) 三張網格各 3×3
    ab = t["c_ablation"]["df"]
    assert len(ab) == 14 and {"roc_auc_mean", "pr_auc_mean", "delta_auc_mean", "n_folds_better", "t6_delta", "t6_ci_low", "t11_ci_high"} <= set(ab.columns)
    for fs in FEATS:
        g = t[f"f_rf_grid_{fs}"]["df"]
        assert list(g["max_depth"]) == ["8", "16", "None"] and g[[f"leaf{l}_is_best" for l in (5, 20, 50)]].to_numpy().sum() == 1


def test_spot_values_equal_the_long_format_files():
    t = _tables()
    lr = rd("r1_lr_cv.csv")
    a = t["a_main_A"]["df"].set_index(["model_type", "features"])
    src = lr[(lr["model"] == "R1") & (lr["eval"] == "A") & (lr["metric"] == "roc_auc")].iloc[0]
    assert a.at[("LR", "R1"), "roc_auc_mean"] == src["value"] and a.at[("LR", "R1"), "roc_auc_sd"] == src["std"]
    c = t["a_main_C"]["df"].set_index(["model_type", "features"])
    src = lr[(lr["model"] == "R1") & (lr["eval"] == "C") & (lr["group"] == "t=6") & (lr["metric"] == "roc_auc")].iloc[0]
    assert (c.at[("LR", "R1"), "t6_roc_auc"], c.at[("LR", "R1"), "t6_roc_auc_ci_low"], c.at[("LR", "R1"), "t6_roc_auc_ci_high"]) == (src["value"], src["ci_low"], src["ci_high"])
    e = t["e_cross_school"]["df"]
    assert list(e["split"]) == ["student_split", "school_group", "student_split", "student_split", "school_group", "student_split", "school_group"]
    # 各 fold 的絕對 AUC：LR 的平均要等於 r1_lr_cv.csv 的 A 平均
    g = t["g_fold_auc"]["df"].set_index(["model_type", "features"])
    for fs in FEATS:
        folds = [g.at[("LR", fs), f"roc_fold{k}"] for k in range(5)]
        ref = lr[(lr["model"] == fs) & (lr["eval"] == "A") & (lr["metric"] == "roc_auc")]["value"].iloc[0]
        assert abs(np.mean(folds) - ref) < 1e-9


if __name__ == "__main__":
    tests = [test_all_expected_tables_exist_and_match_fresh_build, test_every_number_is_found_in_source_files,
             test_numeric_columns_are_not_strings_and_string_columns_are_text, test_formatted_strings_agree_with_numeric_columns,
             test_a_b_c_are_separate_tables_with_expected_columns, test_spot_values_equal_the_long_format_files]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
