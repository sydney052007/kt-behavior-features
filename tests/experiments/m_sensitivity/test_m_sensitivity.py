"""m 敏感度實驗的測試。可直接用 python3 tests/experiments/m_sensitivity/test_m_sensitivity.py 跑。

(1) 判準函式的邊界；(2) m_map 傳入後 item_diff 的公式（假資料手算）；
(3) 不傳 m_map（預設）與明確傳 ITEM_M 的特徵完全相同（逐位），且改 m 只會改 item_diff 那一欄；
(4) 彙總表的差值與 verdict（假的 AUC）。全程只用 trainval 列。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from features_r1 import ITEM_M, item_diff_encode, load_r1_frame, make_fold_features  # noqa: E402
from run_m_sensitivity import (ADOPTED, CELLS, SETTINGS, build_tables, cell_verdict,  # noqa: E402  # import 不讀資料、不寫檔
                               overall_verdict)


def test_verdict_boundaries():
    assert cell_verdict(0.00199, 5) == "不敏感"
    assert cell_verdict(0.002, 5) == "敏感"            # 剛好 0.002：判準是「< 0.002」，所以算敏感
    assert cell_verdict(0.001, 4) == "敏感"            # 剛好 4/5 fold：要 5/5 才算
    assert cell_verdict(0.001, 5) == "不敏感"
    assert cell_verdict(0.0, 0) == "敏感"
    assert overall_verdict(["不敏感"] * 4) == "不敏感"
    assert overall_verdict(["不敏感"] * 3 + ["敏感"]) == "敏感"


def test_item_diff_formula_with_m_map():
    toy = pd.DataFrame({"skill_id": ["47"] * 4 + ["311"] * 4,
                        "problem_id": [1, 1, 1, 2, 10, 10, 10, 10],
                        "correct": [1, 1, 0, 0, 1, 0, 0, 0]})
    rows = pd.DataFrame({"skill_id": ["47", "47", "311", "311"], "problem_id": [1, 99, 10, 98]})
    m_map = {"47": 8, "311": 3}                         # 與採用值不同
    got = item_diff_encode(toy, rows, m_map).to_numpy()
    p47, p311 = 2 / 4, 1 / 4
    expect = [(2 + 8 * p47) / (3 + 8), p47, (1 + 3 * p311) / (4 + 3), p311]   # 手算：(k + m·p̄)/(n + m)
    assert np.allclose(got, expect, atol=1e-12)
    # 沒傳 m_map → 用 ITEM_M；傳 ITEM_M 的副本結果相同；傳別的 m 結果不同
    assert item_diff_encode(toy, rows).equals(item_diff_encode(toy, rows, dict(ITEM_M)))
    assert not np.allclose(got, item_diff_encode(toy, rows).to_numpy())


def test_default_m_map_unchanged_and_only_item_diff_moves():
    tv = load_r1_frame()
    tv = tv[tv["role"] == "trainval"].reset_index(drop=True)
    train, val = tv[tv["fold"] != 0], tv[tv["fold"] == 0]
    a_tr, a_va, a_st = make_fold_features(train, val)
    b_tr, b_va, b_st = make_fold_features(train, val, dict(ITEM_M))
    assert a_tr.equals(b_tr) and a_va.equals(b_va) and a_st["fills"] == b_st["fills"]     # 預設 = 明確傳 ITEM_M（逐位相同）
    c_tr, c_va, _ = make_fold_features(train, val, {"47": 4, "311": 20})
    for x, y in [(a_tr, c_tr), (a_va, c_va)]:
        assert x.drop(columns="item_diff").equals(y.drop(columns="item_diff"))            # 只有 item_diff 變
        s47 = (x["skill_311"] == 0).to_numpy()
        assert not np.allclose(x.loc[s47, "item_diff"], y.loc[s47, "item_diff"])         # 技能 47 變了
        assert x.loc[~s47, "item_diff"].equals(y.loc[~s47, "item_diff"])                 # 技能 311 沒動


def test_tables_deltas_and_verdict():
    base = {(mt, fs): pd.Series([0.60 + 0.01 * f + (0.03 if fs == "R1" else 0) for f in range(5)])
            for mt, fs in CELLS}
    all_auc = {name: {c: s.copy() for c, s in base.items()} for name, _, _ in SETTINGS}
    cv, vt = build_tables(all_auc)
    assert (cv["delta_vs_adopted"].abs() < 1e-15).all()                       # 全部相同 → 差是 0
    assert (vt[vt["model_type"] != "all"]["verdict"] == "不敏感").all() and vt.iloc[-1]["verdict"] == "不敏感"
    assert set(cv["setting"]) == {s[0] for s in SETTINGS} and len(cv) == 5 * 2 * 3 * 6
    # 把 m47=4 下 RF 的 R1 加 0.01：RF-R1 的差 = 0.01 → 敏感，其他三格仍不敏感，整體敏感
    all_auc["m47=4"][("RF", "R1")] = base[("RF", "R1")] + 0.01
    cv, vt = build_tables(all_auc)
    m = cv[(cv["setting"] == "m47=4") & (cv["model_type"] == "RF") & (cv["features"] == "R1") & (cv["fold"] == "mean")]
    assert np.isclose(m["delta_vs_adopted"].iloc[0], 0.01)
    v = vt.set_index(["model_type", "features"])["verdict"]
    assert v[("RF", "R1")] == "敏感" and v[("RF", "R0d")] == "不敏感" and v[("all", "all")] == "敏感"
    # R1 − R0d 在某設定只有 4/5 fold 較好 → 該模型兩格都敏感
    all_auc = {name: {c: s.copy() for c, s in base.items()} for name, _, _ in SETTINGS}
    all_auc["m311=24"][("LR", "R1")] = pd.Series([0.60 + 0.01 * f - (0.05 if f == 0 else 0) for f in range(5)])
    _, vt = build_tables(all_auc)
    assert vt.iloc[-1]["verdict"] == "敏感"


if __name__ == "__main__":
    tests = [test_verdict_boundaries, test_item_diff_formula_with_m_map,
             test_default_m_map_unchanged_and_only_item_diff_moves, test_tables_deltas_and_verdict]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
