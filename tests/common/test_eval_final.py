"""eval_final.py（test 最終評估用的評估函式）的測試。可直接用 python3 tests/common/test_eval_final.py 跑（假資料，不讀 data/）。"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from eval_final import CANNOT_CONFIRM, eval_all_rows, interpret_delta, paired_delta_all  # noqa: E402
from evaluate import eval_c, paired_delta_c  # noqa: E402


def _toy(n_users=80, per_user=10, seed=0):
    """每位學生 per_user 列；回傳 (舊模型 OOF, 新模型 OOF)。"""
    rng = np.random.default_rng(seed)
    uid = np.repeat(np.arange(n_users), per_user)
    correct = rng.integers(0, 2, len(uid))
    base = pd.DataFrame({"user_id": uid, "order_id": np.arange(len(uid)), "t": np.tile(np.arange(1, per_user + 1), n_users),
                         "fold": 0, "correct": correct})
    old = base.assign(p_correct=np.clip(0.3 + 0.2 * correct + rng.random(len(uid)) * 0.5, 0.01, 0.99))
    new = base.assign(p_correct=np.clip(0.2 + 0.4 * correct + rng.random(len(uid)) * 0.4, 0.01, 0.99))
    return old, new


def test_eval_all_rows_matches_sklearn_and_shares_bootstrap_with_eval_c():
    old, _ = _toy()
    rows = {r["metric"]: r for r in eval_all_rows(old, n_boot=100, seed=7)}
    y, s = 1 - old["correct"].to_numpy(), 1 - old["p_correct"].to_numpy()          # 答錯為正類
    assert rows["roc_auc"]["value"] == roc_auc_score(y, s)                         # 逐位相符
    assert rows["pr_auc"]["value"] == average_precision_score(y, s)
    for r in rows.values():
        assert r["n_rows"] == len(old) and r["n_students"] == old["user_id"].nunique()
        assert r["ci_low"] <= r["value"] <= r["ci_high"]
    # 重抽程序與 C 相同：把所有列的 t 設成同一個切點，eval_c 的區間必須與 eval_all_rows 完全一樣
    c = {r["metric"]: r for r in eval_c(old.assign(t=1), cut_points=(1,), n_boot=100, seed=7)}
    for m in ("roc_auc", "pr_auc"):
        assert (c[m]["value"], c[m]["ci_low"], c[m]["ci_high"]) == (rows[m]["value"], rows[m]["ci_low"], rows[m]["ci_high"])
    again = {r["metric"]: r for r in eval_all_rows(old, n_boot=100, seed=7)}
    assert again == rows                                                            # 固定種子重跑相同


def test_paired_delta_all_matches_sklearn_and_paired_delta_c():
    old, new = _toy()
    out = paired_delta_all(old, new, n_boot=100, seed=7)
    assert len(out) == 1 and out["group"].iloc[0] == "all_rows" and out["n_rows"].iloc[0] == len(old)
    y = 1 - old["correct"].to_numpy()
    exp = roc_auc_score(y, 1 - new["p_correct"].to_numpy()) - roc_auc_score(y, 1 - old["p_correct"].to_numpy())
    assert out["delta_auc"].iloc[0] == exp                                          # 點估計與 sklearn 逐位相符
    assert out["ci_low"].iloc[0] < exp < out["ci_high"].iloc[0]
    # 與 C 用同一個配對重抽：若只取 t == 6 的列，paired_delta_c 與（把那些列當全部列的）paired_delta_all 相同
    sub_o, sub_n = old[old["t"] == 6], new[new["t"] == 6]
    c = paired_delta_c(old, new, cut_points=(6,), n_boot=100, seed=7).iloc[0]
    a = paired_delta_all(sub_o, sub_n, n_boot=100, seed=7).iloc[0]
    assert (a["delta_auc"], a["ci_low"], a["ci_high"]) == (c["delta_auc"], c["ci_low"], c["ci_high"])
    # 兩個一樣的模型：Δ 與區間都是 0
    z = paired_delta_all(old, old.copy(), n_boot=50, seed=3).iloc[0]
    assert z["delta_auc"] == 0 and z["ci_low"] == 0 and z["ci_high"] == 0


def test_interpret_delta_boundaries():
    assert interpret_delta(0.0, 0.05) == CANNOT_CONFIRM            # 下界剛好 0：含 0
    assert interpret_delta(-0.05, 0.0) == CANNOT_CONFIRM           # 上界剛好 0：含 0
    assert interpret_delta(-0.01, 0.01) == CANNOT_CONFIRM
    assert interpret_delta(1e-12, 0.05) == "改善"
    assert interpret_delta(-0.05, -1e-12) == "變差"
    assert interpret_delta(np.nan, 0.05) == CANNOT_CONFIRM and interpret_delta(0.01, np.nan) == CANNOT_CONFIRM


if __name__ == "__main__":
    tests = [test_eval_all_rows_matches_sklearn_and_shares_bootstrap_with_eval_c,
             test_paired_delta_all_matches_sklearn_and_paired_delta_c, test_interpret_delta_boundaries]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
