"""R0 特徵管線的因果性測試。

環境沒裝 pytest，所以寫成一般函式 + assert，可以用
    python3 tests/test_features.py
直接跑（檔尾會自己把 test_* 函式都呼叫一次並印 PASS/FAIL），
但函式名稱仍遵守 pytest 慣例，之後要裝 pytest 也能直接收。

三個測試都在記憶體裡用 src.features 重算，不讀 features_r0.csv，
所以測的是特徵函式本身，不是某次跑出來的檔案。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import SEQ, add_position, add_r0_features, load_main  # noqa: E402

FEAT_COLS = ["c", "w", "log1p_c", "log1p_w", "skill_311", "has_history"]


def _base_df() -> pd.DataFrame:
    """load_main + add_position，之後每個測試各自做自己要的事（例如翻轉某列）。"""
    return add_position(load_main())


def test_first_row_of_each_sequence_has_zero_counts():
    """每條序列第 1 列（t==1）：c、w 都是 0，has_history 是 0。"""
    df = add_r0_features(_base_df())
    first = df[df["t"] == 1]
    assert len(first) == df.groupby(SEQ).ngroups  # 每條序列恰好一個 t==1 的列
    assert (first["c"] == 0).all()
    assert (first["w"] == 0).all()
    assert (first["has_history"] == 0).all()


def test_order_id_strictly_increasing_within_sequence():
    """每條序列內，order_id 隨 t 嚴格遞增（涵蓋所有序列，不是只看抽樣的幾條）。

    這是 add_position／add_r0_features 的 cumsum 能算對的前提：
    如果某條序列裡 order_id 沒有嚴格遞增，t 的名次就不等於實際發生順序。
    """
    df = _base_df()
    diffs = df.groupby(SEQ)["order_id"].diff()  # 每條序列第一列是 NaN，之後是與前一列的差
    assert (diffs.dropna() > 0).all(), "有序列的 order_id 沒有隨 t 嚴格遞增"
    # NaN 的數量應該恰好等於序列數（每條序列第一列）
    assert diffs.isna().sum() == df.groupby(SEQ).ngroups


def test_counts_sum_to_t_minus_1():
    """每一列：c + w == t - 1（c、w 是各自獨立算出來的，這是真的恆等式檢查）。"""
    df = add_r0_features(_base_df())
    assert (df["c"] + df["w"] == df["t"] - 1).all()
    assert df[FEAT_COLS + ["t"]].notna().all().all()
    # 每條序列的 t 恰好是 1..L，沒有跳號或重複
    for _, g in df.groupby(SEQ):
        assert sorted(g["t"]) == list(range(1, len(g) + 1))


def test_flipping_correct_is_causal():
    """翻轉某一列的 correct：
    - 該列自己的特徵不變（特徵只看「之前」，不看自己）
    - 同序列 t 更小的列完全不變
    - 同序列 t 更大的列：c/w/log1p_c/log1p_w 改變，但 t、has_history、skill_311 不變
    - 其他序列（不同 user_id 或不同 skill_id）完全不變
    """
    base = _base_df()
    rng = np.random.default_rng(0)

    # 挑幾條序列來翻轉：固定挑一條長度 >= 10 的（確保翻轉點之後還有足夠的列可檢查），
    # 再隨機挑 3 條當補充。
    seq_lens = base.groupby(SEQ).size()
    fixed_seq = seq_lens[seq_lens >= 10].index[0]  # 固定挑一條長序列
    rest = seq_lens.index.drop(fixed_seq)
    extra_seqs = rest[rng.choice(len(rest), size=3, replace=False)]
    picked_seqs = [fixed_seq] + list(extra_seqs)

    for user_id, skill_id in picked_seqs:
        seq_mask = (base["user_id"] == user_id) & (base["skill_id"] == skill_id)
        seq_rows = base[seq_mask].sort_values("t")
        flip_t = int(seq_rows["t"].iloc[len(seq_rows) // 2])  # 翻轉序列中間那一列
        flip_order_id = int(seq_rows.loc[seq_rows["t"] == flip_t, "order_id"].iloc[0])

        before = add_r0_features(base)
        flipped_raw = base.copy()
        row_idx = flipped_raw.index[flipped_raw["order_id"] == flip_order_id][0]
        flipped_raw.loc[row_idx, "correct"] = 1 - flipped_raw.loc[row_idx, "correct"]
        after = add_r0_features(flipped_raw)

        b = before.set_index("order_id")
        a = after.set_index("order_id")

        # 自己這一列：特徵不變（correct 欄位本身當然變了，但不是「特徵」）
        own_before, own_after = b.loc[flip_order_id], a.loc[flip_order_id]
        for col in FEAT_COLS:
            assert own_before[col] == own_after[col], f"自己這一列的 {col} 不該因為翻轉自己的 correct 而變"

        # 同序列、t 更小的列：完全不變
        earlier_order_ids = seq_rows.loc[seq_rows["t"] < flip_t, "order_id"]
        for oid in earlier_order_ids:
            for col in FEAT_COLS:
                assert b.loc[oid, col] == a.loc[oid, col], f"order_id={oid} 的 {col} 不該被之後翻轉的列影響"

        # 同序列、t 更大的列：c/w/log1p_c/log1p_w 要變，t/has_history/skill_311 不變
        later_order_ids = seq_rows.loc[seq_rows["t"] > flip_t, "order_id"]
        assert len(later_order_ids) > 0, "挑的翻轉點要保留至少一列在之後，才能驗證『之後列會變』"
        for oid in later_order_ids:
            delta_c = a.loc[oid, "c"] - b.loc[oid, "c"]
            delta_w = a.loc[oid, "w"] - b.loc[oid, "w"]
            assert {delta_c, delta_w} == {1, -1} or {delta_c, delta_w} == {-1, 1}, (
                f"order_id={oid} 的 c/w 應該剛好一個 +1、一個 -1（因為翻轉了之前的一次答對/答錯）"
            )
            assert b.loc[oid, "t"] == a.loc[oid, "t"]
            assert b.loc[oid, "has_history"] == a.loc[oid, "has_history"]
            assert b.loc[oid, "skill_311"] == a.loc[oid, "skill_311"]

        # 其他序列：完全不變
        other_order_ids = base.loc[~seq_mask, "order_id"]
        sample_other = other_order_ids.sample(n=min(500, len(other_order_ids)), random_state=0)
        for oid in sample_other:
            for col in FEAT_COLS:
                assert b.loc[oid, col] == a.loc[oid, col], f"其他序列的 order_id={oid} 不該被這次翻轉影響"


if __name__ == "__main__":
    tests = [test_first_row_of_each_sequence_has_zero_counts,
             test_order_id_strictly_increasing_within_sequence,
             test_counts_sum_to_t_minus_1,
             test_flipping_correct_is_causal]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
