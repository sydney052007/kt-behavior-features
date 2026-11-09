"""R1r 新特徵的測試。可直接用 python3 tests/test_features_r1r.py 跑。

用真實資料 trainval 的 fold 0（訓練 = fold != 0，驗證 = fold == 0）；全程不碰 test 列。
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features_r1 import ITEM_M, R1_FEATURES, inner_groups, load_r1_frame, make_fold_features  # noqa: E402
from features_r1r import (R1R_FEATURES, REL_FEATURES, item_stats_encode,  # noqa: E402
                          item_stats_train_oof, make_fold_features_r1r)

_cache = {}


def _tv():
    if "tv" not in _cache:
        df = load_r1_frame()
        assert (df["role"] == "test").sum() > 0
        _cache["all"], _cache["tv"] = df, df[df["role"] == "trainval"].reset_index(drop=True)
    return _cache["tv"]


def _fold(k):
    if k not in _cache:
        tv = _tv()
        train, val = tv[tv["fold"] != k], tv[tv["fold"] == k]
        _cache[k] = (train, val) + make_fold_features_r1r(train, val)
    return _cache[k]


def _hint(df):
    return (df["hint_count"] > 0).astype(float)


def _logatt(df):
    return np.log1p(df["attempt_count"].astype(float))


# ------------------------------------------------------------------ 1. 慢算法

def test_slow_loop_matches_vectorized():
    train, val, tf, vf, st = _fold(0)
    grp = pd.Series(inner_groups(train), index=train.index)
    feat = pd.concat([tf, vf]).set_index("order_id")
    raw = pd.concat([train.assign(g=grp), val.assign(g=-1)]).set_index("order_id")   # g = inner 組別；驗證列 -1

    # 參考集的逐題總和／筆數，用最直接的字典迴圈累計（獨立於向量化寫法）
    def accumulate(rows):
        tot, cnt, sk_sum, sk_n = {}, {}, {}, {}
        for r in rows.itertuples():
            for name, v in (("h", float(r.hint_count > 0)), ("a", math.log1p(r.attempt_count))):
                key = (name, r.skill_id, r.problem_id)
                tot[key] = tot.get(key, 0.0) + v
                cnt[key] = cnt.get(key, 0) + 1
                sk_sum[(name, r.skill_id)] = sk_sum.get((name, r.skill_id), 0.0) + v
                sk_n[(name, r.skill_id)] = sk_n.get((name, r.skill_id), 0) + 1
        return tot, cnt, sk_sum, sk_n

    full = accumulate(train)
    per_group = {g: accumulate(train[grp == g]) for g in range(5)}

    def item_value(name, skill, problem, g):
        """題目 problem 的平滑平均；g = 該列所屬 inner 組（驗證列用整個訓練集）。"""
        tot, cnt, ss, sn = full
        s, n = tot.get((name, skill, problem), 0.0), cnt.get((name, skill, problem), 0)
        ssum, sc = ss[(name, skill)], sn[(name, skill)]
        if g >= 0:                                     # 扣掉自己這一組
            gt, gc, gss, gsn = per_group[g]
            s -= gt.get((name, skill, problem), 0.0)
            n -= gc.get((name, skill, problem), 0)
            ssum -= gss.get((name, skill), 0.0)
            sc -= gsn.get((name, skill), 0)
        m = ITEM_M[skill]
        return (s + m * (ssum / sc)) / (n + m)

    def rel_rows(recs):
        return [(float(r["hint_count"] > 0) - item_value("h", r["skill_id"], r["problem_id"], r["g"]),
                 math.log1p(r["attempt_count"]) - item_value("a", r["skill_id"], r["problem_id"], r["g"])) for r in recs]

    # 第 1 題填補值 = 訓練列 rel_*_row 的平均（慢算法）
    tr_recs = train.assign(g=grp).to_dict("records")
    tr_rel = rel_rows(tr_recs)
    fill_h = sum(x[0] for x in tr_rel) / len(tr_rel)
    fill_a = sum(x[1] for x in tr_rel) / len(tr_rel)

    seqs = raw.groupby(["user_id", "skill_id"]).size().index.to_list()
    rng = np.random.default_rng(0)
    picked = [seqs[i] for i in rng.choice(len(seqs), size=300, replace=False)]
    n_rows = n_bad = 0
    for uid, sk in picked:
        recs = raw[(raw["user_id"] == uid) & (raw["skill_id"] == sk)].sort_index().reset_index().to_dict("records")
        rel = rel_rows(recs)
        for i, r in enumerate(recs):
            if i == 0:
                exp = (fill_h, fill_h, fill_a, fill_a)
            else:
                exp = (sum(x[0] for x in rel[:i]) / i, rel[i - 1][0], sum(x[1] for x in rel[:i]) / i, rel[i - 1][1])
            n_rows += 1
            got = [feat.at[r["order_id"], c] for c in ["rel_hint_mean", "rel_hint_last", "rel_att_mean", "rel_att_last"]]
            if not np.allclose(got, exp, atol=1e-9):
                n_bad += 1
    assert n_rows > 1000 and n_bad == 0, f"{n_bad}/{n_rows} 列與慢算法不一致"


# ------------------------------------------------------------------ 2. 不含自己／同組

def test_own_group_excluded_from_item_stats():
    train, val, tf, vf, st = _fold(0)
    grp = inner_groups(train)
    base = item_stats_train_oof(train)
    rng = np.random.default_rng(3)
    other_changed = 0
    for pos in rng.choice(len(train), size=15, replace=False):
        idx = train.index[pos]
        mod = train.copy()
        mod.at[idx, "hint_count"] = 7 if mod.at[idx, "hint_count"] == 0 else 0
        mod.at[idx, "attempt_count"] = mod.at[idx, "attempt_count"] + 9
        new = item_stats_train_oof(mod)
        assert (new.loc[idx] == base.loc[idx]).all(), "自己的 item_hint／item_att 變了"
        same = grp == grp[pos]
        assert np.array_equal(new[same].to_numpy(), base[same].to_numpy()), "同一 inner 組的其他列變了"
        other_changed += int((new[~same] != base[~same]).any(axis=1).sum())
    assert other_changed > 0                 # 別組同題的列確實會變（確認不是空轉）
    # 驗證列：改驗證列自己的行為，驗證列的題目統計不變（只用訓練集）
    v_base = item_stats_encode(train, val)
    v_mod = val.copy()
    v_mod["hint_count"], v_mod["attempt_count"] = 5, 11
    assert item_stats_encode(train, v_mod).equals(v_base)


# ------------------------------------------------------------------ 3. 無 test、統計量只來自訓練列

def test_no_test_rows_and_stats_from_train_only():
    train, val, tf, vf, st = _fold(0)
    full = _cache["all"]
    try:
        make_fold_features_r1r(full[full["role"] == "trainval"].query("fold != 0"), full[full["role"] == "test"].head(50))
    except AssertionError:
        pass
    else:
        raise AssertionError("應該拒絕 test 列")
    test_orders = set(full.loc[full["role"] == "test", "order_id"])
    for k in range(5):
        _, _, tfk, vfk, _ = _fold(k)
        assert test_orders.isdisjoint(tfk["order_id"]) and test_orders.isdisjoint(vfk["order_id"])
    # 填補值只由訓練列決定：把驗證列的行為欄位全改掉，填補值（與訓練列特徵）不變
    val_mod = val.copy()
    val_mod["hint_count"], val_mod["attempt_count"] = 3, 9
    tf2, _, st2 = make_fold_features_r1r(train, val_mod)
    assert st2["rel_fills"] == st["rel_fills"]
    assert tf2[REL_FEATURES].equals(tf[REL_FEATURES])
    # 對照：真的把驗證列併進訓練集重算，填補值會不同（確認上面的『不變』不是空轉）
    both = pd.concat([train, val]).sort_index()
    items_both = item_stats_train_oof(both)
    fill_both = float((_hint(both) - items_both["item_hint"]).mean())
    assert fill_both != st["rel_fills"]["rel_hint"]


# ------------------------------------------------------------------ 4. 沒見過的題目 = 技能平均

def test_unseen_item_is_skill_mean():
    ref = pd.DataFrame({
        "skill_id": ["47"] * 4 + ["311"] * 4,
        "problem_id": [1, 1, 1, 2, 10, 10, 10, 10],
        "hint_count": [0, 2, 0, 0, 1, 0, 0, 0],
        "attempt_count": [1, 3, 1, 2, 4, 1, 1, 1],
    })
    rows = pd.DataFrame({"skill_id": ["47", "47", "311", "311"], "problem_id": [1, 99, 10, 98]})
    got = item_stats_encode(ref, rows)
    h47, h311 = 1 / 4, 1 / 4                                         # 技能內 hint_flag 平均
    a47 = np.mean(np.log1p([1, 3, 1, 2])); a311 = np.mean(np.log1p([4, 1, 1, 1]))
    assert np.isclose(got.at[1, "item_hint"], h47) and np.isclose(got.at[3, "item_hint"], h311)   # 沒見過 = 技能平均
    assert np.isclose(got.at[1, "item_att"], a47) and np.isclose(got.at[3, "item_att"], a311)
    # 見過的題目：題目 1（47）：Σ hint = 1，n = 3，m = 5
    assert np.isclose(got.at[0, "item_hint"], (1 + 5 * h47) / (3 + 5))
    assert np.isclose(got.at[2, "item_att"], (np.log1p(4) + 3 * np.log1p(1) + 20 * a311) / (4 + 20))


# ------------------------------------------------------------------ 5. 第 1 題 = 訓練列平均

def test_first_row_is_train_mean_of_rel_rows():
    for k in range(5):
        train, val, tf, vf, st = _fold(k)
        sig_h, sig_a = _hint(train), _logatt(train)
        items = item_stats_train_oof(train)
        fh = (sig_h - items["item_hint"]).mean()
        fa = (sig_a - items["item_att"]).mean()
        assert np.isclose(st["rel_fills"]["rel_hint"], fh) and np.isclose(st["rel_fills"]["rel_att"], fa)
        for feat in (tf, vf):
            first = feat[feat["t"] == 1]
            assert len(first) > 0
            assert np.allclose(first[["rel_hint_mean", "rel_hint_last"]], fh)
            assert np.allclose(first[["rel_att_mean", "rel_att_last"]], fa)


# ------------------------------------------------------------------ 6. 沒有 NaN／inf

def test_no_nan_or_inf():
    for k in range(5):
        _, _, tf, vf, _ = _fold(k)
        for feat in (tf, vf):
            assert np.isfinite(feat[R1R_FEATURES].to_numpy(float)).all(), f"fold {k} 有 NaN 或 inf"
    assert len(R1R_FEATURES) == 17 and R1R_FEATURES == R1_FEATURES + REL_FEATURES


# ------------------------------------------------------------------ 7. R1 既有 13 個特徵完全相同

def test_existing_r1_features_unchanged():
    tv = _tv()
    for k in range(5):
        train, val, tf, vf, _ = _fold(k)
        tf0, vf0, _ = make_fold_features(train, val)
        assert tf[R1_FEATURES].equals(tf0[R1_FEATURES]) and vf[R1_FEATURES].equals(vf0[R1_FEATURES]), f"fold {k}"
        assert tf.index.equals(train.index) and vf.index.equals(val.index)


if __name__ == "__main__":
    tests = [test_slow_loop_matches_vectorized, test_own_group_excluded_from_item_stats,
             test_no_test_rows_and_stats_from_train_only, test_unseen_item_is_skill_mean,
             test_first_row_is_train_mean_of_rel_rows, test_no_nan_or_inf, test_existing_r1_features_unchanged]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
