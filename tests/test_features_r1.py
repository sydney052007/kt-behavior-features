"""R0d／R1 特徵的測試。可直接用 python3 tests/test_features_r1.py 跑（不需要 pytest）。

大部分測試用真實資料的 fold 0（訓練 = trainval 中 fold != 0，驗證 = fold == 0），
並且全程只用 role == trainval 的列。
"""
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import features_r1 as fr  # noqa: E402
from features_r1 import (BEHAVIOR_FEATURES, ITEM_M, R0D_FEATURES, R1_FEATURES,  # noqa: E402
                         fit_stats, inner_groups, item_diff_encode, item_diff_train_oof,
                         load_r1_frame, make_fold_features, transform)

_cache = {}


def _tv() -> pd.DataFrame:
    if "tv" not in _cache:
        df = load_r1_frame()
        assert (df["role"] == "test").sum() > 0       # 資料裡確實有 test 列
        _cache["tv"] = df[df["role"] == "trainval"].reset_index(drop=True)
    return _cache["tv"]


def _fold(k: int):
    """第 k 個 fold 的 (train, val, train_feat, val_feat, stats)，結果快取。"""
    if k not in _cache:
        tv = _tv()
        train, val = tv[tv["fold"] != k], tv[tv["fold"] == k]
        tf, vf, st = make_fold_features(train, val)
        _cache[k] = (train, val, tf, vf, st)
    return _cache[k]


def _scalar_std_time(ms, problem, stats):
    """逐列慢算：標準化作答時間；缺失回傳 None。"""
    if ms <= 0:
        return None
    med = stats["item_log_median"].get(problem, np.nan)
    if math.isnan(med):
        return None
    return math.log(min(max(ms, stats["p1"]), stats["p99"])) - med


# ------------------------------------------------------------------ 1. 慢算法重算

def test_slow_loop_matches_vectorized():
    train, val, tf, vf, st = _fold(0)
    allraw = pd.concat([train, val]).set_index("order_id")
    feat = pd.concat([tf, vf]).set_index("order_id")
    fills = st["fills"]

    seqs = allraw.groupby(["user_id", "skill_id"]).size().index.to_list()
    rng = np.random.default_rng(0)
    picked = [seqs[i] for i in rng.choice(len(seqs), size=300, replace=False)]

    n_rows = n_bad = 0
    for uid, sk in picked:
        rows = allraw[(allraw["user_id"] == uid) & (allraw["skill_id"] == sk)].sort_index()
        recs = rows.reset_index().to_dict("records")       # 依 order_id 排序的 dict 列表
        for i, r in enumerate(recs):
            hist = recs[:i]                                # 第 1…t-1 列（雙層迴圈：這裡逐一掃）
            if hist:
                hint_rate = sum(h["hint_count"] > 0 for h in hist) / len(hist)
                last_hint = float(hist[-1]["hint_count"] > 0)
                times = []
                for h in hist:
                    v = _scalar_std_time(h["ms_first_response"], h["problem_id"], st)
                    times.append(fills["log_time"] if v is None else v)   # 缺失填訓練集平均
                mean_log_time = sum(times) / len(times)
                last_log_att = math.log1p(hist[-1]["attempt_count"])
                scaffold_rate = sum(h["first_action"] == 2 for h in hist) / len(hist)
                last_correct = float(hist[-1]["correct"])
            else:
                hint_rate = last_hint = fills["hint"]
                mean_log_time = fills["log_time"]
                last_log_att = fills["log_att"]
                scaffold_rate = fills["scaffold"]
                last_correct = fills["last_correct"]
            expect = dict(hint_rate=hint_rate, last_hint=last_hint, mean_log_time=mean_log_time,
                          last_log_att=last_log_att, scaffold_rate=scaffold_rate,
                          last_correct=last_correct)
            n_rows += 1
            if not all(np.isclose(feat.loc[r["order_id"], k], v, atol=1e-9) for k, v in expect.items()):
                n_bad += 1
    assert n_rows > 1000 and n_bad == 0, f"{n_bad}/{n_rows} 列與慢算法不一致"


# ------------------------------------------------------------------ 2. 因果性

def _perturb(df, order_ids, rng):
    """把指定列自己的行為欄位與 correct 改成亂數。"""
    d = df.copy()
    m = d["order_id"].isin(order_ids)
    n = int(m.sum())
    d.loc[m, "hint_count"] = rng.integers(0, 5, n)
    d.loc[m, "first_action"] = rng.integers(0, 3, n)
    d.loc[m, "attempt_count"] = rng.integers(1, 10, n)
    d.loc[m, "ms_first_response"] = rng.integers(500, 60000, n)
    d.loc[m, "correct"] = rng.integers(0, 2, n)
    return d


def test_own_behavior_never_enters_own_features_and_flows_to_next_row():
    train, val, tf, vf, st = _fold(0)
    rng = np.random.default_rng(1)
    n_next_checked = 0
    for name, part, base_feat in [("val", val, vf), ("train", train, tf)]:
        nxt = part.groupby(["user_id", "skill_id"])["order_id"].shift(-1)
        cand = part.index[nxt.notna() & (part["t"] >= 2)]
        for idx in rng.choice(cand, size=30, replace=False):
            oid = part.at[idx, "order_id"]
            nxt_oid = int(nxt.at[idx])
            mod = _perturb(part, [oid], rng)
            if name == "val":
                mod_feat = transform(mod, st, item_diff_encode(train, mod))
            else:
                mod_feat = transform(mod, st, item_diff_train_oof(mod))
            a = base_feat.set_index("order_id")
            b = mod_feat.set_index("order_id")
            # 該列自己的所有特徵不變（包含 item_diff）
            assert np.allclose(a.loc[oid, R1_FEATURES].astype(float),
                               b.loc[oid, R1_FEATURES].astype(float), atol=1e-12), f"{name}：自己的特徵變了"
            # 更早的列（t-1）不變
            prev = part[(part["user_id"] == part.at[idx, "user_id"]) & (part["skill_id"] == part.at[idx, "skill_id"])
                        & (part["t"] == part.at[idx, "t"] - 1)]["order_id"].iloc[0]
            assert np.allclose(a.loc[prev, R1_FEATURES].astype(float),
                               b.loc[prev, R1_FEATURES].astype(float), atol=1e-12)
            # 下一列：該欄位「本來有影響」的特徵要跟著變
            if name == "val":
                old = fr._row_signals(part.loc[[idx]]).iloc[0]
                new = fr._row_signals(mod[mod["order_id"] == oid]).iloc[0]
                old_t = fr._std_time(part.loc[[idx]], st["p1"], st["p99"], st["item_log_median"]).fillna(st["fills"]["log_time"]).iloc[0]
                new_t = fr._std_time(mod[mod["order_id"] == oid], st["p1"], st["p99"], st["item_log_median"]).fillna(st["fills"]["log_time"]).iloc[0]
                changed = {
                    "last_correct": part.at[idx, "correct"] != mod.loc[mod["order_id"] == oid, "correct"].iloc[0],
                    "last_hint": old["hint_flag"] != new["hint_flag"],
                    "hint_rate": old["hint_flag"] != new["hint_flag"],
                    "scaffold_rate": old["scaffold_flag"] != new["scaffold_flag"],
                    "last_log_att": old["log_att"] != new["log_att"],
                    "mean_log_att": old["log_att"] != new["log_att"],
                    "last_log_time": old_t != new_t,
                    "mean_log_time": old_t != new_t,
                }
                for feat_name, should_change in changed.items():
                    if should_change:
                        n_next_checked += 1
                        assert a.loc[nxt_oid, feat_name] != b.loc[nxt_oid, feat_name], \
                            f"{feat_name}：改了第 t 列，第 t+1 列沒跟著變"
    assert n_next_checked > 100      # 確認「會影響下一列」的情況真的有被檢查到


# ------------------------------------------------------------------ 3. item_diff 不含自己

def test_item_diff_excludes_own_group():
    train, val, tf, vf, st = _fold(0)
    grp = inner_groups(train)
    base = item_diff_train_oof(train)
    rng = np.random.default_rng(2)
    other_group_changed = 0
    for pos in rng.choice(len(train), size=15, replace=False):
        idx = train.index[pos]
        mod = train.copy()
        mod.at[idx, "correct"] = 1 - mod.at[idx, "correct"]
        new = item_diff_train_oof(mod)
        assert new.at[idx] == base.at[idx], "自己的 item_diff 變了"
        same_group = grp == grp[pos]
        assert np.array_equal(new[same_group].to_numpy(), base[same_group].to_numpy()), "同一 inner 組的列變了"
        other_group_changed += int((new[~same_group] != base[~same_group]).sum())
    assert other_group_changed > 0          # 別組同題的列確實會變（確認測試不是空轉）

    # 驗證列：翻轉驗證列自己的 correct，整批驗證列的 item_diff 都不變（只用訓練集）
    v_base = item_diff_encode(train, val)
    v_mod = val.copy()
    flip = rng.choice(v_mod.index, size=200, replace=False)
    v_mod.loc[flip, "correct"] = 1 - v_mod.loc[flip, "correct"]
    assert np.array_equal(item_diff_encode(train, v_mod).to_numpy(), v_base.to_numpy())


# ------------------------------------------------------------------ 4. 不洩漏 test

def test_no_test_rows_and_stats_from_train_only():
    train, val, tf, vf, st = _fold(0)
    # 管線輸入若混進 test 列要報錯
    full = load_r1_frame()
    try:
        make_fold_features(full[full["role"] == "trainval"].query("fold != 0"), full[full["role"] == "test"].head(50))
    except AssertionError:
        pass
    else:
        raise AssertionError("make_fold_features 應該拒絕 test 列")
    test_ids = set(full.loc[full["role"] == "test", "order_id"])
    for k in range(5):
        _, _, tfk, vfk, _ = _fold(k)
        assert test_ids.isdisjoint(tfk["order_id"]) and test_ids.isdisjoint(vfk["order_id"])

    # 統計量用另一種寫法獨立重算，只用訓練列
    ms = train["ms_first_response"].to_numpy(float)
    p1 = np.percentile(ms[ms > 0], 1)
    p99 = np.percentile(ms[ms > 0], 99)
    assert st["p1"] == p1 and st["p99"] == p99
    logs = {}
    for p, m_ in zip(train["problem_id"], ms):
        if m_ > 0:
            logs.setdefault(p, []).append(math.log(min(max(m_, p1), p99)))
    for p, v in logs.items():
        assert abs(st["item_log_median"][p] - np.median(v)) < 1e-12
    assert abs(st["fills"]["last_correct"] - train["correct"].mean()) < 1e-12
    assert abs(st["fills"]["hint"] - (train["hint_count"] > 0).mean()) < 1e-12
    assert abs(st["fills"]["log_att"] - np.log1p(train["attempt_count"]).mean()) < 1e-12
    # 加進驗證列會讓統計量不同 → 證明驗證列沒有被算進去
    both = fit_stats(pd.concat([train, val]))
    assert both["p1"] != st["p1"] or both["p99"] != st["p99"] or both["fills"] != st["fills"]


# ------------------------------------------------------------------ 5. m 常數與沒見過的題目

def test_item_m_constants_and_unseen_item():
    assert ITEM_M == {"47": 5, "311": 20}
    toy = pd.DataFrame({
        "skill_id": ["47"] * 4 + ["311"] * 4,
        "problem_id": [1, 1, 1, 2, 10, 10, 10, 10],
        "correct": [1, 1, 0, 0, 1, 0, 0, 0],
    })
    rows = pd.DataFrame({"skill_id": ["47", "47", "47", "311", "311"],
                         "problem_id": [1, 2, 99, 10, 98]})
    got = item_diff_encode(toy, rows).to_numpy()
    p47, p311 = 2 / 4, 1 / 4
    expect = [(2 + 5 * p47) / (3 + 5), (0 + 5 * p47) / (1 + 5), p47,
              (1 + 20 * p311) / (4 + 20), p311]
    assert np.allclose(got, expect, atol=1e-12)
    assert got[2] == p47 and got[4] == p311      # 沒見過的題目 = 該技能 p̄


# ------------------------------------------------------------------ 6. 第 1 題

def test_first_row_features_are_zero_or_train_mean():
    for k in range(5):
        _, _, tf, vf, st = _fold(k)
        f = st["fills"]
        for feat in (tf, vf):
            first = feat[feat["t"] == 1]
            assert len(first) > 0 and (first["has_history"] == 0).all()
            assert (first["log1p_c"] == 0).all() and (first["log1p_w"] == 0).all()
            for col, key in [("last_correct", "last_correct"), ("hint_rate", "hint"), ("last_hint", "hint"),
                             ("mean_log_time", "log_time"), ("last_log_time", "log_time"),
                             ("mean_log_att", "log_att"), ("last_log_att", "log_att"),
                             ("scaffold_rate", "scaffold")]:
                assert np.allclose(first[col], f[key], atol=1e-12), f"fold {k} {col}"
            assert first[R1_FEATURES].notna().all().all()


# ------------------------------------------------------------------ 7. 沒有 NaN / inf

def test_no_nan_or_inf():
    for k in range(5):
        _, _, tf, vf, _ = _fold(k)
        for feat in (tf, vf):
            x = feat[R1_FEATURES].to_numpy(float)
            assert np.isfinite(x).all(), f"fold {k} 有 NaN 或 inf"
            assert feat["item_diff"].between(0, 1).all()
    assert R0D_FEATURES == fr.R0_FEATURES + ["item_diff", "last_correct"]
    assert R1_FEATURES == R0D_FEATURES + BEHAVIOR_FEATURES and len(BEHAVIOR_FEATURES) == 7


if __name__ == "__main__":
    tests = [test_slow_loop_matches_vectorized,
             test_own_behavior_never_enters_own_features_and_flows_to_next_row,
             test_item_diff_excludes_own_group,
             test_no_test_rows_and_stats_from_train_only,
             test_item_m_constants_and_unseen_item,
             test_first_row_features_are_zero_or_train_mean,
             test_no_nan_or_inf]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
