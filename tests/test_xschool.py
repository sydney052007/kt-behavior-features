"""跨學校 5-fold 的測試。可直接用 python3 tests/test_xschool.py 跑。

需要 data/split_groups.csv（先跑 src/make_xschool_groups.py）。全程只用 trainval 列。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import features_r1 as fr  # noqa: E402
from cv import run_cv  # noqa: E402
from features_r1 import R0D_FEATURES, R1_FEATURES, load_r1_frame, make_fold_features  # noqa: E402
from run_r1_lr import make_model as make_lr  # noqa: E402
from xschool import (GROUPS_PATH, assign_group_folds, build_groups, inner_groups_by_group,  # noqa: E402
                     item_diff_train_oof_groups, load_trainval_with_org, load_xschool_frame,
                     make_fold_features_xschool)

_cache = {}


def _x():
    if "x" not in _cache:
        _cache["x"] = load_xschool_frame()
    return _cache["x"]


def _fold(k):
    if k not in _cache:
        x = _x()
        train, val = x[x["fold"] != k], x[x["fold"] == k]
        _cache[k] = (train, val) + make_fold_features_xschool(train, val)
    return _cache[k]


# ------------------------------------------------------------------ 1. 組別與 fold 的結構

def test_groups_are_clean_and_folds_disjoint():
    x = _x()
    # 每所學校只屬於一組、每組只屬於一個 fold
    assert x.groupby("school_id")["grp"].nunique().max() == 1
    assert x.groupby("grp")["fold"].nunique().max() == 1
    # 連通分量：同一個 class 或 teacher 的所有列一定在同一組（才不會有 class／teacher 跨 fold）
    assert x.groupby("student_class_id")["grp"].nunique().max() == 1
    assert x.groupby("teacher_id")["grp"].nunique().max() == 1
    assert x.groupby("user_id")["fold"].nunique().max() == 1            # 學生不會跨 fold
    for k in range(5):
        train, val, *_ = _fold(k)
        assert set(train["grp"]).isdisjoint(set(val["grp"])), f"fold {k} 訓練與驗證的組別有交集"
        assert set(train["school_id"]).isdisjoint(set(val["school_id"]))
        assert set(train["user_id"]).isdisjoint(set(val["user_id"]))
    assert set(x["fold"]) == set(range(5)) and x["grp"].nunique() == 35


def test_groups_file_matches_recomputation_and_has_no_student_data():
    g = pd.read_csv(GROUPS_PATH)
    assert list(g.columns) == ["school_id", "group_id", "fold"]          # 只有學校層級資料
    tv = load_trainval_with_org()
    s2g = build_groups(tv)
    gf = assign_group_folds(tv, s2g)
    redo = pd.DataFrame({"school_id": s2g.index, "group_id": s2g.to_numpy()})
    redo["fold"] = redo["group_id"].map(gf).astype(int)
    assert g.equals(redo), "split_groups.csv 與重新計算的結果不同（確定性）"
    assert g["school_id"].nunique() == 40 and g["group_id"].nunique() == 35


# ------------------------------------------------------------------ 2. 無 test

def test_no_test_students_anywhere():
    full = load_r1_frame()
    test_users = set(full.loc[full["role"] == "test", "user_id"])
    test_orders = set(full.loc[full["role"] == "test", "order_id"])
    assert len(test_users) == 262
    x = _x()
    assert (x["role"] == "trainval").all() and len(x) == 30516
    assert test_users.isdisjoint(set(x["user_id"])) and test_orders.isdisjoint(set(x["order_id"]))
    for k in range(5):
        _, _, tf, vf, _ = _fold(k)
        assert test_orders.isdisjoint(tf["order_id"]) and test_orders.isdisjoint(vf["order_id"])
    # 把 test 列混進來必須報錯
    bad = full[full["role"] == "test"].head(30).assign(grp=-1)
    try:
        make_fold_features_xschool(x[x["fold"] != 0], bad)
    except AssertionError:
        pass
    else:
        raise AssertionError("應該拒絕 test 列")


# ------------------------------------------------------------------ 3. inner fold 依組別

def test_inner_folds_never_split_a_group():
    for k in range(5):
        train = _fold(k)[0]
        inner = inner_groups_by_group(train)
        assert set(inner) == set(range(5))
        per_group = pd.Series(inner, index=train.index).groupby(train["grp"]).nunique()
        assert per_group.max() == 1, f"fold {k}：同一組跨了多個 inner fold"
        assert np.array_equal(inner, inner_groups_by_group(train))      # 確定性


def test_item_diff_excludes_own_group():
    train, val, tf, vf, st = _fold(0)
    inner = inner_groups_by_group(train)
    base = item_diff_train_oof_groups(train, inner)
    assert base.equals(tf["item_diff"])
    rng = np.random.default_rng(1)
    other_changed = 0
    for pos in rng.choice(len(train), size=15, replace=False):
        idx = train.index[pos]
        mod = train.copy()
        mod.at[idx, "correct"] = 1 - mod.at[idx, "correct"]
        new = item_diff_train_oof_groups(mod, inner)
        same = inner == inner[pos]
        assert new.at[idx] == base.at[idx]
        assert np.array_equal(new[same].to_numpy(), base[same].to_numpy()), "同一 inner 組的列變了"
        other_changed += int((new[~same] != base[~same]).sum())
    assert other_changed > 0


def test_xschool_features_equal_student_version_when_inner_is_the_same():
    """我的 item_diff_train_oof_groups 與 features_r1.item_diff_train_oof 在相同 inner 分組下完全一致；
    而且 transform 與驗證列的編碼沿用 features_r1，所以只有 inner 分組不同。"""
    train, val, tf, vf, st = _fold(0)
    same_inner = fr.inner_groups(train)                      # features_r1 依學生的 inner 分組
    assert item_diff_train_oof_groups(train, same_inner).equals(fr.item_diff_train_oof(train))
    tf0, vf0, st0 = make_fold_features(train, val)
    assert vf[R1_FEATURES].equals(vf0[R1_FEATURES])          # 驗證列的特徵與 inner 分組無關：完全相同
    assert st["p99"] == st0["p99"] and st["fills"] == st0["fills"]
    cols = [c for c in R1_FEATURES if c != "item_diff"]
    assert tf[cols].equals(tf0[cols])                        # 訓練列只有 item_diff 不同
    assert not tf["item_diff"].equals(tf0["item_diff"])


# ------------------------------------------------------------------ 4. 學生分組下 LR 與既有 csv 一致

def test_lr_under_student_folds_reproduces_existing_csv():
    full = load_r1_frame()
    tv = full[full["role"] == "trainval"].reset_index(drop=True)      # 原本的學生 fold
    prev = pd.read_csv("results/r1_lr_cv.csv")
    prev = prev[(prev["eval"] == "A")].set_index(["model", "metric"])
    from evaluate import per_fold_metric
    for name, feats in [("R0d", R0D_FEATURES), ("R1", R1_FEATURES)]:
        res = run_cv(tv, feats, make_lr, feature_builder=make_fold_features)
        roc, pr = per_fold_metric(res.oof), per_fold_metric(res.oof, "pr_auc")
        assert abs(roc.mean() - prev.at[(name, "roc_auc"), "value"]) < 1e-9, name
        assert abs(roc.std(ddof=1) - prev.at[(name, "roc_auc"), "std"]) < 1e-9, name
        assert abs(pr.mean() - prev.at[(name, "pr_auc"), "value"]) < 1e-9, name


# ------------------------------------------------------------------ 5. 沒見過的題目 = 技能平均

def test_unseen_problem_uses_skill_mean_in_xschool_val():
    train, val, tf, vf, st = _fold(0)
    seen = set(train["problem_id"])
    unseen = val[~val["problem_id"].isin(seen)]
    if len(unseen):
        p_bar = train.groupby("skill_id")["correct"].mean()
        got = vf.loc[unseen.index, "item_diff"].to_numpy()
        assert np.allclose(got, unseen["skill_id"].map(p_bar).to_numpy(), atol=1e-12)


if __name__ == "__main__":
    tests = [test_groups_are_clean_and_folds_disjoint, test_groups_file_matches_recomputation_and_has_no_student_data,
             test_no_test_students_anywhere, test_inner_folds_never_split_a_group, test_item_diff_excludes_own_group,
             test_xschool_features_equal_student_version_when_inner_is_the_same,
             test_lr_under_student_folds_reproduces_existing_csv, test_unseen_problem_uses_skill_mean_in_xschool_val]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
