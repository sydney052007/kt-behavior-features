"""跨學校 5-fold 的測試。可直接用 python3 tests/experiments/cross_school/test_xschool.py 跑。

需要 data/split_groups.csv（先跑 src/experiments/cross_school/make_xschool_groups.py）。全程只用 trainval 列。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
import features_r1 as fr  # noqa: E402
from cv import run_cv  # noqa: E402
from features_r1 import R0D_FEATURES, R1_FEATURES, load_r1_frame, make_fold_features  # noqa: E402
from run_r1_lr import make_model as make_lr  # noqa: E402
from evaluate import eval_c  # noqa: E402
from run_xschool import by_cluster  # noqa: E402  # import 只會載入函式與常數，不讀資料、不寫檔（main() 在 __main__ 裡）
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
    prev = pd.read_csv("results/02_analysis/r1_lr_cv.csv")
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


def _fake_c_oof(seed, n_groups, effect, lo, hi):
    """假的 OOF（程式內產生，不讀任何真實資料），欄位同 run_xschool 傳給 by_cluster 的 oof：
    user_id, fold, t, correct, p_correct, grp。每位學生有 t = 1…12 的列（含 C 的切點 t=6、t=11）。
    每組有一個共同的組效應（logit 位移 ±effect），學生再各自加小雜訊；correct 與 p_correct 都帶這個組效應，
    所以同組學生的列彼此相關。user_id 從 1001 起、grp 是 0…n_groups-1，兩者數值不同（才分得出有沒有被換掉）。"""
    rng = np.random.default_rng(seed)
    offsets = np.linspace(-effect, effect, n_groups)
    rng.shuffle(offsets)
    rows, uid = [], 1000
    for g in range(n_groups):
        for _ in range(int(rng.integers(lo, hi + 1))):
            uid += 1
            student = rng.normal(0, 0.3)
            for t in range(1, 13):
                correct = int(rng.random() < 1 / (1 + np.exp(-(offsets[g] + student))))
                p_correct = 1 / (1 + np.exp(-(offsets[g] + rng.normal(0, 0.3))))
                rows.append((uid, int(rng.integers(0, 5)), t, correct, p_correct, g))
    return pd.DataFrame(rows, columns=["user_id", "fold", "t", "correct", "p_correct", "grp"])


def test_by_cluster_replaces_user_id_with_group():
    """by_cluster 要把 user_id 欄換成組別（evaluate 的重抽單位就是這一欄），且不改到原本的 oof。"""
    oof = _fake_c_oof(seed=0, n_groups=6, effect=1.0, lo=4, hi=8)
    assert oof["grp"].nunique() == 6 and {6, 11} <= set(oof["t"])
    assert not oof["user_id"].equals(oof["grp"]), "假資料的 user_id 與 grp 不該相同，否則測不出差別"
    before = oof.copy(deep=True)
    out = by_cluster(oof)
    assert out["user_id"].equals(oof["grp"]), "by_cluster 的 user_id 欄必須等於 grp"
    assert oof.equals(before), "by_cluster 不可修改傳進去的 oof"
    assert out is not oof
    for c in oof.columns.drop("user_id"):
        assert out[c].equals(oof[c]), f"欄位 {c} 不該被改動"
    assert list(out.columns) == list(oof.columns) and len(out) == len(oof)


def test_cluster_ci_is_wider_than_student_ci():
    """換重抽單位要真的有效果：組內學生相關時，以組別重抽的 C 信賴區間要比以學生重抽的寬。

    假資料：6 組、每組 8–12 位學生、組效應 ±3.0（logit）。這組參數是挑過的：每組 4–8 人、組效應較小時
    組別 CI 不一定比較寬（20 個資料種子中只有 14～18 次），因為學生層級的重抽在小組時已經很接近組別層級；
    調成這組後 20 個種子、t=6 與 t=11 兩個切點都是組別 CI 較寬（寬度比最小 1.47 與 1.11）。"""
    import contextlib
    import io
    n_groups = 6
    oof = _fake_c_oof(seed=0, n_groups=n_groups, effect=3.0, lo=8, hi=12)
    n_students = oof["user_id"].nunique()
    with contextlib.redirect_stdout(io.StringIO()):          # eval_c 遇到單一類別的重抽會印訊息
        by_student = pd.DataFrame(eval_c(oof, n_boot=300, seed=42))
        by_group = pd.DataFrame(eval_c(by_cluster(oof), n_boot=300, seed=42))

    def roc_row(d, grp):
        r = d[(d["metric"] == "roc_auc") & (d["group"] == grp)]
        assert len(r) == 1
        return r.iloc[0]

    for grp in ("t=6", "t=11"):
        a, b = roc_row(by_student, grp), roc_row(by_group, grp)
        assert a["n_students"] == n_students and b["n_students"] == n_groups, (grp, a["n_students"], b["n_students"])
        assert np.isclose(a["value"], b["value"], rtol=0, atol=1e-12), "點估計不該因重抽單位而改變"
        w_student, w_group = a["ci_high"] - a["ci_low"], b["ci_high"] - b["ci_low"]
        assert w_group > w_student, f"{grp}：組別 CI 寬度 {w_group:.4f} 應大於學生 CI 寬度 {w_student:.4f}"


def _fake_org(shuffle_seed=None):
    """build_groups 只看 school_id、teacher_id、student_class_id（型別與真實資料相同：int64）。
    學校 1—2 共用 class 201；2—3 共用 teacher 102；學校 4 與誰都不共用；5—6 只共用 teacher 105。"""
    rows = [  # (school_id, teacher_id, student_class_id)
        (1, 101, 201), (1, 101, 201),
        (2, 102, 201), (2, 102, 202),
        (3, 102, 203), (3, 103, 203),
        (4, 104, 204), (4, 104, 204),
        (5, 105, 205), (6, 105, 206),
    ]
    df = pd.DataFrame(rows, columns=["school_id", "teacher_id", "student_class_id"]).astype("int64")
    if shuffle_seed is not None:
        df = df.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    return df


def test_build_groups_transitive_minimal_and_deterministic():
    tv = _fake_org()
    g = build_groups(tv)
    assert g.index.is_monotonic_increasing and list(g.index) == [1, 2, 3, 4, 5, 6]
    assert g[1] == g[2] == g[3], "傳遞性：1—2（class）、2—3（teacher）→ 同一組"
    assert g[4] != g[1] and g[4] != g[5], "最小性：學校 4 沒有共用 → 自己一組"
    assert (g == g[4]).sum() == 1
    assert g[5] == g[6] and g[5] != g[1], "另一條鏈：5—6 只靠 teacher → 同一組，且不與 1、2、3 同組"
    # group_id 依組內最小 school_id 由小到大編號 0…
    assert g.to_dict() == {1: 0, 2: 0, 3: 0, 4: 1, 5: 2, 6: 2}
    # 確定性：呼叫兩次相同；列順序打亂也相同
    assert build_groups(tv).equals(g)
    for seed in (0, 1, 2):
        assert build_groups(_fake_org(shuffle_seed=seed)).equals(g)


def test_build_groups_numbering_follows_min_school_id_not_appearance_order():
    # 學校 9 先出現、與學校 3 共用 class；學校 5 單獨。依最小 school_id：{3, 9} → 0，{5} → 1
    tv = pd.DataFrame({"school_id": [9, 3, 5], "teacher_id": [1, 2, 3],
                       "student_class_id": [7, 7, 8]}).astype("int64")
    assert build_groups(tv).to_dict() == {3: 0, 5: 1, 9: 0}


if __name__ == "__main__":
    tests = [test_groups_are_clean_and_folds_disjoint, test_groups_file_matches_recomputation_and_has_no_student_data,
             test_no_test_students_anywhere, test_inner_folds_never_split_a_group, test_item_diff_excludes_own_group,
             test_xschool_features_equal_student_version_when_inner_is_the_same,
             test_lr_under_student_folds_reproduces_existing_csv, test_unseen_problem_uses_skill_mean_in_xschool_val,
             test_by_cluster_replaces_user_id_with_group, test_cluster_ci_is_wider_than_student_ci,
             test_build_groups_transitive_minimal_and_deterministic,
             test_build_groups_numbering_follows_min_school_id_not_appearance_order]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
