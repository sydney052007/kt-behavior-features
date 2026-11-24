"""test 最終評估（協定見 CLAUDE.md「test 最終評估協定」，跑之前已登記；只能跑一次）。

    python3 src/experiments/final_test/run_final_test.py                  # 預設：只跑假資料自我檢查，不讀 data/
    python3 src/experiments/final_test/run_final_test.py --run-real-test > results/03_process/test_final_run.log 2>&1

只有加 --run-real-test 才會讀真實資料裡的 test 列。真實執行前會檢查：
    (1) git 工作樹乾淨（只允許那個 log 檔是未追蹤的）；(2) 四個輸出檔都還不存在（已存在就中止，不覆寫）。

流程：用全部 trainval 訓練 R0／R0d／R1 × LR／RF 共 6 組（RF 超參數沿用 5-fold 選出的，不重新搜尋），
訓練集統計量（p1、p99、題目中位數、填補值、item_diff）由全 trainval 重算；
訓練列的 item_diff 用 inner OOF，test 列用全 trainval。F1／Accuracy 的閾值用最終模型在訓練列上的預測選。
輸出（只有彙總，不含 user_id、school_id，逐列預測不存檔）：
    test_final_a.csv          全部列 ROC-AUC／PR-AUC ＋ cluster bootstrap 區間，並列 5-fold 的 A 平均 ± 標準差
    test_final_paired.csv     R1 − R0d、R0d − R0 的配對 ΔAUC 與區間（全部列、C 的 t = 6、11；LR、RF）與判讀
    test_final_b_c.csv        B 位置分桶、C（t = 6、11，附區間）
    test_final_threshold.csv  閾值、F1、Accuracy、混淆矩陣（描述性）
"""
import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # src/
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from eval_final import eval_all_rows, interpret_delta, paired_delta_all  # noqa: E402
from evaluate import N_BOOT, eval_b, eval_c, paired_delta_c, select_threshold, to_wrong  # noqa: E402
from features import R0_FEATURES, SEQ  # noqa: E402
from features_r1 import (R0D_FEATURES, R1_FEATURES, fit_stats, item_diff_encode,  # noqa: E402
                         item_diff_train_oof, load_r1_frame, transform)
from run_r1_lr import make_model as make_lr  # noqa: E402
from run_rf import make_rf  # noqa: E402

OUT_DIR = "results/02_analysis/"
OUT_NAMES = ["test_final_a.csv", "test_final_paired.csv", "test_final_b_c.csv", "test_final_threshold.csv"]
LOG_PATH = "results/03_process/test_final_run.log"      # 重導向 stdout 產生的 log，不算「工作樹不乾淨」
FEATURE_SETS = {"R0": R0_FEATURES, "R0d": R0D_FEATURES, "R1": R1_FEATURES}
RF_CFG = {"R0": (8, 50), "R0d": (8, 5), "R1": (None, 50)}      # (max_depth, min_samples_leaf)，沿用 5-fold 的選擇
MODEL_TYPES = ["LR", "RF"]
PAIRS = [("R1", "R0d"), ("R0d", "R0")]                          # (新, 舊)：Δ = 新 − 舊
CV_LR, CV_RF = "results/02_analysis/r1_lr_cv.csv", "results/02_analysis/rf_cv.csv"
N_TRAINVAL_STUDENTS, N_TEST_STUDENTS = 1481, 262


class GateError(RuntimeError):
    """安全閘門沒過（工作樹不乾淨、輸出檔已存在……）。"""


# ---------------------------------------------------------------- 安全閘門

def check_outputs_absent(out_dir=OUT_DIR):
    exist = [n for n in OUT_NAMES if os.path.exists(os.path.join(out_dir, n))]
    if exist:
        raise GateError(f"輸出檔已存在，中止（test 只能跑一次，不覆寫）：{exist}")


def check_clean_tree():
    out = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], capture_output=True, text=True, check=True).stdout
    dirty = [ln for ln in out.splitlines() if not ln.rstrip().endswith(LOG_PATH)]
    if dirty:
        raise GateError("git 工作樹不乾淨，中止（請先 commit）：\n" + "\n".join(dirty))


# ---------------------------------------------------------------- 特徵與模型

def make_final_features(train, test):
    """最終用的特徵：統計量全由全 trainval 算；test 列只用「自己的過去作答」。

    和 make_fold_features 的差別只有一個：這裡的第二個輸入「必須」是 test 列
    （make_fold_features 刻意拒絕 test 列，所以另寫）。重用 fit_stats、transform、item_diff_*。
    """
    assert (train["role"] == "trainval").all() and (test["role"] == "test").all()
    assert set(train["user_id"]).isdisjoint(set(test["user_id"])), "trainval 與 test 學生有交集"
    stats = fit_stats(train)
    train_feat = transform(train, stats, item_diff_train_oof(train))       # 訓練列：inner OOF
    test_feat = transform(test, stats, item_diff_encode(train, test))      # test 列：全 trainval；沒見過的題目 = p̄
    return train_feat, test_feat, stats


def fit_predict(train_feat, y_train, test_feat, features, make_model):
    """用全部 trainval 訓練一個最終模型；閾值用它在訓練列上的預測選。回傳 dict（模型、閾值、test 預測）。"""
    model = make_model()
    model.fit(train_feat[features], y_train)
    col = list(model.classes_).index(1)                                    # P(答對) 那一欄
    p_train = model.predict_proba(train_feat[features])[:, col]
    thr = select_threshold(y_train.to_numpy(), p_train)
    return dict(model=model, threshold=thr, p_train=p_train, p_test=model.predict_proba(test_feat[features])[:, col])


def make_model_for(model_type, fs):
    return make_lr if model_type == "LR" else make_rf(*RF_CFG[fs])


# ---------------------------------------------------------------- 5-fold 對照值

def load_cv_reference():
    """5-fold 的 A ROC-AUC／PR-AUC 平均與標準差（既有結果檔）；RF 要確認用的就是登記的超參數。"""
    lr = pd.read_csv(CV_LR, keep_default_na=False, float_precision="round_trip")
    rf = pd.read_csv(CV_RF, keep_default_na=False, float_precision="round_trip", dtype={"max_depth": str})
    rows = []
    for fs in FEATURE_SETS:
        d, leaf = RF_CFG[fs]
        for mt, tab in [("LR", lr), ("RF", rf)]:
            sel = tab[(tab["model"] == fs) & (tab["eval"] == "A") & (tab["group"] == "all_rows")]
            if mt == "RF":
                sel = sel[(sel["max_depth"] == str(d)) & (sel["min_samples_leaf"] == leaf)]
            for metric in ["roc_auc", "pr_auc"]:
                r = sel[sel["metric"] == metric]
                assert len(r) == 1, (mt, fs, metric)
                rows.append(dict(model_type=mt, features=fs, metric=metric,
                                 cv_mean=float(r["value"].iloc[0]), cv_std=float(r["std"].iloc[0])))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程（輸入是一張含 trainval 與 test 的表）

def run_final(df, cv_ref=None, n_boot=N_BOOT):
    """回傳 {檔名: 彙總表}。df 需要 role（trainval／test）與特徵管線需要的欄位（見 load_r1_frame）。
    不寫檔、不存逐列預測；cv_ref = load_cv_reference() 的結果（可為 None，則對照欄位留空）。"""
    train = df[df["role"] == "trainval"].reset_index(drop=True)
    test = df[df["role"] == "test"].reset_index(drop=True)
    assert len(train) > 0 and len(test) > 0
    train_feat, test_feat, _ = make_final_features(train, test)

    oofs, fits = {}, {}                      # 只在記憶體；格式同 evaluate 的 OOF（fold 全設 0）
    for mt in MODEL_TYPES:
        for fs, feats in FEATURE_SETS.items():
            fit = fit_predict(train_feat, train["correct"], test_feat, feats, make_model_for(mt, fs))
            fits[(mt, fs)] = fit
            oofs[(mt, fs)] = test[["user_id", "skill_id", "order_id", "t", "correct"]].assign(
                fold=0, p_correct=fit["p_test"])

    a_rows, bc_rows, thr_rows, pair_rows = [], [], [], []
    for (mt, fs), oof in oofs.items():
        key = dict(model_type=mt, features=fs)
        a_rows += [dict(key, **r) for r in eval_all_rows(oof, n_boot=n_boot)]
        bc_rows += [dict(key, **r) for r in eval_b(oof) + eval_c(oof, n_boot=n_boot)]
        y, score = to_wrong(oof["correct"], oof["p_correct"])
        thr = fits[(mt, fs)]["threshold"]
        pred = (score >= thr).astype(int)                                  # 分數 >= 閾值 → 預測答錯（正類）
        thr_rows.append(dict(key, threshold=thr, f1=f1_score(y, pred, zero_division=0),
                             accuracy=accuracy_score(y, pred), acc_baseline_all_correct=accuracy_score(y, np.zeros_like(y)),
                             TP=int(((pred == 1) & (y == 1)).sum()), FP=int(((pred == 1) & (y == 0)).sum()),
                             FN=int(((pred == 0) & (y == 1)).sum()), TN=int(((pred == 0) & (y == 0)).sum()),
                             n_rows=len(oof)))
    for mt in MODEL_TYPES:
        for new, old in PAIRS:
            o_old, o_new = oofs[(mt, old)], oofs[(mt, new)]
            res = pd.concat([paired_delta_all(o_old, o_new, n_boot=n_boot).assign(eval="ALL"),
                             paired_delta_c(o_old, o_new, n_boot=n_boot).assign(eval="C")], ignore_index=True)
            res.insert(0, "comparison", f"{new} - {old}")
            res.insert(0, "model_type", mt)
            res["interpretation"] = [interpret_delta(lo, hi) for lo, hi in zip(res["ci_low"], res["ci_high"])]
            pair_rows.append(res)

    a = pd.DataFrame(a_rows)
    ref = cv_ref if cv_ref is not None else pd.DataFrame(columns=["model_type", "features", "metric", "cv_mean", "cv_std"])
    a = a.merge(ref, on=["model_type", "features", "metric"], how="left")
    a["test_minus_cv"] = a["value"] - a["cv_mean"]
    tables = {"test_final_a.csv": a, "test_final_paired.csv": pd.concat(pair_rows, ignore_index=True),
              "test_final_b_c.csv": pd.DataFrame(bc_rows), "test_final_threshold.csv": pd.DataFrame(thr_rows)}
    assert list(tables) == OUT_NAMES
    for name, t in tables.items():                                         # 只有彙總：不得含學生／學校層級欄位
        assert not ({"user_id", "school_id", "order_id", "p_correct"} & set(t.columns)), name
    return tables


def write_outputs(tables, out_dir=OUT_DIR):
    """先確認四個檔都不存在，再用 'x' 模式建立（不會覆寫既有檔）。"""
    check_outputs_absent(out_dir)
    for name, t in tables.items():
        with open(os.path.join(out_dir, name), "x", newline="") as f:
            t.to_csv(f, index=False)


# ---------------------------------------------------------------- 假資料（自我檢查與測試用；不讀 data/）

def make_fake_frame(seed=0, n_trainval=60, n_test=25):
    """和 load_r1_frame 同欄位的假資料：每位學生兩個技能、各 8–16 列。回傳含 role 的 DataFrame。"""
    rng = np.random.default_rng(seed)
    rows, order = [], 0
    for u in range(n_trainval + n_test):
        role = "trainval" if u < n_trainval else "test"
        for sk in ("47", "311"):
            c = w = 0
            for t in range(1, int(rng.integers(8, 17)) + 1):
                prob = int(rng.integers(1, 25)) + (0 if sk == "47" else 100)
                correct = int(rng.random() < 0.55 + 0.25 * (prob % 3 == 0) - 0.2 * (prob % 5 == 0))
                hint = 0 if correct else int(rng.integers(0, 3))
                rows.append(dict(
                    user_id=1000 + u, skill_id=sk, order_id=order, t=t, correct=correct,
                    fold=(u % 5) if role == "trainval" else -1, role=role,
                    log1p_c=np.log1p(c), log1p_w=np.log1p(w), skill_311=int(sk == "311"), has_history=int(t > 1),
                    problem_id=prob, hint_count=hint, first_action=2 if hint == 2 else 0,
                    attempt_count=int(rng.integers(1, 5)), ms_first_response=int(rng.integers(-1000, 60000))))
                order += 1
                c, w = c + correct, w + (1 - correct)
    return pd.DataFrame(rows).sort_values(SEQ + ["order_id"]).reset_index(drop=True)


def self_check():
    """預設模式：用假資料把整個流程跑一遍（小的 n_boot），不讀 data/、不寫檔。"""
    tables = run_final(make_fake_frame(), cv_ref=None, n_boot=30)
    for name, t in tables.items():
        print(f"[self-check] {name}：{len(t)} 列，欄位 {list(t.columns)}")
    print("[self-check] 通過（假資料；沒有讀 data/、沒有寫檔）。真實執行要加 --run-real-test。")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-real-test", action="store_true", help="讀真實 test 列並跑最終評估（只能跑一次）")
    if not ap.parse_args().run_real_test:
        self_check()
        return
    try:
        check_clean_tree()
        check_outputs_absent()
    except GateError as e:
        sys.exit(f"[final_test] {e}")
    df = load_r1_frame()
    n_tv, n_te = (df.loc[df["role"] == r, "user_id"].nunique() for r in ("trainval", "test"))
    assert (n_tv, n_te) == (N_TRAINVAL_STUDENTS, N_TEST_STUDENTS), (n_tv, n_te)
    print(f"[final_test] trainval {n_tv} 位、test {n_te} 位學生；開始最終評估（只跑這一次）", flush=True)
    tables = run_final(df, cv_ref=load_cv_reference())
    write_outputs(tables)
    for name, t in tables.items():
        print(f"[final_test] 已寫出 {OUT_DIR}{name}（{len(t)} 列）")
    print(tables["test_final_a.csv"].to_string(index=False))
    print(tables["test_final_paired.csv"].to_string(index=False))


if __name__ == "__main__":
    main()
