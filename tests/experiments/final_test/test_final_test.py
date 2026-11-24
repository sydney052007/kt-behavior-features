"""test 最終評估流程（run_final_test.py）的測試。可直接用 python3 tests/experiments/final_test/test_final_test.py 跑。

全部用假資料（make_fake_frame），不讀 data/、不碰真實的 test 學生。
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
import run_final_test as rf  # noqa: E402  # import 只載入函式與常數，不讀資料、不寫檔

N_BOOT = 40
_cache = {}


def _tables():
    if "t" not in _cache:
        ref = pd.DataFrame([dict(model_type=mt, features=fs, metric=m, cv_mean=0.7, cv_std=0.02)
                            for mt in rf.MODEL_TYPES for fs in rf.FEATURE_SETS for m in ("roc_auc", "pr_auc")])
        _cache["t"] = rf.run_final(rf.make_fake_frame(), cv_ref=ref, n_boot=N_BOOT)
    return _cache["t"]


def test_pipeline_runs_and_outputs_are_aggregates_only():
    t = _tables()
    assert list(t) == rf.OUT_NAMES
    a, p, bc, th = (t[n] for n in rf.OUT_NAMES)
    assert len(a) == 12 and set(a["metric"]) == {"roc_auc", "pr_auc"}               # 6 組模型 × 2 指標
    assert len(th) == 6 and len(p) == 2 * 2 * 3                                     # 模型類型 × 2 組比較 × (全部列、t=6、t=11)
    assert set(p["comparison"]) == {"R1 - R0d", "R0d - R0"} and set(p["group"]) == {"all_rows", "t=6", "t=11"}
    assert set(bc["eval"]) == {"B", "C"}
    for name, tab in t.items():
        assert not ({"user_id", "school_id", "order_id", "teacher_id", "student_class_id", "problem_id", "p_correct"}
                    & {c.lower() for c in tab.columns}), name
        assert len(tab) < 500, name                                                 # 彙總表很小，不是逐列資料
    assert np.allclose(a["test_minus_cv"], a["value"] - 0.7)                         # 與 5-fold 並列
    assert set(p["interpretation"]) <= {"改善", "變差", "無法確認"}
    # 混淆矩陣的內部一致：TP+FP+FN+TN = 列數；F1、Accuracy 由它推得
    assert ((th["TP"] + th["FP"] + th["FN"] + th["TN"]) == th["n_rows"]).all()
    assert np.allclose(th["accuracy"], (th["TP"] + th["TN"]) / th["n_rows"])
    assert np.allclose(th["f1"], 2 * th["TP"] / (2 * th["TP"] + th["FP"] + th["FN"]))


def test_test_labels_never_influence_training():
    df = rf.make_fake_frame()
    shuffled = df.copy()
    m = (shuffled["role"] == "test").to_numpy()
    shuffled.loc[m, "correct"] = np.random.default_rng(99).integers(0, 2, int(m.sum()))   # test 答案全部改亂數
    assert not shuffled["correct"].equals(df["correct"])
    fits, feats = [], []
    for d in (df, shuffled):
        train, test = d[d["role"] == "trainval"].reset_index(drop=True), d[d["role"] == "test"].reset_index(drop=True)
        tr_f, te_f, st = rf.make_final_features(train, test)
        out = {}
        for mt in rf.MODEL_TYPES:
            for fs, cols in rf.FEATURE_SETS.items():
                out[(mt, fs)] = rf.fit_predict(tr_f, train["correct"], te_f, cols, rf.make_model_for(mt, fs))
        fits.append(out)
        feats.append((tr_f, te_f, st))
    assert feats[0][0].equals(feats[1][0]), "訓練列的特徵被 test 標籤影響了"
    assert feats[0][2]["fills"] == feats[1][2]["fills"] and feats[0][2]["p1"] == feats[1][2]["p1"]
    assert not feats[0][1]["last_correct"].equals(feats[1][1]["last_correct"])     # test 列自己的歷史特徵確實變了（不是空轉）
    for fs in rf.FEATURE_SETS:
        a, b = fits[0][("LR", fs)], fits[1][("LR", fs)]
        assert np.array_equal(a["model"].named_steps["lr"].coef_, b["model"].named_steps["lr"].coef_)   # 係數逐位相同
        assert a["threshold"] == b["threshold"]
        r, s = fits[0][("RF", fs)], fits[1][("RF", fs)]
        assert np.allclose(r["p_train"], s["p_train"], atol=1e-9)                   # RF 多執行緒有約 1e-16 的浮點誤差
    # 需要 test 標籤的只有評估：改亂數後 test 的 AUC 會不同
    assert not np.allclose(fits[0][("LR", "R1")]["p_test"], fits[1][("LR", "R1")]["p_test"])


def test_final_features_reject_trainval_in_test_slot():
    df = rf.make_fake_frame()
    train = df[df["role"] == "trainval"].reset_index(drop=True)
    try:
        rf.make_final_features(train, train.head(50))
    except AssertionError:
        pass
    else:
        raise AssertionError("第二個輸入不是 test 列時應該報錯")


def test_existing_outputs_abort_and_nothing_is_overwritten():
    t = _tables()
    with tempfile.TemporaryDirectory() as d:
        rf.write_outputs(t, d)
        assert sorted(os.listdir(d)) == sorted(rf.OUT_NAMES)
        before = {n: open(os.path.join(d, n)).read() for n in rf.OUT_NAMES}
        try:
            rf.write_outputs(t, d)
        except rf.GateError:
            pass
        else:
            raise AssertionError("輸出檔已存在時應該中止")
        assert before == {n: open(os.path.join(d, n)).read() for n in rf.OUT_NAMES}      # 沒被覆寫
    with tempfile.TemporaryDirectory() as d:                                               # 只存在一個檔也要中止，且不寫其他檔
        open(os.path.join(d, rf.OUT_NAMES[2]), "w").close()
        try:
            rf.write_outputs(t, d)
        except rf.GateError:
            pass
        else:
            raise AssertionError("有任何一個輸出檔存在就應該中止")
        assert os.listdir(d) == [rf.OUT_NAMES[2]]


def test_clean_tree_gate():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as d:
        try:
            os.chdir(d)
            subprocess.run(["git", "init", "-q"], check=True)
            rf.check_clean_tree()                                                       # 空的乾淨工作樹：通過
            os.makedirs("results/03_process")
            open(rf.LOG_PATH, "w").close()
            rf.check_clean_tree()                                                       # 只有 log 檔是未追蹤：允許
            open("other.txt", "w").close()
            try:
                rf.check_clean_tree()
            except rf.GateError:
                pass
            else:
                raise AssertionError("工作樹不乾淨應該中止")
        finally:
            os.chdir(cwd)


def test_default_mode_runs_fake_self_check_without_data():
    """預設（沒有旗標）只跑假資料：在沒有 data/ 的空資料夾執行也要成功，且不產生任何檔案。"""
    script = os.path.abspath(os.path.join(HERE, "..", "..", "..", "src", "experiments", "final_test", "run_final_test.py"))
    with tempfile.TemporaryDirectory() as d:
        r = subprocess.run([sys.executable, script], cwd=d, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr[-500:]
        assert "[self-check] 通過" in r.stdout and os.listdir(d) == []


if __name__ == "__main__":
    tests = [test_pipeline_runs_and_outputs_are_aggregates_only, test_test_labels_never_influence_training,
             test_final_features_reject_trainval_in_test_slot, test_existing_outputs_abort_and_nothing_is_overwritten,
             test_clean_tree_gate, test_default_mode_runs_fake_self_check_without_data]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
