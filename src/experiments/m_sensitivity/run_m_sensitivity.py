"""item_diff 平滑參數 m 的敏感度實驗（只報告，不改採用的 m；規則見 CLAUDE.md 同名一節，跑之前已登記）。

    python3 src/experiments/m_sensitivity/run_m_sensitivity.py > results/03_process/m_sensitivity_run.log 2>&1

設定：學生分組、既有 split 與 inner fold（種子 42），只動 m；一次只動一個技能，另一個維持採用值。
    採用值（47 → 5、311 → 20）、47 ∈ {4, 6}（311 = 20）、311 ∈ {16, 24}（47 = 5），共 5 組設定。
特徵集：R0d、R1。模型：LR（StandardScaler + LogisticRegression(C=np.inf)）、
    RF（300 棵、sqrt、種子 42；R0d 用 depth 8／leaf 5，R1 用 depth None／leaf 50，不重新搜尋）。
判準（寫死）：對每個（模型, 特徵集），任何設定相對採用值的 |ΔA ROC-AUC| < 0.002，且各設定下
    R1 − R0d 都是 5/5 fold 較好 → 「不敏感」；否則「敏感」。四個（模型, 特徵集）全部不敏感才算整體不敏感。

輸出（只有彙總數字）：results/02_analysis/m_sensitivity_cv.csv（長格式）、m_sensitivity_verdict.csv
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # src/
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from cv import N_FOLDS, run_cv  # noqa: E402
from evaluate import per_fold_auc  # noqa: E402
from features_r1 import ITEM_M, R0D_FEATURES, R1_FEATURES, load_r1_frame, make_fold_features  # noqa: E402
from run_r1_lr import make_model as make_lr  # noqa: E402
from run_rf import make_rf  # noqa: E402

MAX_ABS_DELTA = 0.002            # 判準（寫死）：|ΔA ROC-AUC| 要小於這個值
OUT_CV = "results/02_analysis/m_sensitivity_cv.csv"
OUT_VERDICT = "results/02_analysis/m_sensitivity_verdict.csv"
ADOPTED = "adopted"
# (設定名稱, 47 的 m, 311 的 m)；採用值只算一次
SETTINGS = [(ADOPTED, 5, 20), ("m47=4", 4, 20), ("m47=6", 6, 20), ("m311=16", 5, 16), ("m311=24", 5, 24)]
# (模型, 特徵集) → (特徵欄位, 建模函式)；RF 用既有最佳設定，不重新搜尋
CELLS = {
    ("LR", "R0d"): (R0D_FEATURES, make_lr),
    ("LR", "R1"): (R1_FEATURES, make_lr),
    ("RF", "R0d"): (R0D_FEATURES, make_rf(8, 5)),
    ("RF", "R1"): (R1_FEATURES, make_rf(None, 50)),
}
LR_FOLDS = "results/02_analysis/r1_lr_folds.csv"
RF_GRID = "results/02_analysis/rf_grid_auc.csv"
RF_BEST = {"R0d": ("8", 5), "R1": ("None", 50)}     # (max_depth 字串, min_samples_leaf)
TOL = 1e-9


# ---------------------------------------------------------------- 判準（獨立函式，方便測邊界）

def cell_verdict(max_abs_delta: float, min_folds_better: int) -> str:
    """一個（模型, 特徵集）的判斷。max_abs_delta = 各非採用設定的 |ΔA ROC-AUC| 最大值；
    min_folds_better = 各設定（含採用值）下 R1 − R0d 較好 fold 數的最小值。"""
    ok = (max_abs_delta < MAX_ABS_DELTA) and (min_folds_better >= N_FOLDS)
    return "不敏感" if ok else "敏感"


def overall_verdict(cell_verdicts) -> str:
    """四個（模型, 特徵集）全部不敏感才算整體不敏感。"""
    return "不敏感" if all(v == "不敏感" for v in cell_verdicts) else "敏感"


# ---------------------------------------------------------------- 實驗

def caching_builder(m_map):
    """同一個 fold 的特徵只算一次（R0d 與 R1 共用）；m 由 m_map 指定。"""
    cache = {}

    def build(train, val):
        key = tuple(sorted(val["fold"].unique()))
        if key not in cache:
            cache[key] = make_fold_features(train, val, m_map)
        return cache[key]
    return build


def run_setting(tv, m47, m311):
    """一組 m 設定：回傳 {(模型, 特徵集): 各 fold 的 A ROC-AUC（Series，index = fold）}。"""
    builder = caching_builder({"47": m47, "311": m311})
    return {cell: per_fold_auc(run_cv(tv, feats, make, feature_builder=builder).oof)
            for cell, (feats, make) in CELLS.items()}


def check_adopted_matches_existing(auc):
    """採用值的各 fold A ROC-AUC 必須與既有結果一致（差 < 1e-9），否則停下來。"""
    lr = pd.read_csv(LR_FOLDS, float_precision="round_trip")
    grid = pd.read_csv(RF_GRID, float_precision="round_trip", dtype={"max_depth": str}, keep_default_na=False)  # "None" 是字串，不是缺失
    worst = 0.0
    for (mt, fs), s in auc.items():
        if mt == "LR":
            old = lr[lr["model"] == fs].set_index("fold")["roc_auc"]
        else:
            d, leaf = RF_BEST[fs]
            row = grid[(grid["features"] == fs) & (grid["max_depth"] == d) & (grid["min_samples_leaf"] == leaf)]
            assert len(row) == 1, (fs, d, leaf)
            old = pd.Series({k: row.iloc[0][f"fold{k}"] for k in range(N_FOLDS)})
        diff = float(np.max(np.abs(s.to_numpy() - old.loc[s.index].to_numpy())))
        worst = max(worst, diff)
        assert diff < TOL, f"採用值與既有結果不一致：{mt} {fs} 最大差 {diff:.2e}"
    print(f"[m_sens] 採用值與既有結果一致：4 組各 fold A ROC-AUC 最大差 {worst:.1e} < {TOL:.0e}", flush=True)


def build_tables(all_auc):
    """all_auc：{設定: {(模型, 特徵集): 各 fold AUC}} → (長格式表, verdict 表)。"""
    rows = []
    for name, m47, m311 in SETTINGS:
        auc = all_auc[name]
        for mt in ["LR", "RF"]:
            series = {"R0d": auc[(mt, "R0d")], "R1": auc[(mt, "R1")]}
            series["R1 - R0d"] = series["R1"] - series["R0d"]
            for fs, s in series.items():
                ref = {"R0d": all_auc[ADOPTED][(mt, "R0d")], "R1": all_auc[ADOPTED][(mt, "R1")]}
                ref["R1 - R0d"] = ref["R1"] - ref["R0d"]
                base = dict(setting=name, m_47=m47, m_311=m311, model_type=mt, features=fs, metric="A_roc_auc")
                for f in s.index:
                    rows.append(dict(base, fold=str(f), value=s[f], delta_vs_adopted=s[f] - ref[fs][f]))
                rows.append(dict(base, fold="mean", value=s.mean(), delta_vs_adopted=s.mean() - ref[fs].mean(),
                                 n_folds_better=int((s > 0).sum()) if fs == "R1 - R0d" else np.nan))
    cv = pd.DataFrame(rows)
    mean = cv[cv["fold"] == "mean"]
    vrows = []
    for (mt, fs) in CELLS:
        d = mean[(mean["model_type"] == mt) & (mean["features"] == fs) & (mean["setting"] != ADOPTED)]
        pair = mean[(mean["model_type"] == mt) & (mean["features"] == "R1 - R0d")]
        mad, minb = float(d["delta_vs_adopted"].abs().max()), int(pair["n_folds_better"].min())
        vrows.append(dict(model_type=mt, features=fs, max_abs_delta_auc=mad, min_folds_better_r1_minus_r0d=minb,
                          verdict=cell_verdict(mad, minb)))
    vt = pd.DataFrame(vrows)
    vt = pd.concat([vt, pd.DataFrame([dict(model_type="all", features="all", verdict=overall_verdict(vt["verdict"]))])],
                   ignore_index=True)
    vt["min_folds_better_r1_minus_r0d"] = vt["min_folds_better_r1_minus_r0d"].astype("Int64")   # 整數（總列是缺失）
    return cv, vt


def main() -> None:
    t0 = time.time()
    df = load_r1_frame()
    tv = df[df["role"] == "trainval"].reset_index(drop=True)          # 明確丟掉 test 列
    assert (tv["role"] == "trainval").all() and len(tv) == 30516
    assert ITEM_M == {"47": 5, "311": 20}                              # 採用值沒被動過
    print(f"[m_sens] trainval {len(tv):,} 列（丟掉 test {len(df) - len(tv):,} 列）", flush=True)

    all_auc = {}
    for name, m47, m311 in SETTINGS:
        all_auc[name] = run_setting(tv, m47, m311)
        if name == ADOPTED:
            check_adopted_matches_existing(all_auc[name])
        for cell, s in all_auc[name].items():
            print(f"[m_sens] {name:8s} (m47={m47}, m311={m311}) {cell[0]} {cell[1]:3s} A ROC-AUC "
                  f"{s.mean():.4f}   (累計 {time.time() - t0:.0f}s)", flush=True)

    cv, vt = build_tables(all_auc)
    cv.to_csv(OUT_CV, index=False)
    vt.to_csv(OUT_VERDICT, index=False)
    pd.set_option("display.width", 200)
    print("\n=== 各設定的 A ROC-AUC 平均、相對採用值的差 ===")
    print(cv[cv["fold"] == "mean"].drop(columns=["fold", "metric"]).to_string(index=False, float_format=lambda x: f"{x:+.5f}"))
    print("\n=== 判準 ===")
    print(vt.to_string(index=False))
    assert not ({"user_id", "school_id"} & (set(cv.columns) | set(vt.columns)))      # 只有彙總
    print(f"\n[m_sens] 完成（{time.time() - t0:.0f}s）；已寫出 {OUT_CV}、{OUT_VERDICT}", flush=True)


if __name__ == "__main__":
    main()
