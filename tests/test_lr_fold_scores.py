"""LR 各 fold 絕對分數（r1_lr_folds.csv）的測試。可直接用 python3 tests/test_lr_fold_scores.py 跑。"""
import pandas as pd

F = "results/02_analysis/r1_lr_folds.csv"
CV = "results/02_analysis/r1_lr_cv.csv"
PAIRED = "results/02_analysis/r1_lr_paired.csv"


def test_means_and_stds_match_r1_lr_cv():
    d, ref = pd.read_csv(F), pd.read_csv(CV)
    ref = ref[ref["eval"] == "A"].set_index(["model", "metric"])
    assert len(d) == 15 and set(d["model"]) == {"R0", "R0d", "R1"} and sorted(d["fold"].unique()) == [0, 1, 2, 3, 4]
    for name in ["R0", "R0d", "R1"]:
        for m in ["roc_auc", "pr_auc"]:
            g = d[d["model"] == name][m]
            assert abs(g.mean() - ref.at[(name, m), "value"]) < 1e-9
            assert abs(g.std(ddof=1) - ref.at[(name, m), "std"]) < 1e-9


def test_fold_differences_match_paired_csv():
    """各 fold 的絕對 AUC 相減，必須等於 r1_lr_paired.csv 裡既有的逐 fold 差值。"""
    d = pd.read_csv(F).pivot(index="fold", columns="model", values="roc_auc")
    p = pd.read_csv(PAIRED)
    p = p[(p["eval"] == "A")].set_index("comparison")
    for comp, (new, old) in {"R0d - R0": ("R0d", "R0"), "R1 - R0d": ("R1", "R0d")}.items():
        for k in range(5):
            assert abs((d.at[k, new] - d.at[k, old]) - p.at[comp, f"fold{k}"]) < 1e-9, (comp, k)
    n = pd.read_csv(F).groupby("fold")["n_rows"].nunique()
    assert (n == 1).all()                                   # 三組模型的每個 fold 列數相同


if __name__ == "__main__":
    for t in [test_means_and_stds_match_r1_lr_cv, test_fold_differences_match_paired_csv]:
        t()
        print(f"PASS {t.__name__}")
    print("\n全部 2 個測試通過")
