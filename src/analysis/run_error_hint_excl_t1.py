"""錯誤分析「提示歷史」分段的描述性補充：排除 t=1 的列後，hint_rate = 0 與 > 0 兩段的 ΔAUC（R1 − R0d）。

    python3 src/analysis/run_error_hint_excl_t1.py > results/03_process/error_hint_excl_t1_run.log 2>&1
這是事後加的描述性補充，不是事先登記的分析（登記的分段定義見 CLAUDE.md「錯誤分析規則」，結果在 error_segments.csv，
不被這個檔案覆蓋）。做法與原本完全相同：同一組 OOF（LR、RF × R0d、R1）、同一個配對 cluster bootstrap
（以學生為單位，2000 次、種子 42）；只是把 t = 1 的列從分析中拿掉。
輸出：results/02_analysis/error_hint_excl_t1.csv（只有彙總；沒有逐列資料）。
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # src/
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from run_error_analysis import OUT, build_runs, paired_delta_ci, pr, reproduction_check, roc  # noqa: E402


def main() -> None:
    runs = build_runs()
    rep = reproduction_check(runs)
    assert rep["abs_diff"].max() < 1e-9, "OOF 與既有結果不一致"
    print(f"[重現檢查] A 的 5 fold 平均與既有結果一致（最大差 {rep['abs_diff'].max():.1e}）")

    oof = runs[("LR", "R1")]["oof"]                                  # 驗證列特徵（hint_rate、t、has_history）兩個模型相同
    t1, has_hist = oof["t"] == 1, oof["has_history"] == 1
    assert (t1 == ~has_hist).all()
    # (1) 核對：t>1 的 hint_rate 是『原始歷史比例』（過去列中有要提示的比例 = 整數 / (t−1)），不是填補值；t=1 才是填補值
    hist = oof.loc[~t1]
    k = hist["hint_rate"] * (hist["t"] - 1)
    assert np.allclose(k, np.round(k), atol=1e-9), "t>1 的 hint_rate 不是 整數/(t−1)"
    fill_values = oof.loc[t1, "hint_rate"].round(10).unique()
    print(f"[核對] t>1 的 hint_rate 全部是 整數/(t−1)（原始歷史比例）；t=1 的 hint_rate 只有 {len(fill_values)} 個值（各 fold 的訓練集平均填補值，"
          f"範圍 {oof.loc[t1, 'hint_rate'].min():.4f}–{oof.loc[t1, 'hint_rate'].max():.4f}）")

    # (2) t=1 列在原本分段中的歸屬（原本規則：有歷史且 hint_rate > 0 → '>0'；其餘含 t=1 → '= 0'）
    orig_pos = has_hist & (oof["hint_rate"] > 0)
    n_t1 = int(t1.sum())
    t1_in_pos, t1_in_zero = int((t1 & orig_pos).sum()), int((t1 & ~orig_pos).sum())
    print(f"[歸屬] t=1 共 {n_t1:,} 列：分到 'hint_rate>0' {t1_in_pos:,} 列、分到 'hint_rate=0 (incl. t=1)' {t1_in_zero:,} 列")
    print(f"[歸屬] 'hint_rate=0 (incl. t=1)' 共 {int((~orig_pos).sum()):,} 列 = t=1 的 {t1_in_zero:,} + t>1 且 hint_rate=0 的 {int((~orig_pos & ~t1).sum()):,}")

    # (3) 排除 t=1 的重算
    rows = []
    segs = [("hint_rate=0 (excl. t=1)", (~t1) & (oof["hint_rate"] == 0)), ("hint_rate>0", (~t1) & (oof["hint_rate"] > 0))]
    assert sum(int(m.sum()) for _, m in segs) == len(oof) - n_t1
    for mt in ["LR", "RF"]:
        old, new = runs[(mt, "R0d")]["oof"], runs[(mt, "R1")]["oof"]
        assert old["order_id"].equals(new["order_id"])
        for name, m in segs:
            mm = m.to_numpy()
            y = (1 - new.loc[mm, "correct"]).to_numpy()
            s_old, s_new = (1 - old.loc[mm, "p_correct"]).to_numpy(), (1 - new.loc[mm, "p_correct"]).to_numpy()
            base = dict(model_type=mt, segment_type="hint_history_excl_t1", segment=name, n_rows=int(mm.sum()),
                        n_students=int(new.loc[mm, "user_id"].nunique()), wrong_rate=float(y.mean()))
            rows.append(dict(base, model="R0d", metric="roc_auc", value=roc(y, s_old)))
            rows.append(dict(base, model="R1", metric="roc_auc", value=roc(y, s_new)))
            rows.append(dict(base, model="R0d", metric="pr_auc", value=pr(y, s_old)))
            rows.append(dict(base, model="R1", metric="pr_auc", value=pr(y, s_new)))
            d, lo, hi = paired_delta_ci(y, s_old, s_new, new.loc[mm, "user_id"].to_numpy())
            rows.append(dict(base, model="R1 - R0d", metric="delta_roc_auc", value=d, ci_low=lo, ci_high=hi))
            print(f"[重算] {mt} {name:24s} n={base['n_rows']:6d} wrong={base['wrong_rate']:.4f}  ΔAUC {d:+.4f} [{lo:+.4f}, {hi:+.4f}]", flush=True)
    out = pd.DataFrame(rows)
    out["note"] = "descriptive supplement (post hoc, not pre-registered); t=1 rows excluded"
    out.to_csv(OUT + "error_hint_excl_t1.csv", index=False)

    # 與原本結果並排（原本的檔案不改）
    orig = pd.read_csv(OUT + "error_segments.csv")
    orig = orig[(orig["segment_type"] == "hint_history") & (orig["model"] == "R1 - R0d") & (orig["metric"] == "delta_roc_auc")]
    print("\n=== 並排：原本（含 t=1 在 '= 0'）vs 排除 t=1 的 ΔAUC（R1 − R0d）===")
    for mt in ["LR", "RF"]:
        for oname, nname in [("hint_rate=0 (incl. t=1)", "hint_rate=0 (excl. t=1)"), ("hint_rate>0", "hint_rate>0")]:
            o = orig[(orig["model_type"] == mt) & (orig["segment"] == oname)].iloc[0]
            n = out[(out["model_type"] == mt) & (out["segment"] == nname) & (out["model"] == "R1 - R0d")].iloc[0]
            print(f"  {mt} {oname:24s} n={int(o['n_rows']):6d} ΔAUC {o['value']:+.4f} [{o['ci_low']:+.4f}, {o['ci_high']:+.4f}]   →   "
                  f"{nname:24s} n={int(n['n_rows']):6d} ΔAUC {n['value']:+.4f} [{n['ci_low']:+.4f}, {n['ci_high']:+.4f}]")


if __name__ == "__main__":
    main()
