"""R1r 判準函式 r1r_verdict 的測試。可直接用 python3 tests/experiments/variant_r1r/test_r1r_verdict.py 跑。

判準（跑之前寫死）：RF 的 A，R1r − R1 平均 ΔAUC ≥ +0.002 且至少 4/5 個 fold 較好。
測邊界；最後用已存好的 results/02_analysis/r1r_verdict.csv（彙總檔）對一次。
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "src"))
import _setup_path  # noqa: F401,E402  # 把 src/ 底下所有子資料夾加進 import 路徑
from run_r1r import MIN_DELTA, MIN_FOLDS, r1r_verdict  # noqa: E402  # import 只載入函式與常數，不讀資料、不寫檔

GOOD, BAD = "R1 的改良版", "沒有明顯改善的變體"


def test_thresholds_are_the_registered_ones():
    assert MIN_DELTA == 0.002 and MIN_FOLDS == 4


def test_boundaries():
    assert r1r_verdict(0.002, 4) == GOOD          # 剛好等於門檻（兩個條件都是 ≥）
    assert r1r_verdict(0.00199, 5) == BAD         # ΔAUC 差一點點
    assert r1r_verdict(0.01, 3) == BAD            # 較好的 fold 數不夠
    assert r1r_verdict(0.01, 4) == GOOD
    assert r1r_verdict(-0.01, 5) == BAD           # ΔAUC 為負：即使 fold 數夠也不算
    assert r1r_verdict(0.01, 5) == GOOD


def test_matches_saved_verdict_file():
    v = pd.read_csv("results/02_analysis/r1r_verdict.csv", float_precision="round_trip")
    assert len(v) == 1
    r = v.iloc[0]
    assert r1r_verdict(r["delta_auc"], r["n_folds_better"]) == r["verdict"]


if __name__ == "__main__":
    tests = [test_thresholds_are_the_registered_ones, test_boundaries, test_matches_saved_verdict_file]
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except AssertionError as e:
            print(f"FAIL {t.__name__}: {e}")
            raise
    print(f"\n全部 {len(tests)} 個測試通過")
