"""隨機抽 2 位學生，印出他們每條序列前 10 列的 correct 與 R0 特徵，給人手算對照。

只印在終端機，不寫進任何檔案 —— 這是學生層級資料，不要貼進報告或看板
（CLAUDE.md 資料使用條款）。

用法：python3 src/peek_features.py [--seed N]
"""
import sys

import numpy as np
import pandas as pd

from features import SEQ, add_position, add_r0_features, load_main

seed = 0
if "--seed" in sys.argv:
    seed = int(sys.argv[sys.argv.index("--seed") + 1])

df = add_r0_features(add_position(load_main()))

rng = np.random.default_rng(seed)
# 優先挑有長序列（>=10 列）的學生，這樣真的印得出 10 列可對照
long_users = df.groupby("user_id")["t"].max()
eligible = long_users[long_users >= 10].index.to_numpy()
picked_users = rng.choice(eligible, size=2, replace=False)

print(f"（seed={seed}；只在終端機顯示，不要貼進報告或看板）\n")
pd.set_option("display.width", 120)

for uid in picked_users:
    print(f"===== user_id={uid} =====")
    sub = df[df["user_id"] == uid]
    for skill_id, g in sub.groupby("skill_id"):
        g = g.sort_values("t").head(10)
        print(f"-- skill_id={skill_id}（序列長度 {len(sub[sub.skill_id == skill_id])}，只顯示前 10 列）--")
        print(g[["order_id", "t", "correct", "c", "w", "log1p_c", "log1p_w",
                  "skill_311", "has_history"]].to_string(index=False))
        print()
