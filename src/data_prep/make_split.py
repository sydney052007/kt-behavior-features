"""W1：以學生為單位切 test 集與 5-fold。只輸出彙總，學生對照表存在 data/interim/（不進 git）。

規則（設計修訂第 3 節）：
  1. 分組單位 = user_id，跨技能 47、311 一起切（443 位學生兩個技能都做）。
  2. 先留 15% 學生當 test（最後只跑一次）；其餘 85% 做 5-fold GroupKFold。
  3. 隨機種子固定為 SEED，不為了讓某個分數好看而換種子。
  4. test 用「學生的總題數分箱」分層抽樣，避免抽到幾個答 100 題以上的學生就讓列數比例偏離。
     分層只用題數，不用答對與否。
"""
import os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GroupKFold

SEED = 42
OUT = "data/interim/split_users.csv"
# 切分一旦產生就是整個專案的基準，不可無意間被覆蓋（不同版本的 scikit-learn 可能切出不同結果）
if os.path.exists(OUT) and "--force" not in sys.argv:
    sys.exit(f"{OUT} 已存在，為避免覆蓋既有切分而停止。真的要重切請加 --force（並通知設計討論）。")
TEST_FRAC = 0.15
N_FOLDS = 5

m = pd.read_csv("data/interim/main_s47_311.csv", low_memory=False)
m["skill_id"] = m["skill_id"].astype(str)
users = m.groupby("user_id").size().rename("n_rows").reset_index()
users["bin"] = pd.qcut(users["n_rows"], 5, labels=False, duplicates="drop")

# ---- 1. test ----
trainval_u, test_u = train_test_split(users["user_id"], test_size=TEST_FRAC,
                                      stratify=users["bin"], random_state=SEED)
users["role"] = np.where(users["user_id"].isin(set(test_u)), "test", "trainval")
users["fold"] = -1

# ---- 2. 其餘學生做 5-fold（以學生分組；每位學生一個群）----
tv = users[users.role == "trainval"].reset_index(drop=True)
gkf = GroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for f, (_, va_idx) in enumerate(gkf.split(tv, groups=tv["user_id"])):
    tv.loc[va_idx, "fold"] = f
users = users.drop(columns="fold").merge(tv[["user_id", "fold"]], on="user_id", how="left")
users["fold"] = users["fold"].fillna(-1).astype(int)
users[["user_id", "role", "fold"]].to_csv(OUT, index=False)

# ---- 3. 驗證與彙總 ----
d = m.merge(users[["user_id", "role", "fold"]], on="user_id")
d["wrong"] = 1 - d["correct"]
seq = d.groupby(["user_id", "skill_id"]).size().rename("L").reset_index().merge(users[["user_id", "role", "fold"]], on="user_id")
print(f"SEED={SEED}  test={TEST_FRAC}  folds={N_FOLDS}")
# (a) 每位學生恰好一個身分；test 與 fold 無交集；fold 互斥且涵蓋全部 trainval
assert users["user_id"].is_unique
assert set(users[users.role == "test"].user_id).isdisjoint(set(users[users.role == "trainval"].user_id))
assert (users[users.role == "trainval"].fold.between(0, N_FOLDS - 1)).all()
assert (users[users.role == "test"].fold == -1).all()
assert len(d) == len(m) and d["role"].notna().all()
# 兩技能都做的學生不會被拆開：每位學生只有一個 role / fold
assert d.groupby("user_id")[["role", "fold"]].nunique().max().max() == 1
print("檢查通過：每位學生只屬於一個身分；test 與 trainval 學生無交集；兩技能都做的學生沒被拆開")

tot_u, tot_r = users.user_id.nunique(), len(d)
t = d[d.role == "test"]
print(f"\n[test] 學生 {t.user_id.nunique()}（{t.user_id.nunique()/tot_u:.1%}）列 {len(t):,}（{len(t)/tot_r:.1%}）答錯率 {t.wrong.mean():.3f}")
for k in ("47", "311"):
    a = d[d.skill_id == k]; b = t[t.skill_id == k]
    print(f"  技能 {k}：test 學生 {b.user_id.nunique()}/{a.user_id.nunique()}（{b.user_id.nunique()/a.user_id.nunique():.1%}）列 {len(b):,}/{len(a):,}（{len(b)/len(a):.1%}）")
print("\n[5-fold]（trainval 內）")
tv_d = d[d.role == "trainval"]
rows = []
for f in range(N_FOLDS):
    g = tv_d[tv_d.fold == f]; s = seq[(seq.role == "trainval") & (seq.fold == f)]
    rows.append({"fold": f, "學生": g.user_id.nunique(), "列": len(g), "答錯率": round(g.wrong.mean(), 3),
                 "序列": len(s), "序列≥6": int((s.L >= 6).sum()), "序列≥11": int((s.L >= 11).sum()),
                 "47列": int((g.skill_id == "47").sum()), "311列": int((g.skill_id == "311").sum())})
print(pd.DataFrame(rows).to_string(index=False))
print("\nfold 列數最大/最小 =", round(max(r["列"] for r in rows) / min(r["列"] for r in rows), 2),
      "| 答錯率範圍", min(r["答錯率"] for r in rows), "–", max(r["答錯率"] for r in rows))
print(f"test 序列 {len(seq[seq.role=='test'])}，其中 ≥6：{int((seq[seq.role=='test'].L>=6).sum())}，≥11：{int((seq[seq.role=='test'].L>=11).sum())}")
print("trainval 列數", len(tv_d), "答錯率", round(tv_d.wrong.mean(), 3), "| 全部答錯率", round(d.wrong.mean(), 3))
