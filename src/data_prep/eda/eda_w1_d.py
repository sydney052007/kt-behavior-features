"""W1 EDA（D）：原版重複列、原版 vs 修正版對照、opportunity 落差來源、鷹架順序（乾淨版）。只輸出彙總。"""
import pandas as pd, numpy as np
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
o_all = pd.read_csv("data/raw/original.csv", low_memory=False)
print("== 0. 原版整檔 ==", o_all.shape)
print("order_id 缺失：", o_all.order_id.isna().sum(), "| user_id 缺失：", o_all.user_id.isna().sum(),
      "| 完全空白列：", o_all.isna().all(axis=1).sum())
print("order_id 重複列（第 2 次起算）：", o_all.order_id.duplicated().sum(), "| 完全相同列：", o_all.duplicated().sum())
print("order_id 唯一值：", o_all.order_id.nunique(), "（官方說明頁的唯一 order_id 數 = 346,860）")
print("最後 1 列的 order_id 是否與其他列重複：", bool(o_all.order_id.duplicated(keep=False).iloc[-1]),
      "| 第 1 列：", bool(o_all.order_id.duplicated(keep=False).iloc[0]))

o = pd.read_csv("data/interim/original_s47_311.csv", low_memory=False)
c = pd.read_csv("data/interim/corrected_s47_311.csv", low_memory=False)
c["skill_id"] = c["skill_id"].astype(str)
o["skill_id"] = o["skill_id"].astype(int).astype(str)
print("\n== 1. 47+311 的原版重複 ==")
for k in ("47", "311"):
    s = o[o.skill_id == k]
    sz = s.groupby("order_id").size()
    print(f"技能 {k}：原版 {len(s):,} 列，唯一 order_id {s.order_id.nunique():,}，多出的重複列 {len(s)-s.order_id.nunique():,}，"
          f"重複組大小分布 {sz.value_counts().sort_index().to_dict()}；完全相同列 {s.duplicated().sum():,}")
oo = o[o.original == 1]
for k in ("47", "311"):
    s = oo[oo.skill_id == k]
    print(f"  original==1：{k}：{len(s):,} 列，唯一 order_id {s.order_id.nunique():,}")
# 重複組內哪些欄位不同
dup_ids = o[o.duplicated("order_id", keep=False)]
print("重複組涉及的列：", len(dup_ids), "(占 47+311 原版", round(len(dup_ids) / len(o), 3), ")，涉及學生", dup_ids.user_id.nunique())
diffcols = {}
for col in o.columns:
    if col in ("order_id",): continue
    diffcols[col] = int((dup_ids.groupby("order_id")[col].nunique(dropna=False) > 1).sum())
print("重複組中「該欄位值不一致」的組數（非 0 的）：", {k: v for k, v in diffcols.items() if v})
print("重複組數：", dup_ids.order_id.nunique())

print("\n== 2. 原版與修正版對照（以 order_id）==")
for k in ("47", "311"):
    so = set(o[o.skill_id == k].order_id); sc = set(c[c.skill_id.str.split("_").map(lambda t: k in t)].order_id)
    print(f"技能 {k}：原版 order_id {len(so):,}｜修正版 {len(sc):,}｜交集 {len(so & sc):,}｜只在原版 {len(so - sc):,}｜只在修正版 {len(sc - so):,}")
ob = o.drop_duplicates("order_id").set_index("order_id"); cb = c.set_index("order_id")
com = ob.index.intersection(cb.index)
print("共同 order_id：", len(com))
for col in ["correct", "attempt_count", "hint_count", "first_action", "ms_first_response", "original", "user_id", "problem_id", "opportunity", "opportunity_original"]:
    a, b = ob.loc[com, col], cb.loc[com, col]
    print(f"  {col}: 不一致 {int(((a != b) & ~(a.isna() & b.isna())).sum())}")

print("\n== 3. 序列層級：原版比修正版多多少列 vs opportunity 落差 ==")
cs = c[(c.original == 1) & (~c.skill_id.str.contains("_"))]
cs_n = cs.groupby(["user_id", "skill_id"]).size().rename("n_corr")
os_n = oo.groupby(["user_id", "skill_id"]).size().rename("n_orig")
m = pd.concat([cs_n, os_n], axis=1).fillna(0)
m["extra"] = m.n_orig - m.n_corr
print("原版比修正版多列的序列：", int((m.extra > 0).sum()), "/", len(m), "| 多出列數合計", int(m.extra.sum()),
      "| 修正版比原版多：", int((m.extra < 0).sum()))
print("多出列數分位數（>0 的序列）：", m[m.extra > 0].extra.describe().round(1).to_dict())
cs2 = cs.sort_values(["user_id", "skill_id", "order_id"]).copy()
cs2["pos"] = cs2.groupby(["user_id", "skill_id"]).cumcount() + 1
cs2["gap"] = cs2.opportunity_original - cs2.pos
gs = cs2.groupby(["user_id", "skill_id"]).gap.max().rename("max_gap")
mm = m.join(gs, how="inner")
bad = mm[mm.max_gap > 0]
print("opportunity 落差 >0 的序列：", len(bad), "| 這些序列 max_gap 與 extra 相等：", int((bad.max_gap == bad.extra).sum()),
      "| 其餘（extra<max_gap）：", int((bad.extra < bad.max_gap).sum()), "| extra=0：", int((bad.extra == 0).sum()))

print("\n== 4. 鷹架列與主題目順序（只看該學生該 assistment_id 恰有 1 個主題目者）==")
S = c[c.original == 0][["user_id", "assistment_id", "order_id"]].rename(columns={"order_id": "o_sc"})
Mn = c[c.original == 1][["user_id", "assistment_id", "order_id"]].rename(columns={"order_id": "o_main"})
cnt = Mn.groupby(["user_id", "assistment_id"]).size().rename("n_main").reset_index()
j = S.merge(Mn, on=["user_id", "assistment_id"]).merge(cnt, on=["user_id", "assistment_id"])
j1 = j[j.n_main == 1]
print("鷹架列", len(S), "→ 配到唯一主題目的 pair", len(j1), "| 主題目在前：", (j1.o_main < j1.o_sc).mean().round(3), "| 主題目在後：", (j1.o_main > j1.o_sc).mean().round(3))
