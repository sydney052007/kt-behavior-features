"""W1：從修正版與原版各篩出技能 47、311。只印彙總統計。

輸出（都在 data/interim/，不進 git）：
  corrected_s47_311.csv  修正版中 skill_id 含 47 或 311 的「所有列」（含鷹架題、多技能題）
                         → 特徵管線要用鷹架題當歷史，所以不先濾掉 original==0
  original_s47_311.csv   原版中 skill_id 為 47 或 311 的所有列（含重複 order_id）→ leakage 稽核用
"""
import pandas as pd

SKILLS = {"47", "311"}

def tokens(s):
    return set(str(s).split("_")) if pd.notna(s) else set()

# ---- 修正版（skill_id 是字串，多技能以 '_' 連接）----
c = pd.read_csv("data/raw/corrected_collapsed.csv", low_memory=False)
c["skill_id"] = c["skill_id"].astype("string")
hit = c["skill_id"].map(lambda s: bool(tokens(s) & SKILLS))
cs = c[hit].copy()
cs.to_csv("data/interim/corrected_s47_311.csv", index=False)
print(f"[修正版] 全部 {len(c):,} 列 → 含 47/311 的列 {len(cs):,}")
print("  skill_id 取值（前 10）：")
print(cs["skill_id"].value_counts().head(10).to_string())

# ---- 原版（skill_id 是數字，每個技能一列）----
o = pd.read_csv("data/raw/original.csv", low_memory=False)
o["skill_id_num"] = pd.to_numeric(o["skill_id"], errors="coerce")
os_ = o[o["skill_id_num"].isin([47, 311])].copy()
os_["skill_id"] = os_["skill_id_num"].astype(int)  # 原版 skill_id 是單一整數
os_ = os_.drop(columns="skill_id_num")
os_.to_csv("data/interim/original_s47_311.csv", index=False)
print(f"\n[原版] 全部 {len(o):,} 列 → skill_id 為 47/311 的列 {len(os_):,}")
print(os_["skill_id"].value_counts().to_string())

# ---- W1 決策（11/3）：丟掉多技能列與鷹架列，只留「單一技能、original==1」的主題目 ----
# 產出 data/interim/main_s47_311.csv：特徵管線的輸入（目標與歷史都用這份）
cs = pd.read_csv("data/interim/corrected_s47_311.csv", low_memory=False)
cs["skill_id"] = cs["skill_id"].astype(str)
multi = cs["skill_id"].str.contains("_")
scaf = (cs["original"] == 0) & ~multi
main = cs[(cs["original"] == 1) & ~multi].copy()
assert set(main["skill_id"]) == {"47", "311"}
main.to_csv("data/interim/main_s47_311.csv", index=False)
print(f"\n[W1 決策] 修正版 47/311 {len(cs):,} 列 → 丟多技能 {int(multi.sum())} 列、鷹架 {int(scaf.sum()):,} 列 → 主題目 {len(main):,} 列")
print("  各技能列數：", main["skill_id"].value_counts().to_dict(), "| 學生：", main["user_id"].nunique())
# 丟掉鷹架列「損失」多少歷史資訊
ks = ["user_id", "skill_id"]
allr = cs[~multi].sort_values(ks + ["order_id"]).copy()
allr["n_scaf_before"] = allr.groupby(ks)["original"].transform(lambda s: (s == 0).cumsum())
tgt = allr[allr["original"] == 1]
print(f"  歷史中含鷹架列的目標列：{(tgt['n_scaf_before'] > 0).sum():,}（{(tgt['n_scaf_before'] > 0).mean():.1%}）；"
      f"有鷹架列的學生×技能序列：{allr[allr['original']==0].groupby(ks).ngroups}／{main.groupby(ks).ngroups}")
print("  ─ 其中技能 311：", f"{(tgt[tgt.skill_id=='311']['n_scaf_before'] > 0).mean():.1%} 的目標列歷史含鷹架")
