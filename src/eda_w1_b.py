"""W1 EDA（B）：精熟規則、位置、第 1 題、多技能、鷹架題、opportunity 不一致。只輸出彙總。"""
import pandas as pd, numpy as np
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

c = pd.read_csv("data/interim/corrected_s47_311.csv", low_memory=False)
c["skill_id"] = c["skill_id"].astype(str)
T = c[(c.original == 1) & (~c.skill_id.str.contains("_"))].sort_values(["user_id", "skill_id", "order_id"]).copy()
T["pos"] = T.groupby(["user_id", "skill_id"]).cumcount() + 1
T["L"] = T.groupby(["user_id", "skill_id"]).pos.transform("max")
T["wrong"] = 1 - T.correct

# run of consecutive corrects ending at each row
def run_len(s):
    out, r = [], 0
    for v in s:
        r = r + 1 if v == 1 else 0
        out.append(r)
    return out
T["run"] = T.groupby(["user_id", "skill_id"]).correct.transform(lambda s: pd.Series(run_len(s.values), index=s.index))

seq = T.groupby(["user_id", "skill_id"]).agg(L=("pos", "max"), end_run=("run", "last"), max_run=("run", "max"))
first3 = T[T.run >= 3].groupby(["user_id", "skill_id"]).pos.min().rename("first3")
seq = seq.join(first3)
seq["rows_after_first3"] = seq.L - seq.first3
print("== 1. 精熟規則（單一技能、original==1，序列 = 學生×技能）==")
print("序列數：", len(seq))
print("序列結尾連續答對數 end_run 分布：\n", seq.end_run.clip(upper=6).value_counts(normalize=True).sort_index().round(3).to_string())
print("出現過 ≥3 連對的序列：", seq.first3.notna().mean().round(3),
      "| 其中第一次 3 連對後還有繼續作答（rows_after_first3>0）：", (seq.rows_after_first3 > 0).sum(),
      "（占全部 3 連對序列", ((seq.rows_after_first3 > 0).sum() / seq.first3.notna().sum()).round(3), "）")
print("3 連對後還繼續的列數分布（只看 >0）：\n", seq[seq.rows_after_first3 > 0].rows_after_first3.describe().round(1).to_string())
print("最長連對數 max_run 分布：\n", seq.max_run.clip(upper=6).value_counts().sort_index().to_string())
print("未達 3 連對的序列（沒精熟就停）：", int((seq.first3.isna()).sum()), "條；其長度分布：\n",
      seq[seq.first3.isna()].L.describe().round(1).to_string())
print("序列長度 L 計數（1–25）：\n", seq.L.clip(upper=25).value_counts().sort_index().to_string())
# 長度剛好 = first3 的比例（3 連對那一題就結束）
print("L == first3（第一次 3 連對就是最後一題）占 3 連對序列：",
      (seq.L == seq.first3).sum() / seq.first3.notna().sum())
for k in (11, 6):
    sub = seq[seq.L >= k]
    # 前 k-1 題內是否已經出現過 3 連對
    pre = T[(T.pos <= k - 1)].merge(sub.reset_index()[["user_id", "skill_id"]], on=["user_id", "skill_id"])
    has3 = pre.groupby(["user_id", "skill_id"]).run.max().ge(3)
    print(f"L>={k} 的序列 {len(sub)} 條；前 {k-1} 題內已出現 3 連對的序列：{int(has3.sum())}")

print("\n== 1b. 位置 vs 答錯率（單一技能目標列）==")
bins = pd.cut(T.pos, [0, 1, 4, 10, 10**6], labels=["1", "2-4", "5-10", "11+"])
print(T.groupby(bins, observed=True).agg(rows=("wrong", "size"), wrong=("wrong", "mean")).round(3).to_string())
for k in (11, 6):
    r = T[(T.pos == k)]
    print(f"第 {k} 列（序列長度≥{k}）：n={len(r)}，答錯率={r.wrong.mean():.3f}；"
          f"同批序列的全部列答錯率={T[T.L>=k].wrong.mean():.3f}；全部列={T.wrong.mean():.3f}")
print("答錯率 by pos 1..15：\n", T[T.pos <= 15].groupby("pos").wrong.mean().round(3).to_string())

print("\n== 2. 第 1 題沒有歷史 ==")
print("pos==1 的列：", (T.pos == 1).sum(), f"= {(T.pos==1).mean():.3f} 的目標列；答錯率", T[T.pos==1].wrong.mean().round(3))
print("L==1 的序列：", int((seq.L == 1).sum()), f"（{(seq.L==1).mean():.3f}）；L<=2：{(seq.L<=2).mean():.3f}")
# 第一列之前有沒有鷹架／多技能列（歷史存在但被我們濾掉）
allr = c.sort_values(["user_id", "order_id"]).copy()
firsts = T[T.pos == 1][["user_id", "skill_id", "order_id"]].rename(columns={"order_id": "o1"})
m = allr.merge(firsts, on="user_id")
m = m[(m.skill_id_x if "skill_id_x" in m else m.skill_id) is not None] if False else m
prior = m[m.order_id < m.o1]
sk = "skill_id_x" if "skill_id_x" in prior else "skill_id"
print("第一列之前，同學生在 47/311 範圍內其他列（任何技能）的筆數>0 的序列數：",
      prior.groupby(["user_id", "skill_id_y" if "skill_id_y" in prior else "skill_id"]).size().shape[0], "/", len(firsts))

print("\n== 3. 多技能列 ==")
M = c[c.skill_id.str.contains("_")]
print("含 47/311 的多技能列：", len(M), M.skill_id.value_counts().to_dict(), "| original：", M.original.value_counts().to_dict())
us = set(M.user_id)
for k in ("47", "311"):
    ss = T[(T.skill_id == k) & T.user_id.isin(us)]
    print(f"  這些學生在單一技能 {k} 的列：{len(ss)}")
print("  多技能列占（單一 + 多技能）目標列：", len(M) / (len(T) + len(M)))

print("\n== 4. 鷹架題（original==0）==")
S = c[c.original == 0]
print(S.groupby("skill_id").size().to_dict(), "| 有鷹架的學生×技能序列：", S.groupby(["user_id", "skill_id"]).ngroups)
allr = c.sort_values(["user_id", "order_id"]).reset_index(drop=True)
allr["prev_orig"] = allr.groupby("user_id").original.shift()
allr["prev_correct"] = allr.groupby("user_id").correct.shift()
allr["prev_fa"] = allr.groupby("user_id").first_action.shift()
sc = allr[allr.original == 0]
print("鷹架列的前一列（同學生按 order_id）：original=1 且答錯：", ((sc.prev_orig == 1) & (sc.prev_correct == 0)).mean().round(3),
      "| original=0（連續鷹架）：", (sc.prev_orig == 0).mean().round(3),
      "| original=1 且答對：", ((sc.prev_orig == 1) & (sc.prev_correct == 1)).mean().round(3))
print("鷹架列的 correct 分布：", sc.correct.value_counts().to_dict())
print("first_action by skill（目標列）：\n", pd.crosstab(T.skill_id, T.first_action).to_string())
main = allr[allr.original == 1].copy()
main["next_orig"] = allr.groupby("user_id").original.shift(-1).loc[main.index]
main["next_user_same"] = True
print("主題目之後緊接鷹架列的比例 by first_action：\n", main.groupby("first_action").next_orig.apply(lambda s: (s == 0).mean()).round(3).to_string())
print("主題目之後緊接鷹架列的比例 by correct：\n", main.groupby("correct").next_orig.apply(lambda s: (s == 0).mean()).round(3).to_string())

print("\n== 5. opportunity_original 與 order_id 名次不一致 ==")
T["gap"] = T.opportunity_original - T.pos
mis = T[T.gap != 0]
print("不一致列：", len(mis), "| 涉及序列：", mis.groupby(["user_id", "skill_id"]).ngroups, "| by skill：", mis.skill_id.value_counts().to_dict())
print("gap 分位數（不一致列）：\n", mis.gap.describe().round(1).to_string())
sq = T.groupby(["user_id", "skill_id"]).gap.agg(["min", "max"])
print("序列內 gap 單調不減？", (T.groupby(["user_id", "skill_id"]).gap.apply(lambda s: s.diff().dropna().ge(0).all())).mean().round(4))
print("gap 在序列第 1 列 ≠ 0 的序列：", int((T[T.pos == 1].gap != 0).sum()))
