"""W1 EDA（A）：重現設計修訂的子集數字、first_action、correct 的定義、order/opportunity。
只輸出彙總統計。"""
import pandas as pd, numpy as np
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)

c = pd.read_csv("data/interim/corrected_s47_311.csv", low_memory=False)
c["skill_id"] = c["skill_id"].astype(str)
print("== 0. 子集總覽（修正版，含 47/311 的所有列）==", len(c))
print("order_id 重複：", c.order_id.duplicated().sum(), "| 完全重複列：", c.duplicated().sum())
print("original 取值：\n", c.groupby(["skill_id", "original"]).size().unstack(fill_value=0).to_string())

T = c[(c.original == 1) & (~c.skill_id.str.contains("_"))].copy()
print("\n== 0b. 重現設計修訂第 2 節（單一技能、original==1）==")
g = T.groupby("skill_id")
print(pd.DataFrame({"students": g.user_id.nunique(), "rows": g.size(),
                    "schools": g.school_id.nunique(), "classes": g.student_class_id.nunique(),
                    "teachers": g.teacher_id.nunique(), "assign": g.assignment_id.nunique(),
                    "seq_ids": g.sequence_id.nunique(), "problems": g.problem_id.nunique(),
                    "wrong": (1 - g.correct.mean()).round(3)}).to_string())
print("合計列數", len(T), "| 兩技能都做的學生：",
      len(set(T[T.skill_id=="47"].user_id) & set(T[T.skill_id=="311"].user_id)),
      "| 全部學生", T.user_id.nunique())

print("\n== 1. correct 的定義：與 attempt_count / hint_count / first_action 的關係（目標列）==")
print("correct x attempt_count:\n", pd.crosstab(T.attempt_count.clip(upper=4), T.correct).to_string())
print("correct x (hint_count>0):\n", pd.crosstab(T.hint_count > 0, T.correct).to_string())
print("\n== 2. first_action（目標列，兩技能）==")
print(T.first_action.value_counts(dropna=False).to_string())
print("first_action x correct:\n", pd.crosstab(T.first_action, T.correct, dropna=False).to_string())
print("first_action x hint_count>0:\n", pd.crosstab(T.first_action, T.hint_count > 0).to_string())
print("first_action x attempt_count(clip 4):\n", pd.crosstab(T.first_action, T.attempt_count.clip(upper=4)).to_string())
print("first_action x bottom_hint:\n", pd.crosstab(T.first_action, T.bottom_hint.fillna(-1)).to_string())
print("hint_count>0 但 first_action==0 的列：", ((T.hint_count > 0) & (T.first_action == 0)).sum())
print("hint_total 分布（前幾名）：\n", T.hint_total.value_counts().head(6).to_string())
print("hint_count 分布：\n", T.hint_count.clip(upper=5).value_counts().sort_index().to_string())
print("ms_first_response：缺失", T.ms_first_response.isna().sum(), "| <=0", (T.ms_first_response <= 0).sum(),
      "| >1h", (T.ms_first_response > 3.6e6).sum())
print(T.ms_first_response.quantile([.01, .25, .5, .75, .95, .99]).round(0).to_string())

print("\n== 3. 順序：order_id vs opportunity_original ==")
T = T.sort_values(["user_id", "skill_id", "order_id"])
T["rank_in_seq"] = T.groupby(["user_id", "skill_id"]).cumcount() + 1
print("opportunity_original == 依 order_id 排序的名次：", (T.opportunity_original == T.rank_in_seq).mean().round(4))
d = (T.opportunity_original - T.rank_in_seq)
print("差值分布：\n", d.describe().round(2).to_string())
print("首列 opportunity_original 分布（序列第一列）：\n",
      T[T.rank_in_seq == 1].opportunity_original.clip(upper=5).value_counts().sort_index().to_string())
# 用 opportunity_original 自己排序時，order_id 是否單調
T2 = T.sort_values(["user_id", "skill_id", "opportunity_original", "order_id"])
bad = T2.groupby(["user_id", "skill_id"]).order_id.apply(lambda s: (s.diff().dropna() < 0).sum())
print("依 opportunity_original 排序時，order_id 倒退的次數總計：", int(bad.sum()),
      "| 涉及序列：", int((bad > 0).sum()), "/", len(bad))
print("opportunity_original 有重複（同序列同值）的列：",
      T.duplicated(["user_id", "skill_id", "opportunity_original"]).sum())
print("opportunity_original 缺失：", T.opportunity_original.isna().sum())
# 序列在 order_id 上是否連續（中間沒有該學生其他技能的列）
allc = c.sort_values(["user_id", "order_id"])
allc["blk"] = (allc.skill_id != allc.groupby("user_id").skill_id.shift()).cumsum()
runs = allc.groupby(["user_id", "skill_id"]).blk.nunique()
print("同學生同技能被別的列切成幾段（含鷹架列；段數分布）：\n", runs.value_counts().sort_index().to_string())
