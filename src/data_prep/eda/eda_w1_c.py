"""W1 EDA（C）：精熟規則（以 assignment 為單位重驗）、鷹架順序、年級欄位、學校規模、題組。只輸出彙總。"""
import pandas as pd, numpy as np
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
c = pd.read_csv("data/interim/corrected_s47_311.csv", low_memory=False)
c["skill_id"] = c["skill_id"].astype(str)
T = c[(c.original == 1) & (~c.skill_id.str.contains("_"))].sort_values(["user_id", "skill_id", "order_id"]).copy()
T["pos"] = T.groupby(["user_id", "skill_id"]).cumcount() + 1
T["L"] = T.groupby(["user_id", "skill_id"]).pos.transform("max")
T["wrong"] = 1 - T.correct

def run_len(v):
    out, r = [], 0
    for x in v:
        r = r + 1 if x == 1 else 0
        out.append(r)
    return out

print("== 1. 以 assignment_id 為單位重驗精熟規則 ==")
g = T.groupby(["user_id", "skill_id"])
print("每個學生×技能涉及的 assignment 數：\n", g.assignment_id.nunique().clip(upper=5).value_counts().sort_index().to_string())
print("每個學生×技能涉及的 sequence_id 數：\n", g.sequence_id.nunique().clip(upper=5).value_counts().sort_index().to_string())
for key in ("assignment_id", "sequence_id"):
    U = T.sort_values(["user_id", "skill_id", key, "order_id"]).copy()
    U["p"] = U.groupby(["user_id", "skill_id", key]).cumcount() + 1
    U["run"] = U.groupby(["user_id", "skill_id", key]).correct.transform(lambda s: pd.Series(run_len(s.values), index=s.index))
    sq = U.groupby(["user_id", "skill_id", key]).agg(L=("p", "max"), first3=("run", lambda s: (s.values >= 3).argmax() + 1 if (s.values >= 3).any() else np.nan),
                                                    end_run=("run", "last"))
    has = sq.first3.notna()
    print(f"\n-- 單位 = {key}：{len(sq)} 條；出現 3 連對 {has.mean():.3f}")
    print("   3 連對之後還繼續的比例：", ((sq.L > sq.first3)[has]).mean().round(3),
          "| 第一次 3 連對就是最後一題：", ((sq.L == sq.first3)[has]).mean().round(3))
    print("   結尾連對數 end_run 分布：", sq.end_run.clip(upper=5).value_counts(normalize=True).sort_index().round(3).to_dict())
    print("   3 連對後繼續的列數（>0）分位數：", sq[has & (sq.L > sq.first3)].eval("L-first3").describe().round(1).to_dict())

print("\n== 1c. 同一 assignment 內，連對 3 之後的下一題答對率 vs 一般 ==")
U = T.sort_values(["user_id", "skill_id", "assignment_id", "order_id"]).copy()
U["run"] = U.groupby(["user_id", "skill_id", "assignment_id"]).correct.transform(lambda s: pd.Series(run_len(s.values), index=s.index))
U["prev_run"] = U.groupby(["user_id", "skill_id", "assignment_id"]).run.shift()
print(U.groupby(U.prev_run.clip(upper=4)).wrong.agg(["size", "mean"]).round(3).to_string())

print("\n== 2. 鷹架列與其主題目的順序（同學生、同 assistment_id）==")
S = c[c.original == 0][["user_id", "assistment_id", "order_id", "skill_id"]].rename(columns={"order_id": "o_sc"})
Mn = c[c.original == 1][["user_id", "assistment_id", "order_id"]].rename(columns={"order_id": "o_main"})
j = S.merge(Mn, on=["user_id", "assistment_id"], how="left")
print("鷹架列：", len(S), "| 找得到同學生同 assistment_id 的主題目：", j.o_main.notna().mean().round(3))
jj = j.dropna(subset=["o_main"])
print("主題目 order_id < 鷹架 order_id 的比例：", (jj.o_main < jj.o_sc).mean().round(3), "| 反過來：", (jj.o_main > jj.o_sc).mean().round(3))

print("\n== 3. 年級／年齡 欄位 ==")
print("欄位：", list(c.columns))
cands = ["assignment_id", "sequence_id", "base_sequence_id", "student_class_id", "teacher_id", "school_id", "position", "type", "tutor_mode", "answer_type", "template_id"]
print({k: int(T[k].nunique()) for k in cands})

print("\n== 4. 各校人數（單一技能、original==1 目標列）==")
sch = T.groupby("school_id").agg(students=("user_id", "nunique"), rows=("user_id", "size"))
print("學校數：", len(sch))
print("學生數分位數：", sch.students.describe().round(1).to_dict())
print("學生數由大到小前 8：", sch.sort_values("students", ascending=False).students.head(8).tolist())
print("列數由大到小占比前 8：", (sch.sort_values("rows", ascending=False).rows.head(8) / sch.rows.sum()).round(3).tolist())
for t in (5, 10, 20, 30):
    print(f"  學生數 < {t} 的學校：{(sch.students < t).sum()} 間，合計學生 {sch[sch.students < t].students.sum()}，列 {sch[sch.students < t].rows.sum()}")
for k in ("47", "311"):
    s2 = T[T.skill_id == k].groupby("school_id").user_id.nunique()
    print(f"  技能 {k}：學校 {len(s2)}，<10 人 {(s2<10).sum()} 間，最大 {s2.max()}，中位數 {s2.median()}")
# 學生屬於幾所學校
print("學生對應學校數 >1 的學生：", int((T.groupby("user_id").school_id.nunique() > 1).sum()))
# 學校分組 5-fold 的可行性：貪婪分配，看各 fold 列數
rows = sch.rows.sort_values(ascending=False)
folds = [0] * 5
for r in rows:
    folds[int(np.argmin(folds))] += r
print("貪婪分成 5 fold 的列數：", folds, "| 最大/最小 =", round(max(folds) / min(folds), 2))
# 最大學校占比對 fold 的影響
print("最大學校占全部列：", round(rows.iloc[0] / rows.sum(), 3))

print("\n== 5. 題組（題型相近）==")
for key in ("sequence_id", "base_sequence_id", "student_class_id", "assignment_id"):
    for k in ("47", "311"):
        s3 = T[T.skill_id == k]
        vc = s3[key].value_counts(normalize=True)
        print(f"  skill {k} {key}: {s3[key].nunique()} 個, 前 3 名占列 {vc.head(3).round(3).tolist()}")
print("skill_name：", c.groupby("skill_id").skill_name.first().to_dict())
print("type 取值：", T.type.value_counts().to_dict(), "| tutor_mode：", T.tutor_mode.value_counts().to_dict(), "| answer_type：", T.answer_type.value_counts().to_dict())
print("template_id 每技能：", T.groupby("skill_id").template_id.nunique().to_dict())
