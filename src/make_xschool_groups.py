"""步驟 0：跨學校分組（只在 trainval 上）。輸出分組表，並把 學校 → 組別 → fold 存到 data/split_groups.csv。

    python3 src/make_xschool_groups.py
停損（規則在 CLAUDE.md）：併完最大組超過 trainval 列數的 40%，或組數少於 5 → 印出原因並結束，不寫檔。
data/split_groups.csv 已存在時不重新產生：只重新計算並確認與檔案內容完全相同。
輸出（results/，只有彙總）：xschool_groups.csv、xschool_folds.csv
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from xschool import (GROUPS_PATH, MAX_GROUP_SHARE, MIN_GROUPS, assign_group_folds, build_groups,  # noqa: E402
                     load_trainval_with_org)


def main() -> None:
    tv = load_trainval_with_org()
    s2g = build_groups(tv)
    tv["grp"] = tv["school_id"].map(s2g)
    g = tv.groupby("grp").agg(schools=("school_id", "nunique"), students=("user_id", "nunique"), rows=("correct", "size"))
    n_stu, n_rows = tv["user_id"].nunique(), len(tv)
    big = g.sort_values("rows", ascending=False).iloc[0]
    share = big["rows"] / n_rows
    print(f"原本 {tv['school_id'].nunique()} 所學校 → 併完 {len(g)} 組（含多所學校的組 {int((g.schools > 1).sum())} 個）")
    print(f"最大組：列 {big['rows']:,}（{100 * share:.1f}%）、學生 {big['students']}（{100 * big['students'] / n_stu:.1f}%）")
    if share > MAX_GROUP_SHARE or len(g) < MIN_GROUPS:
        print(f"[停損] 最大組占比 {share:.1%}（上限 {MAX_GROUP_SHARE:.0%}）或組數 {len(g)}（下限 {MIN_GROUPS}）→ 停下來回報，不繼續")
        sys.exit(1)

    gfold = assign_group_folds(tv, s2g)
    table = pd.DataFrame({"school_id": s2g.index, "group_id": s2g.to_numpy()})
    table["fold"] = table["group_id"].map(gfold).astype(int)
    if os.path.exists(GROUPS_PATH):
        old = pd.read_csv(GROUPS_PATH)
        assert old.equals(table), f"{GROUPS_PATH} 已存在，且與重新計算的結果不同（不覆蓋）"
        print(f"[ok] {GROUPS_PATH} 已存在，內容與重新計算的完全相同，沒有覆蓋")
    else:
        table.to_csv(GROUPS_PATH, index=False)
        print(f"[ok] 已寫出 {GROUPS_PATH}（{len(table)} 所學校 → {len(g)} 組 → 5 個 fold）")

    # ---- 彙總輸出
    def q(s):
        return dict(min=s.min(), p25=s.quantile(.25), median=s.median(), p75=s.quantile(.75), max=s.max())
    rows = [dict(item="學校數（原本）", value=tv["school_id"].nunique()), dict(item="組數（併完）", value=len(g)),
            dict(item="含多所學校的組數", value=int((g.schools > 1).sum())),
            dict(item="最大組占 trainval 列數", value=share), dict(item="最大組占 trainval 學生數", value=big["students"] / n_stu),
            dict(item="最大組內學校數", value=big["schools"])]
    rows += [dict(item=f"各組學生數_{k}", value=v) for k, v in q(g["students"]).items()]
    rows += [dict(item=f"各組列數_{k}", value=v) for k, v in q(g["rows"]).items()]
    pd.DataFrame(rows).to_csv("results/xschool_groups.csv", index=False)

    tv["xfold"] = tv["grp"].map(gfold)
    folds = tv.groupby("xfold").agg(groups=("grp", "nunique"), students=("user_id", "nunique"), rows=("correct", "size"),
                                    wrong_rate=("correct", lambda x: 1 - x.mean())).reset_index().rename(columns={"xfold": "fold"})
    folds["contains_largest_group"] = folds["fold"] == gfold[g["rows"].idxmax()]
    folds.to_csv("results/xschool_folds.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n各組的學生數與列數分布（35 組）")
    print(pd.DataFrame({"學生數": q(g["students"]), "列數": q(g["rows"])}).round(1).T.to_string())
    print("\n各 fold（跨學校）")
    print(folds.round(3).to_string(index=False))
    print(f"最大 fold / 最小 fold 列數比 = {folds['rows'].max() / folds['rows'].min():.2f}")


if __name__ == "__main__":
    main()
