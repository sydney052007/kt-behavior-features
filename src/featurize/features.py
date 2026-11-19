"""因果特徵管線（R0）。

序列 = (user_id, skill_id)；位置 t = 依 order_id 排序後的名次（從 1 開始），
不用資料自帶的 opportunity_original（CLAUDE.md：8 位學生的重複紀錄會讓它膨脹）。

R0 模型特徵（設計討論後決定，11/3 之後的更新）：
    log1p_c     log1p(前面答對次數)
    log1p_w     log1p(前面答錯次數)
    skill_311   技能 311 = 1，技能 47 = 0（一個共用模型，不分技能交互）
    has_history t > 1

t 本身保留成欄位（給位置分桶評估用），但**不是**模型特徵：
t = c + w + 1，會跟 c、w 共線。c、w 兩個原始次數欄位也保留（非 log 版本），
方便除錯與之後的 ablation。

所有特徵只能用同一學生、同一技能、第 1…t-1 列（都是主題目，make_subset.py
已經丟掉多技能列與鷹架列）——這份管線本身不再丟任何列，印出來確認「丟 0 列」。
"""
import numpy as np
import pandas as pd

SEQ = ["user_id", "skill_id"]  # 序列的分組鍵

# R0 真正會餵進模型的欄位；位置 t、原始次數 c/w 只是輔助欄位，不放進去。
R0_FEATURES = ["log1p_c", "log1p_w", "skill_311", "has_history"]

MAIN_PATH = "data/interim/main_s47_311.csv"
SPLIT_PATH = "data/interim/split_users.csv"
OUT_PATH = "data/interim/features_r0.csv"


def load_main(path: str = MAIN_PATH) -> pd.DataFrame:
    """讀 main_s47_311.csv，只取特徵管線需要的欄位，skill_id 轉成字串。"""
    df = pd.read_csv(path, usecols=["order_id", "user_id", "skill_id", "original", "correct"],
                      low_memory=False)
    df["skill_id"] = df["skill_id"].astype(str)

    # 管線輸入的前提：單一技能 47/311、全是主題目（original==1）。
    # make_subset.py 已經丟掉多技能列與鷹架列，這裡只是確認、不重丟。
    n_before = len(df)
    bad = (~df["skill_id"].isin({"47", "311"})) | (df["original"] != 1)
    assert not bad.any(), f"輸入不符合 main_s47_311.csv 的前提，有 {bad.sum()} 列不是單一技能 47/311 的主題目"
    print(f"[features] 輸入 {n_before:,} 列，全部是單一技能 47/311 的主題目 → 本步驟丟掉 0 列")

    assert df["order_id"].is_unique, "order_id 應該唯一（EDA 已確認），否則排序名次會有並列"
    return df.drop(columns="original")


def add_position(df: pd.DataFrame) -> pd.DataFrame:
    """依 order_id 排序，算出每條序列內的名次 t（從 1 開始）。"""
    df = df.sort_values(SEQ + ["order_id"]).reset_index(drop=True)
    df["t"] = df.groupby(SEQ).cumcount() + 1
    return df


def add_r0_features(df: pd.DataFrame) -> pd.DataFrame:
    """算 R0 特徵。輸入須已經過 add_position（有 order_id 排序好的 t）。

    c、w 分別用「正確」「錯誤」各自的序列內累計和算出來，
    再各自扣掉當前這一列本身 —— 兩者是獨立算出來的，不是互相推導，
    所以之後 c + w == t - 1 是一個真的可以檢查出錯誤的恆等式，不是套套邏輯。
    """
    # cumsum 是依「目前 df 的列順序」累計的，不是依 order_id 的值；
    # 如果輸入沒先排好序，算出來的 c/w 會是錯的歷史（甚至跨到別條序列）。
    # 所以這裡先確認輸入已經依 (user_id, skill_id, order_id) 排好。
    key_cols = SEQ + ["order_id"]
    sorted_keys = df[key_cols].sort_values(key_cols).reset_index(drop=True)
    actual_keys = df[key_cols].reset_index(drop=True)
    assert sorted_keys.equals(actual_keys), (
        "add_r0_features 的輸入必須先依 (user_id, skill_id, order_id) 排序"
        "（例如先呼叫 add_position），否則 cumsum 會算錯"
    )

    df = df.copy()
    df["wrong"] = 1 - df["correct"]

    # 累計「含當前列」，再扣掉當前列自己 → 只剩第 1...t-1 列的歷史。
    # 這一步是整個因果性的關鍵：絕對不能漏減 df["correct"] / df["wrong"]。
    cum_correct_incl = df.groupby(SEQ)["correct"].cumsum()
    cum_wrong_incl = df.groupby(SEQ)["wrong"].cumsum()
    df["c"] = cum_correct_incl - df["correct"]
    df["w"] = cum_wrong_incl - df["wrong"]

    df["log1p_c"] = np.log1p(df["c"])
    df["log1p_w"] = np.log1p(df["w"])
    df["skill_311"] = (df["skill_id"] == "311").astype(int)
    df["has_history"] = (df["t"] > 1).astype(int)

    return df.drop(columns="wrong")


def attach_split(df: pd.DataFrame, path: str = SPLIT_PATH) -> pd.DataFrame:
    """併入 split_users.csv 的 role/fold。絕對不重跑 make_split.py。"""
    split = pd.read_csv(path)
    out = df.merge(split, on="user_id", how="left")
    assert out["role"].notna().all(), "有學生在 split_users.csv 查不到，不該發生"
    assert len(out) == len(df)
    return out


def summarize(df: pd.DataFrame) -> None:
    """只印彙總統計（列數、分位數、has_history==0 的列數），不印任何學生層級的列。"""
    print(f"\n[彙總] 總列數 {len(df):,} | 序列數 {df.groupby(SEQ).ngroups:,}")
    print("  各 role 列數：", df["role"].value_counts().to_dict())

    cols = ["t", "c", "w", "log1p_c", "log1p_w", "skill_311", "has_history"]
    q = df[cols].quantile([0, .25, .5, .75, .9, .99, 1.0])
    print("\n  各特徵分位數：")
    print(q.to_string())

    n_no_hist = int((df["has_history"] == 0).sum())
    print(f"\n  has_history == 0 的列數：{n_no_hist:,}（預期 2,186，即序列數）")


def main() -> None:
    df = load_main()
    df = add_position(df)
    df = add_r0_features(df)
    df = attach_split(df)

    out_cols = ["user_id", "skill_id", "order_id", "t", "correct", "role", "fold",
                "c", "w", "log1p_c", "log1p_w", "skill_311", "has_history"]
    df = df[out_cols].sort_values(SEQ + ["order_id"]).reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"[features] 已寫出 {OUT_PATH}（{len(df):,} 列）")

    summarize(df)


if __name__ == "__main__":
    main()
