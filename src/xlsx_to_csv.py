"""把 ASSISTments xlsx 轉成 csv（python-calamine，比 openpyxl 快很多）。

注意：原版 skill_builder_data.xlsx 的表頭列被放在「最後一列」（第一列就是資料），
所以讀取時用 header=None，套用官方欄位名稱，並丟掉內容是欄位名稱的那一列
（否則多出 1 列、整欄變成文字型態）。修正版多一欄無名稱的列號 'Unnamed: 0'，會被丟掉。
用法：python src/xlsx_to_csv.py <in.xlsx> <out.csv> [--no-header]
"""
import sys, time
import pandas as pd

ORIG_COLS = ['order_id','assignment_id','user_id','assistment_id','problem_id','original',
             'correct','attempt_count','ms_first_response','tutor_mode','answer_type',
             'sequence_id','student_class_id','position','type','base_sequence_id',
             'skill_id','skill_name','teacher_id','school_id','hint_count','hint_total',
             'overlap_time','template_id','answer_id','answer_text','first_action',
             'bottom_hint','opportunity','opportunity_original']

src, dst = sys.argv[1], sys.argv[2]
no_header = "--no-header" in sys.argv
t = time.time()
if no_header:
    df = pd.read_excel(src, engine="calamine", header=None)
    assert df.shape[1] == len(ORIG_COLS), df.shape
    df.columns = ORIG_COLS
    is_header = df["order_id"].astype(str) == "order_id"
    print("表頭列（出現在資料中間或結尾）數：", int(is_header.sum()), "位置：", df.index[is_header].tolist(), flush=True)
    df = df[~is_header].reset_index(drop=True)
else:
    df = pd.read_excel(src, engine="calamine")
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
print("shape:", df.shape, f"{time.time()-t:.0f}s", flush=True)
print("columns:", list(df.columns), flush=True)
df.to_csv(dst, index=False, encoding="utf-8")
print("saved", dst, flush=True)
