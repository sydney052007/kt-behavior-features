# ML 期末專題：用作答行為特徵做知識追蹤（ASSISTments 2009，技能 47 + 311）

現行設計是 Claude Project 裡的 `期末專題_新題目/設計修訂_1103.md`（11/3 提案討論後修訂）。
舊設計（全資料、R2/LSTM）已作廢，`工作流程.md` 裡的舊 CLAUDE.md 範本不要用。
設計決策若需要改變，先停下來告訴使用者，不要自己改。

## 資料
- 原始檔是 xlsx（`data/*.xlsx`），用 `src/xlsx_to_csv.py` 轉成 `data/raw/*.csv`。
  - 原版 xlsx **沒有表頭，表頭列在最後一列**，腳本已處理（525,534 列、346,860 個唯一 order_id）。
- `src/make_subset.py` 產出 `data/interim/`：
  - `main_s47_311.csv`：**特徵管線的輸入**。修正版、單一技能 47／311、`original==1`，共 36,048 列（47：18,737；311：17,311；1,743 位學生）。
  - `corrected_s47_311.csv`：修正版中含 47／311 的所有列（含 3 列多技能、6,944 列鷹架題），只供 EDA 參考。
  - `original_s47_311.csv`：原版子集。**leakage 稽核已取消（11/3 決定），目前不使用**。
- 資料使用條款：不轉交原始資料；`data/` 不進 git；貼到對話、報告、看板的只能是彙總統計，不放學生層級的列。

## 預測任務
- 目標列 = `main_s47_311.csv` 的每一列（修正版、`original == 1`、`skill_id` 恰為 `"47"` 或 `"311"`）。共 36,048 列。
- **多技能列（3 列）與鷹架列（`original == 0`，6,944 列，幾乎都在技能 311）一律丟掉**：不當目標，也不進歷史（11/3 決定）。管線要印出丟掉的列數。
- 序列 = (user_id, skill_id)；位置 t = 依 `order_id` 排序後的名次（從 1 開始）。
  - 不要用資料自帶的 `opportunity_original` 當位置：99.5% 與名次相同，但有 8 位學生的重複紀錄會讓它膨脹。
  - `order_id` 官方說明是「non-chronological id」，沒有時間戳可驗證；目前只知道它與 opportunity 順序一致、答對率的序列相依性合理。
- 特徵只能用同一學生、同一技能、第 1…t−1 列（都是主題目）。
- **目標列自己的 `hint_count`、`first_action`、`attempt_count`、`ms_first_response`、`overlap_time`、`bottom_hint` 永遠不能當特徵**
  （`hint_count>0` 或 `first_action∈{1,2}` 時 `correct` 必為 0，沒有例外）。
- 切分：以 `user_id` 分組，跨兩個技能一起切（443 位學生兩個技能都做）。先留 15% test，最後只跑一次；其餘 5-fold GroupKFold。
- 切分已產生：`src/make_split.py`（SEED=42，test 用總題數分箱分層抽樣）→ `data/interim/split_users.csv`（user_id, role=test/trainval, fold=0–4；test 的 fold=-1）。彙總在 `results/split_summary.txt`。**不要為了分數換種子。** 之後所有實驗讀這個檔，不要重新切。
- 題目難度、作答時間的百分位與中位數只用訓練學生算。
- 作答時間：≤0 標缺失 → 訓練集第 99 百分位截斷 → log → 除以該題訓練集中位數。
- 評估三種：A 全部列（主要）、B 位置分桶（1、2–4、5–10、11+）、C 固定切點（只用在驗證／測試）。**切點固定為 t=6、t=11，程式仍把 k 寫成參數，不要寫死。**
- 指標：ROC-AUC 為主；另報 PR-AUC（答錯為正類）、Accuracy、F1（閾值用該 fold 的訓練列選）。C 用 bootstrap 信賴區間。
- 模型：R0 = PFA（答對／答錯次數）+ LR；R1 = R0 + 行為特徵 + 題目難度 + 學生整體正確率，LR 與 RF。R1 比 R0 多的不只行為特徵，要靠 ablation 拆開。

## 已決定（11/3）
- 序列第 1 列（沒有歷史）：保留為目標。次數類特徵 = 0；比率類特徵用訓練集平均；加 `has_history` 旗標；位置分桶「1」單獨報告。
- Level 3 延伸只做**跨學校**（以學校分組 5-fold）；**leakage 稽核取消**。
- R0 模型特徵 = `log1p_c`、`log1p_w`、`skill_311`、`has_history`（常數 `R0_FEATURES` 在 `src/features.py`）；`t` 只是欄位，不進模型，因為 `t = c + w + 1` 會共線。一個共用模型、不分技能交互、R0 的 LR 不加正則化（`penalty=None`）。
- 次數用 `log1p`：R0 的 5-fold 試跑顯示比原始次數好約 0.012 AUC，5 個 fold 一致；R1 沿用同樣設定。
- 評估 A（全部列，主要，選模型與調參以它為準）、B（位置分桶）、C（固定切點）每個模型都要全部報告，評估寫成 `src/evaluate.py` 可重用。
- 固定切點事先定為兩個：前 5 題 → 第 6 題（t=6）、前 10 題 → 第 11 題（t=11），不依分數挑；信賴區間用以學生為單位重抽的 cluster bootstrap（2000 次，固定種子）。
- F1 與 Accuracy 的閾值用「該 fold 的訓練列」的預測選，不用驗證列。
- 全程不碰 `role == "test"` 的列，要在程式裡 assert。

## 尚未決定（不要自己決定，遇到時回報）
- 跨學校是否也要縮減（12/1 斷點時決定）。

## 工作方式
- 程式碼要讓大學部學生看得懂，關鍵步驟寫註解。
- 實驗結果輸出到 `results/`（csv 存數字）；EDA 彙總輸出在 `results/eda_w1/`。
- 先 `source` 虛擬環境再跑；套件見 `requirements.txt`。
