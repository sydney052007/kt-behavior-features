# ML 期末專題：用作答行為特徵做知識追蹤（ASSISTments 2009，技能 47 + 311）

現行設計是 Claude Project 裡的 `期末專題_新題目/設計修訂_1103.md`（11/3 提案討論後修訂）。
舊設計（全資料、R2/LSTM）已作廢，`工作流程.md` 裡的舊 CLAUDE.md 範本不要用。
設計決策若需要改變，先停下來告訴使用者，不要自己改。

## 資料
- 原始檔是 xlsx（`data/*.xlsx`），用 `src/data_prep/xlsx_to_csv.py` 轉成 `data/raw/*.csv`。
  - 原版 xlsx **沒有表頭，表頭列在最後一列**，腳本已處理（525,534 列、346,860 個唯一 order_id）。
- `src/data_prep/make_subset.py` 產出 `data/interim/`：
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
- 切分已產生：`src/data_prep/make_split.py`（SEED=42，test 用總題數分箱分層抽樣）→ `data/interim/split_users.csv`（user_id, role=test/trainval, fold=0–4；test 的 fold=-1）。彙總在 `results/01_preprocessing/split_summary.txt`。**不要為了分數換種子。** 之後所有實驗讀這個檔，不要重新切。
- 題目難度、作答時間的百分位與中位數只用訓練學生算。
- 作答時間：≤0 標缺失 → 訓練集第 99 百分位截斷 → log → 除以該題訓練集中位數。
- 評估三種：A 全部列（主要）、B 位置分桶（1、2–4、5–10、11+）、C 固定切點（只用在驗證／測試）。**切點固定為 t=6、t=11，程式仍把 k 寫成參數，不要寫死。**
- 指標：ROC-AUC 為主；另報 PR-AUC（答錯為正類）、Accuracy、F1（閾值用該 fold 的訓練列選）。C 用 bootstrap 信賴區間。
- 模型：R0 = PFA（答對／答錯次數）+ LR；R1 = R0 + 行為特徵 + 題目難度 + 學生整體正確率，LR 與 RF。R1 比 R0 多的不只行為特徵，要靠 ablation 拆開。

## 已決定（11/3）
- 序列第 1 列（沒有歷史）：保留為目標。次數類特徵 = 0；比率類特徵用訓練集平均；加 `has_history` 旗標；位置分桶「1」單獨報告。
- Level 3 延伸只做**跨學校**（以學校分組 5-fold）；**leakage 稽核取消**。
- R0 模型特徵 = `log1p_c`、`log1p_w`、`skill_311`、`has_history`（常數 `R0_FEATURES` 在 `src/featurize/features.py`）；`t` 只當欄位（位置分桶用），不進模型：t 與 c + w 資訊重複（`t = c + w + 1`），模型用的是 `log1p_c`、`log1p_w`。一個共用模型、不分技能交互、R0 的 LR 不加正則化（`C=np.inf`，等同 `penalty=None`，但避免 sklearn 1.8 之後的棄用警告）。
- 次數用 `log1p`：R0 的 5-fold 試跑顯示比原始次數好約 0.012 AUC，5 個 fold 一致；R1 沿用同樣設定。
- 評估 A（全部列，主要，選模型與調參以它為準）、B（位置分桶）、C（固定切點）每個模型都要全部報告，評估寫成 `src/common/evaluate.py` 可重用。
- 固定切點事先定為兩個：前 5 題 → 第 6 題（t=6）、前 10 題 → 第 11 題（t=11），不依分數挑；信賴區間用以學生為單位重抽的 cluster bootstrap（2000 次，固定種子）。
- F1 與 Accuracy 的閾值用「該 fold 的訓練列」的預測選，不用驗證列。
- 全程不碰 `role == "test"` 的列，要在程式裡 assert。
- 比較階梯：R0 → R0d（R0 + item_diff + last_correct）→ R1（R0d + 7 個行為特徵）；核心比較是 R0d 對 R1。
- item_diff：平滑 (答對數 + m×p̄)/(作答數 + m)，m 技能 47 = 5、技能 311 = 20，固定不調；訓練列用 inner 5-fold（以學生分組）、驗證列用整個訓練集；禁止 leave-one-out。
- 標準化作答時間：夾在訓練集 [p1, p99] → log → 減去該題訓練集 log 中位數。
- 所有統計量（p1、p99、題目中位數、填補值、item_diff）每個 fold 由訓練列重算；test 只在最後跑一次。
- 調整規則：特徵與模型的選擇只看 5-fold 的 A；R0d 與 R1 用同樣的超參數範圍；記下嘗試次數。
- 跨學校：**不縮減，已於 11/9 排入 11/26–11/28**。
- 跨學校實驗規則（跑之前登記，不得因結果修改；test 的 262 位學生完全不碰）：
  - 分組：在 trainval 上，把共用同一個 `student_class_id` 或 `teacher_id` 的學校併成同一組（union-find 連通分量）。併完最大組超過 trainval 列數的 40%、或組數少於 5 → 停下來回報。否則 `GroupKFold(n_splits=5)` 依組別切（不 shuffle，確定性），「學校 → 組別 → fold」存 `data/split_groups.csv`（untracked，只有 school_id、group_id、fold；存好之後不重新產生）。
  - 特徵 R0d 與 R1（`features_r1.py` 不改）；item_diff 的 inner 5-fold 改成依同一個組別分組；驗證列用完整訓練集的題目統計，沒見過的題目用 p̄。
  - 模型：LR = StandardScaler + LogisticRegression(C=np.inf)；RF = 300 棵、sqrt、種子 42、n_jobs=-1，R0d 用 max_depth=8、min_samples_leaf=5，R1 用 max_depth=None、min_samples_leaf=50，不重新搜尋。
  - 評估 A／B／C（t=6、11）全部報告；R1 − R0d 的配對比較（LR 與 RF）：A 報各 fold 差值、未加權平均、標準差、較好 fold 數與列數加權平均；C 的 cluster bootstrap 以『組別』為單位（2000 次、種子 42）。另報驗證列中訓練集沒出現過的 `problem_id` 比例（跨學校與學生分組各一份），並單獨列出最大組所在 fold 的 R0d、R1 AUC 與差值。
  - 結果只放彙總（`results/0*/xschool_*.csv`），不放 user_id、school_id。
- 跨學校補充實驗（R0 → R0d 的增益能不能搬到新學校；跑之前登記，不得因結果修改；test 的 262 位學生完全不碰）：
  - 比較：LR 的 R0d − R0（同為 StandardScaler + LogisticRegression(C=np.inf)），在學校群分組 5-fold（沿用 `data/split_groups.csv`，不重新產生）下的 A ROC-AUC。R0 特徵 = `R0_FEATURES`（log1p_c、log1p_w、skill_311、has_history），不含 item_diff、last_correct。
  - 學生分組下的對照值從 `results/02_analysis/r1_lr_cv.csv`、`results/02_analysis/r1_lr_paired.csv` 重算並印出實際數字（約 +0.078，不使用四捨五入值）。
  - 判讀規則（A 的未加權 5 fold 平均 ΔAUC）：≥ +0.06 → item_diff 的增益大部分可搬移到新學校；< +0.04 → item_diff 有相當比例是學校／班級效應；介於兩者 → 照實報告，不下結論。
  - 報告：R0 各 fold 的 ROC-AUC、PR-AUC 與平均 ± 標準差；R0d（沿用 `xschool_a_folds.csv`，不重跑）− R0 的各 fold 差值、未加權平均、標準差、較好 fold 數、列數加權平均；學生分組與學校分組的並排表；最大組所在 fold 的單獨數字；C（t=6、t=11）的 R0d − R0 配對 ΔAUC 與 95% CI（組別為單位的 cluster bootstrap，2000 次、種子 42）。結果放 `results/0*/xschool_r0_*.csv`。

## 尚未決定（不要自己決定，遇到時回報）
- （目前沒有）

## 錯誤分析規則（跑之前登記，看到結果後不得修改；只用 5-fold 驗證預測 OOF，test 的 262 位學生完全不碰）
- 範圍：LR 與 RF × 特徵組 R0d、R1。RF 用既有最佳設定（R0d：depth 8、leaf 5；R1：depth None、leaf 50）。用既有程式與同一個 split 重新產生 OOF，並確認 A 的 5 fold 平均與既有結果一致（差 < 1e-9）才繼續。
- 資料保護：不把任何逐列預測存進 repo（OOF 只放記憶體）；輸出只有彙總，不含 user_id、school_id。結果放 `results/02_analysis/error_*.csv`（長格式），寬表放 `results/05_report_tables/`。
- 分段表現：位置（1、2–4、5–10、≥11）、技能（47、311）、item_diff 三等分（切點 = 全部 OOF 驗證列 item_diff 的 33%、67% 分位數；低 ≤ q33、中 (q33, q67]、高 > q67）、提示歷史（有歷史且 hint_rate > 0 vs 其餘；t = 1 歸入「= 0」）。每段輸出列數、答錯率、R0d 與 R1 各自的 ROC-AUC 與 PR-AUC、R1 − R0d 的 ΔAUC 與 95% CI（以學生為單位的配對 cluster bootstrap，2000 次、種子 42）。分段內只有單一類別時 AUC 填缺失並標註。LR 與 RF 各一份。
- 混淆矩陣與錯誤類型：閾值 = 各 fold 訓練列預測選出的 F1 最大閾值（同 A）；答錯為正類；5 個 fold 合併。slip = 預測答對、實際答錯（FN）；guess = 預測答錯、實際答對（FP）。比例主要報「占全部列」，並補報「占實際同類列」（slip 占實際答錯、guess 占實際答對）。
- 校準：預測答對機率分 10 個等寬桶（[0, 0.1)、…、[0.9, 1.0]），每桶輸出列數、平均預測機率、實際答對率；ECE = Σ（桶列數 / 總列數）× |實際答對率 − 平均預測機率|。R0d、R1 × LR、RF 共四份。
- 高信心錯誤：預測答對機率 ≥ 0.8 但實際答錯者；預測答對機率 ≤ 0.2 但實際答對者。輸出兩群的列數、占全部列的比例與占該信心區間列的比例，以及各群在 position（t）、item_diff、hint_rate、last_correct、log1p_c、log1p_w、has_history 的平均 vs 全部列的平均。只有彙總。

- 補充（事後加、描述性，不屬於上面事先登記的分析，不覆蓋原本的結果檔）：「提示歷史」分段的 `hint_rate` 用的是特徵值，但 t = 1 的列沒有歷史、其值是訓練集平均的填補值，所以規則把它們強制歸入「= 0」（1,841 列全在這一組）。為了看這個歸類有沒有影響，另外重算「排除 t = 1」的 hint_rate = 0（10,013 列）與 > 0（18,662 列）兩段的 ΔAUC（R1 − R0d，LR 與 RF，同樣的配對 cluster bootstrap），結果在 `results/02_analysis/error_hint_excl_t1.csv`。

## item_diff 平滑參數 m 的敏感度實驗規則（跑之前登記，看到結果後不得修改；只報告，不改採用的 m；test 的 262 位學生完全不碰）
- 目的：檢查 R0d、R1 的 A ROC-AUC 對 item_diff 的 m 是否敏感。採用值 47 → 5、311 → 20 不改；不論結果如何都維持。若判為敏感，只在報告限制中寫明，不改 m、不挑最佳值。
- 設定：學生分組、既有 split 與 inner fold（種子 42）；只動 m，其他一律不變。一次只動一個技能，另一個維持採用值：47 ∈ {4, 5, 6}（311 固定 20）；311 ∈ {16, 20, 24}（47 固定 5）。5 組不同設定（採用值只算一次）。
- 特徵集：R0d、R1。模型：LR（StandardScaler + LogisticRegression(C=np.inf)）、RF（300 棵、sqrt、種子 42、n_jobs=-1；R0d 用 depth 8、leaf 5；R1 用 depth None、leaf 50；不重新搜尋）。
- 報告：每個設定 × 每個（模型, 特徵集）的 A ROC-AUC 平均；相對採用值的差；R1 − R0d 的 A 差值（平均、各 fold 差、較好 fold 數，LR 與 RF 各一份）。
- 判準（跑之前寫死）：對每個（模型, 特徵集），任何一個設定相對採用值的 |ΔA ROC-AUC| < 0.002，且各設定下 R1 − R0d 都是 5/5 fold 較好 → 「不敏感」；否則 → 「敏感」。整體判讀：四個（模型, 特徵集）全部不敏感才算整體不敏感。

## test 最終評估協定（跑之前登記，看到結果後不得修改；只能跑一次）
- 模型：R0、R0d、R1，各用 LR 與 RF，共 6 組。R1r 與 R0d_rel 不進 test。
- 最終訓練：用全部 trainval（1,481 位學生）訓練。RF 超參數沿用 5-fold 選出的（R0：depth 8、leaf 50；R0d：depth 8、leaf 5；R1：depth None、leaf 50；300 棵、sqrt、種子 42、n_jobs=-1），不重新搜尋；LR = StandardScaler + LogisticRegression(C=np.inf)。所有訓練集統計量（p1、p99、各題中位數、填補值、item_diff）改由全 trainval 重算；訓練列的 item_diff 用 inner OOF（依 user_id、shuffle、種子 42），test 列用全 trainval，沒見過的題目 = p̄。F1／Accuracy 的閾值用最終模型在 trainval 訓練列上的預測選（同 `select_threshold`）。test 學生的歷史特徵只用自己的過去作答。
- 報告：(1) 全部列的 ROC-AUC 與 PR-AUC，附學生為單位的 cluster bootstrap 95% 信賴區間（2000 次、種子 42）；(2) R1 − R0d、R0d − R0 的配對 ΔAUC 與配對 cluster bootstrap 區間（全部列與 C 的 t = 6、11；LR、RF 各一份）；(3) B 位置分桶、C（t = 6、11，附區間）；(4) F1、Accuracy、混淆矩陣（描述性）；(5) 與 5-fold 的 A 平均 ± 標準差並列。
- 判讀規則：任何比較，信賴區間含 0 → 寫「無法確認」，不下結論。test 結果不用來調整任何設定，不論好壞照實報告；若 test 比驗證差，一併寫出。
- 輸出只有彙總，不含 user_id、school_id；逐列預測不存進 repo。結果檔 `results/02_analysis/test_final_*.csv`；只能跑一次：輸出檔已存在就中止。
- 變體總數：R1 特徵版本 = 2（R1、R1r）；另記錄 m 敏感度實驗 1 次（只報告，m 不變，結果：不敏感）。
- 補登記（跑真實 test 之前、沒有任何 test 結果時加入；釐清與精簡上面的報告項目，以這裡為準）：
  - 配對 ΔAUC 只報「全部列」（= test 全部列，不分 fold），共 4 個。主要（事先登記）：R1 − R0d，LR 與 RF 各一個。次要（描述性、不校正多重比較，報告須明說）：R0d − R0，LR 與 RF 各一個。
  - 不計算、不報告 C（t = 6、11）的配對差值；C 只報每個模型自己的 AUC 與 cluster bootstrap 區間（上面 (3) 不變）。
  - B 位置分桶沿用 `eval_b`，不附區間。R0 的 `log1p_c`、`log1p_w` 直接取資料欄位；test 列的 `log1p_c`、`log1p_w` 來自切分前算好的 `features_r0.csv`（只用該學生自己的歷史）。
  - 判讀用語：區間含 0 →「無法確認」；下界 > 0 →「改善」；上界 < 0 →「變差」。

## 變體紀錄（調整次數；報告要明說）
- **R1r 是看過 R1 結果之後才設計的事後變體**（R1 = R0d + 7 個行為特徵；R1r = R1 + 4 個「相對於題目平均」的特徵 `rel_hint_mean`、`rel_hint_last`、`rel_att_mean`、`rel_att_last`，實作在 `src/featurize/features_r1r.py`）。R0d_rel = R0d + 同樣 4 個特徵，只當診斷。
- 到目前為止嘗試過的 R1 特徵版本數 = **2**（R1、R1r）。之後若再加特徵變體，要在這裡累計並在報告說明。
- RF 超參數：每個特徵集（R0、R0d、R1、R1r、R0d_rel）都試過 9 組（`max_depth` ∈ {8, 16, None} × `min_samples_leaf` ∈ {5, 20, 50}）；最佳設定的選擇只看 5-fold 的 A ROC-AUC 平均，所以最佳設定的 A 略偏樂觀（選擇與評估用同樣的 5 個 fold）。LR 沒有超參數網格。
- R1r 的判準（跑之前寫死，不得因結果修改）：以 RF 的 A 為準，R1r − R1（各自最佳設定）的平均 ΔAUC ≥ +0.002 且至少 4/5 個 fold 較好 → 列為「R1 的改良版」；否則列為「沒有明顯改善的變體」。
- R1r 的結果：RF 的 A，R1r − R1 平均 ΔAUC = +0.0054，5/5 個 fold 較好 → 依判準列為「R1 的改良版」。LR 的 R1r − R1 = +0.0030（5/5）。數字見 `results/0*/r1r_*.csv`。因為 R1r 是事後設計的，報告要同時呈現 R1 與 R1r，不能只放 R1r。

## 工作方式
- 程式碼要讓大學部學生看得懂，關鍵步驟寫註解。
- 實驗結果輸出到 `results/` 的子資料夾（`01_preprocessing`、`02_analysis`、`03_process`；索引見 `results/INDEX.md`，檔案搬移對照見 `results/MOVED.csv`）；EDA 彙總輸出在 `results/01_preprocessing/eda_w1/`。
- 先 `source` 虛擬環境再跑；套件見 `requirements.txt`。

## 交接補充（換 session 用；補上上面沒寫的規則、狀態與 repo 外的檔案）

### 資料與 git 的安全規則
- 所有進 git 的輸出（csv、png／svg、log、寬表、`figures/data`）**只能有彙總數字**：不得含 `user_id`、`school_id`（以及 `teacher_id`、`student_class_id`、`order_id`、`problem_id` 這類能回推到學生或單一列的欄位）、逐列預測（如 OOF 的 `p_correct`）或逐列特徵。OOF 只放記憶體或 `data/`（untracked）。新增輸出時要檢查欄位名稱，測試也有檢查（`tests/analysis/test_error_analysis.py`）。
- `data/` 底下全部 untracked：`data/interim/`（`features_r0.csv`、`main_s47_311.csv`、`split_users.csv`、`oof_cmp_*.csv`、`oof_r0_lr.csv` …）、`data/raw/`、原始 xlsx、`data/split_groups.csv`。
- `.gitignore` 擋 `*.xlsx` 與 `*.npz`（保護原始資料的 xlsx；沒有任何例外，不要用 `git add -f` 加入 xlsx）。
- `.gitignore` 的 `data/` 規則會連 `figures/data/` 一起擋，所以另有一行 `!figures/data/`（讓圖背後的彙總數據表可被追蹤）。這兩行都不要移除；注意 gitignore 不支援行內註解，註解要放在獨立一行。
- **不使用 `git push --force`**。push 前確認 remote 是使用者自己的 repo（`origin` = `sydney052007/kt-behavior-features`）、目前在 `main`。commit 訊息不加 `Co-Authored-By`（使用者的要求）。
- 文字（回覆、文件、程式註解）一律用**繁體中文**，不要用簡體字（使用者的要求）。

### test 的 262 位學生
- 到目前為止 test 完全沒被用在任何建模、調參、特徵統計量、評估、錯誤分析或圖表：所有結果都是 trainval 的 5-fold 驗證預測，程式內有 assert。`data/interim/features_r0.csv` 含 test 列的因果特徵（只用該學生自己的歷史，沒有跨學生統計量），但沒有被用來建模。W1 EDA（`results/01_preprocessing/eda_w1/`）是切分前對全部資料做的描述統計，沒有用在任何建模決定。
- **test 的最終評估（只跑一次）尚未執行**；要跑之前先告訴使用者。
- `data/interim/split_users.csv` 與 `data/split_groups.csv` 都已存好，不重新產生、不換種子；`src/experiments/cross_school/make_xschool_groups.py` 在檔案已存在時只驗證不覆蓋。

### 環境與可重現性
- 虛擬環境在 repo 外：`/home/sydney/venvs/mlfinal`（Python 3.14.4）。跑程式與測試用 `/home/sydney/venvs/mlfinal/bin/python`（或先 `source`）。
- **套件版本固定**：`scikit-learn==1.9.1`、`pandas==3.0.6`、`numpy==2.5.4`（`requirements.txt` 已釘；matplotlib 3.11.2、openpyxl 3.1.5 是目前環境的版本，未釘）。理由：item_diff 的 inner 5-fold（`GroupKFold(shuffle=True, random_state=42)`）的分組會隨 sklearn 版本不同，結果才能重現。不要升級套件；非升不可的話，要重跑並重新核對既有結果。
- RF 用 `n_jobs=-1` 時，預測有約 1e-16 的浮點誤差（多執行緒加總順序不固定），樹本身完全相同；測試用 1e-12／1e-9 的容差，不要求逐位元組相同。
- 05 的寬表用 pandas 預設浮點解析讀長格式再寫出，數值可能與 02 的原始 csv 差 1 個浮點末位（約 1e-17）；測試以 1e-12 容差對回 02。`make_figures.py` 用 `float_precision="round_trip"` 讀檔。
- 沒有 pytest：每個測試檔可直接 `python3 tests/featurize/test_features.py` 這樣的方式跑（目前 20 個檔案全過）。

### 資料品質備註（2026-11-19，描述性；沒有改任何處理參數，只用 trainval 的聚合統計）
- `hint_count > 0`（5,958 列）與 `first_action ∈ {1, 2}`（3,553 列）的 `correct` 全為 0，沒有例外（與上面規則一致）。`attempt_count` 沒有這種保證：`attempt_count == 1` 答對率 0.83，≥ 2 不到 1%。
- **36 列異常（A 組）**：`attempt_count ≥ 2`、無提示、`first_action == 0` 卻 `correct == 1`。36 位不同學生各 1 列；嘗試次數 2 次 30 列、3 次 1 列、4 次 2 列、23 次 2 列、329 次 1 列；`ms_first_response == overlap_time` 占 100%（同條件答錯的列只有 0.03%）。**推測**是 `attempt_count` 的記錄問題（無直接證據，未檢視原始 log）。同條件的 3,018 列中 99.2% 答錯。
- **550 列 `attempt_count == 0`（B 組）**：94% 在技能 311；511 列 `first_action == 2`（一進題目就要求鷹架，`correct` 全為 0）；答對的 16 列剛好就是 `first_action == 0` 的 16 列，同樣像記錄異常。`ms ≤ 0` 占 1.1%（約 6 列）。這些是主題目（`original == 1`），仍在目標列內。
- 影響：不影響任何已決定的參數。`attempt_count` 的歷史特徵用 `log1p`、沒有像作答時間一樣截斷；極端值（329 次 → `log1p` 5.80，即 `r1_feature_stats.csv` 中 `last_log_att` 的最大值）只有少數幾十列。**決定：先不改**。報告需要時用一句話交代即可。
- 另：題目歷史很稀疏——同一學生序列內重複作答同一題只占 1.6%（trainval）；列數 < 20 的題目占全部題目 54%（fold != 0 訓練列為 65%，技能 311 為 84%），但只含 20%（訓練列 29%）的列。

### 結果資料夾（`results/`）的分類規則與維護
- `01_preprocessing`：前處理與資料描述（EDA、切分／分組摘要、特徵統計、題目覆蓋）。`02_analysis`：要放進報告的最終分析表；若同一檔案既是分析表又被別的腳本當輸入，放 02，不拆。`03_process`：過程與中間產物（log、僅作過程的網格等）。`05_report_tables`：報告用寬表，由 `src/report/make_report_tables.py` 從 02 的長格式產生，**不重算任何數字**，`tests/report/test_report_tables.py` 檢查每個數字都能回到來源。
- 一個檔案只放一處，不合併、不拆；不確定時放 `03_process`，並在 `INDEX.md` 標「待我確認」。搬檔用 `git mv`。
- `results/INDEX.md` 是**手動維護**的索引（每個檔案的內容、產生腳本、列數、有沒有被程式讀、狀態、備註，另有報告候選圖表對照表）。「狀態」欄：KEEP＝最終管線與報告輸出；ARCHIVE＝過程紀錄（不搬移，只標記）；事後變體＝R1r 相關、報告須與 R1 並陳。新增、搬移結果檔時要同步更新 `INDEX.md` 的檔案數與列。`results/MOVED.csv` 是整理時的舊路徑 → 新路徑對照。
- 腳本內寫死的 `results/…` 路徑要指向新位置（搬移後已全部更新）。

### 圖（`figures/`）
- `figures/`：`fig2_ladder`、`fig3_behavior_groups`、`fig4_cross_school`、`fig5_error_analysis`（各有 PNG 300 dpi 與 SVG），由 `src/report/make_figures.py` 產生；圖背後的彙總數據表在 `figures/data/*.csv`（已追蹤）。圖編號從 fig2 開始（fig1 不在這個 repo）。
- 圖只讀 `results/02_analysis` 與 `05_report_tables` 的 csv，不重算、不重跑模型。`tests/report/test_figures.py` 檢查每個畫出來的數字都能回到來源、平均與 05 的對應表一致，並用 `check_layout` 檢查標註不互相覆蓋、不超出圖框、不壓到資料點與水平線。
- 視覺編碼（四張圖一致）：顏色 = 特徵組（R0 `#0072B2`、R0d `#E69F00`、R1 `#009E73`，Okabe-Ito 色盲友善配色），形狀 = 模型（LR 圓、RF 方）。

### 程式碼結構（依用途分資料夾；第 3 階段整理）
- `src/data_prep/`（轉檔、子集、切分；`eda/` 是 W1 EDA）、`src/featurize/`（features、features_r1、features_r1r）、`src/common/`（cv、evaluate）、`src/experiments/`（`lr/`、`rf/`、`cross_school/`、`variant_r1r/` = R1r 事後變體）、`src/analysis/`（錯誤分析）、`src/report/`（報告表、圖、特徵描述統計、m 的估計）。`tests/` 的資料夾與 `src/` 一一對應。舊路徑 → 新路徑見 `results/MOVED.csv`。
- 一律從 repo 根目錄執行，例如 `python3 src/experiments/rf/run_rf.py`、`python3 tests/featurize/test_features.py`；`data/`、`results/`、`figures/` 都是相對於根目錄的路徑。
- import 維持扁平寫法（`from cv import run_cv`）：每支腳本開頭先把 `src/` 加進 `sys.path`，再 `import _setup_path`（`src/_setup_path.py`），它會把 `src/` 底下所有子資料夾加進 import 路徑。**模組檔名全部不可重複**；新增檔案不要和既有模組或資料夾同名（例如不要用 `features/` 當資料夾名）。

### 產生腳本與執行順序
| 腳本 | 產生 |
|---|---|
| `src/data_prep/make_subset.py`、`src/data_prep/make_split.py`、`src/data_prep/eda/eda_w1_[a-d].py` | `data/interim/` 的子集與切分；`results/01_preprocessing/` 的彙總與 EDA（以 stdout 重導向存成） |
| `src/featurize/features.py`、`src/featurize/features_r1.py`、`src/featurize/features_r1r.py` | R0 特徵（`features_r0.csv`）、R0d／R1 特徵、R1r 的 4 個新特徵（模組，每個 fold 內由訓練列重算統計量） |
| `src/experiments/lr/run_r0_lr.py`、`src/experiments/lr/run_r1_lr.py`、`src/experiments/lr/make_lr_fold_scores.py` | `r0_lr_cv.csv`；`r1_lr_{cv,paired,coef}.csv`；`r1_lr_folds.csv`（LR 各 fold 絕對分數） |
| `src/experiments/lr/run_ablation_lr.py`、`src/experiments/rf/run_rf.py` | `ablation_lr_*.csv`；`rf_*.csv`（含 `rf_grid_auc.csv`） |
| `src/experiments/variant_r1r/run_r1r.py` → **再跑** `src/experiments/variant_r1r/finalize_r1r_results.py` | `r1r_*`；`run_r1r.py` 的原始輸出要由 `finalize_r1r_results.py` 切分成最後的檔案（重複執行會自己擋下） |
| `src/experiments/cross_school/make_xschool_groups.py` → `src/experiments/cross_school/run_xschool.py` → `src/experiments/cross_school/run_xschool_r0.py` | 分組檔與 `xschool_groups.csv`、`xschool_folds.csv` → `xschool_*.csv`、`xschool_unseen_problems.csv` → `xschool_r0_*.csv` |
| `src/report/check_smoothing_m.py`、`src/report/report_features_r1.py` | `smoothing_m_estimate.txt`（item_diff 的 m 重新估計，只回報不改 m）；`r1_feature_stats.csv`（13 個特徵的描述統計）；皆只用 trainval，可獨立執行（不屬 `run_*`，不會覆寫實驗結果） |
| `src/experiments/m_sensitivity/run_m_sensitivity.py` | `m_sensitivity_{cv,verdict}.csv`（item_diff 的 m 敏感度；先確認採用值與既有結果一致；`make_fold_features` 的選用參數 `m_map` 預設 = `ITEM_M`；約 2 分鐘；log 存 `results/03_process/m_sensitivity_run.log`） |
| `src/experiments/final_test/run_final_test.py`（評估函式在 `src/common/eval_final.py`） | **test 最終評估，只能跑一次**（協定見「test 最終評估協定」）。預設（無旗標）只跑假資料自我檢查；真實執行：`python3 src/experiments/final_test/run_final_test.py --run-real-test > results/03_process/test_final_run.log 2>&1`（需 git 工作樹乾淨、輸出檔不存在）→ `results/02_analysis/test_final_{a,paired,b_c,threshold}.csv` |
| `src/analysis/run_error_analysis.py`、`src/analysis/run_error_hint_excl_t1.py` | `error_*.csv`（約 8.5 分鐘，主要是 bootstrap）；`error_hint_excl_t1.csv` |
| `src/report/make_report_tables.py`、`src/report/make_figures.py` | `results/05_report_tables/`；`figures/` |
- 共用模組：`src/common/cv.py`、`src/common/evaluate.py`、`src/experiments/cross_school/xschool.py`。長時間的腳本用 `> results/03_process/<name>_run.log 2>&1` 存 log。

### 目前進度與已知問題
- 已完成：R0／R0d／R1 的 LR 與 RF（含 9 組網格）、ablation、R1r（事後變體）、跨學校 5-fold（R0d、R1）與 R0 補充實驗、錯誤分析（含事後描述性補充）、報告用寬表、圖 2–5 草稿。
- 尚未做：test 的最終評估、報告文字、資料集規模表（原始數字在 `01_preprocessing` 的各 summary 檔）。細節見 `results/INDEX.md` 最後的候選圖表對照表。

### repo 外的檔案與路徑（不在 git，之後可能還會用到）
- 虛擬環境：`/home/sydney/venvs/mlfinal`。
- `~/scratch/`（臨時腳本）：
  - `check_problem_counts.py`：題目作答數分布與未見題目比例；`step0_groups.py`：跨學校分組預覽；`query_units.py`：school／teacher／class／assignment 單位統計。
  - `mutation_check.py`：破壞性檢查範例；`ablation_out.txt`、`r1r_paired_raw_backup.csv`：一次性備份（ARCHIVE）。
- Claude Project 內的設計文件 `期末專題_新題目/設計修訂_1103.md`、`工作流程.md`（不在 repo，以使用者貼給我的內容為準）。
