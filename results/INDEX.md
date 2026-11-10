# results/ 索引

整理日期見 git log。所有檔案只有彙總數字（沒有 user_id、school_id 欄位）。舊路徑 → 新路徑對照在 `MOVED.csv`。
「有沒有被程式讀」只計 `src/`、`tests/` 內的讀取（`src/make_report_tables.py` 與 `tests/test_report_tables.py` 讀的是 02_analysis 內的多個檔案，各表的來源見 05 一節，這裡不逐一重複）；`~/scratch` 內的臨時腳本不算。

共 76 個檔案：01_preprocessing 10 個、02_analysis 38 個、03_process 6 個、04_exports 3 個、05_report_tables 19 個。
待我確認：無（`rf_grid_auc.csv` 已依你的決定移到 02_analysis）。

## 01_preprocessing — 前處理與資料描述

| 檔案 | 內容 | 產生它的腳本 | 列數 | 有沒有被程式讀 | 備註 |
|---|---|---|---|---|---|
| `eda_w1/eda_w1_a.txt` | W1 EDA（A）：子集規模、first_action、correct 的定義、order／opportunity 的彙總 | src/eda_w1_a.py | 109 行 | 無 | 由 stdout 重導向存成（腳本只 print）；推測，腳本內沒有寫檔指令 |
| `eda_w1/eda_w1_b.txt` | W1 EDA（B）：精熟規則、位置、第 1 題、多技能、鷹架題、opportunity 不一致 | src/eda_w1_b.py | 140 行 | 無 | 由 stdout 重導向存成（腳本只 print）；推測，腳本內沒有寫檔指令 |
| `eda_w1/eda_w1_c.txt` | W1 EDA（C）：以 assignment 重驗精熟規則、鷹架順序、年級欄位、學校規模、題組 | src/eda_w1_c.py | 70 行 | 無 | 由 stdout 重導向存成（腳本只 print）；推測，腳本內沒有寫檔指令；內文有欄位名稱清單，但沒有學生層級的值 |
| `eda_w1/eda_w1_d.txt` | W1 EDA（D）：原版重複列、原版 vs 修正版對照、opportunity 落差來源 | src/eda_w1_d.py | 37 行 | 無 | 由 stdout 重導向存成（腳本只 print）；推測，腳本內沒有寫檔指令 |
| `make_subset_summary.txt` | 技能 47／311 子集篩選的彙總（修正版與原版的列數） | src/make_subset.py | 16 行 | 無 | 由 stdout 重導向存成（腳本只 print）；推測，腳本內沒有寫檔指令 |
| `r1r_feature_stats.csv` | R1r 的 4 個新特徵的平均、標準差、最小、最大與和 correct 的相關 | src/run_r1r.py（原 r1r_new_features.csv）→ src/finalize_r1r_results.py 改名 | 4 列 | 無 |  |
| `split_summary.txt` | train／test 與 5-fold 切分的彙總（SEED=42） | src/make_split.py | 18 行 | 無 | 由 stdout 重導向存成（腳本只 print）；推測，腳本內沒有寫檔指令 |
| `xschool_folds.csv` | 跨學校 5 個 fold 的組數、學生數、列數、答錯率 | src/make_xschool_groups.py | 5 列 | 有：`src/run_xschool.py`、`src/run_xschool_r0.py` | 也當輸入：用它找『最大組所在的 fold』 |
| `xschool_groups.csv` | 跨學校分組摘要：40 所學校併成 35 組，最大組占比與各組大小分布 | src/make_xschool_groups.py | 16 列 | 無 |  |
| `xschool_unseen_problems.csv` | 驗證列中訓練集沒出現過的 problem_id 比例（跨學校 vs 學生分組，各 fold） | src/run_xschool.py | 5 列 | 無 | 題目覆蓋 |

## 02_analysis — 要放進報告的最終分析表

| 檔案 | 內容 | 產生它的腳本 | 列數 | 有沒有被程式讀 | 備註 |
|---|---|---|---|---|---|
| `ablation_lr_cv.csv` | 15 組特徵 LR ablation 的 A／B／C | src/run_ablation_lr.py | 360 列 | 無 |  |
| `ablation_lr_paired.csv` | ablation 各組對參考組的配對 ΔAUC（A 逐 fold、C 配對 CI） | src/run_ablation_lr.py | 42 列 | 無 | 行為特徵群邊際貢獻的逐 fold 值在這裡 |
| `ablation_lr_steps.csv` | R0→R0d→R1 每一步的 ΔAUC | src/run_ablation_lr.py | 9 列 | 無 |  |
| `error_reproduction.csv` | 錯誤分析的重現檢查：重新產生的 OOF 其 A 平均（ROC-AUC、PR-AUC）與既有結果的差（LR、RF × R0d、R1） | src/run_error_analysis.py | 8 列 | 有：`src/make_report_tables.py`、`tests/test_error_analysis.py` | 規則見 CLAUDE.md「錯誤分析規則」；只有彙總（OOF 逐列預測沒有存檔） |
| `error_cutpoints.csv` | 錯誤分析：item_diff 三等分的切點（全部 OOF 驗證列的 33%、67% 分位數） | src/run_error_analysis.py | 2 列 | 有：`src/make_report_tables.py`、`tests/test_error_analysis.py` | 規則見 CLAUDE.md「錯誤分析規則」；只有彙總（OOF 逐列預測沒有存檔） |
| `error_segments.csv` | 錯誤分析：分段表現（位置／技能／item_diff 三等分／提示歷史）；R0d、R1 的 ROC-AUC、PR-AUC 與 R1−R0d 的 ΔAUC 與 95% CI（學生為單位的配對 cluster bootstrap） | src/run_error_analysis.py | 110 列 | 有：`src/make_report_tables.py`、`tests/test_error_analysis.py` | 規則見 CLAUDE.md「錯誤分析規則」；只有彙總（OOF 逐列預測沒有存檔） |
| `error_confusion.csv` | 錯誤分析：5 個 fold 合併的混淆矩陣（TP／FP／FN／TN）與 slip、guess 的比例（R0d、R1 × LR、RF） | src/run_error_analysis.py | 24 列 | 有：`src/make_report_tables.py`、`tests/test_error_analysis.py` | 規則見 CLAUDE.md「錯誤分析規則」；只有彙總（OOF 逐列預測沒有存檔） |
| `error_calibration.csv` | 錯誤分析：預測答對機率的 10 個等寬桶（列數、平均預測機率、實際答對率） | src/run_error_analysis.py | 40 列 | 有：`src/make_report_tables.py`、`tests/test_error_analysis.py` | 規則見 CLAUDE.md「錯誤分析規則」；只有彙總（OOF 逐列預測沒有存檔） |
| `error_calibration_ece.csv` | 錯誤分析：ECE（R0d、R1 × LR、RF） | src/run_error_analysis.py | 4 列 | 有：`src/make_report_tables.py`、`tests/test_error_analysis.py` | 規則見 CLAUDE.md「錯誤分析規則」；只有彙總（OOF 逐列預測沒有存檔） |
| `error_high_conf_groups.csv` | 錯誤分析：高信心錯誤兩群（預測答對 ≥ 0.8 但答錯；預測 ≤ 0.2 但答對）的列數與占比 | src/run_error_analysis.py | 8 列 | 有：`src/make_report_tables.py`、`tests/test_error_analysis.py` | 規則見 CLAUDE.md「錯誤分析規則」；只有彙總（OOF 逐列預測沒有存檔） |
| `error_high_conf_features.csv` | 錯誤分析：高信心錯誤兩群在 7 個特徵的平均 vs 全部列的平均 | src/run_error_analysis.py | 56 列 | 有：`src/make_report_tables.py`、`tests/test_error_analysis.py` | 規則見 CLAUDE.md「錯誤分析規則」；只有彙總（OOF 逐列預測沒有存檔） |
| `r0_lr_cv.csv` | R0 + LR（沒有 StandardScaler）的 A／B／C，W2 第一版結果 | src/run_r0_lr.py | 24 列 | 有：`src/run_r1_lr.py`、`tests/test_xschool_r0.py` | **被 `r1_lr_cv.csv` 的 R0 列取代，報告以後者為準**（兩者在小數點三位相同；此檔只保留作為『加 StandardScaler 後數字不變』的對照） |
| `r1_lr_coef.csv` | R1 LR 的標準化係數（5 個 fold 平均與標準差） | src/run_r1_lr.py | 13 列 | 無 |  |
| `r1_lr_cv.csv` | R0／R0d／R1 的 LR A／B／C（StandardScaler + LR，學生分組） | src/run_r1_lr.py | 72 列 | 有：`src/run_r1r.py`、`src/run_ablation_lr.py`、`src/run_rf.py`、`src/run_xschool_r0.py`、`tests/test_xschool.py`、`tests/test_xschool_r0.py` | 被很多腳本當對照輸入 |
| `r1_lr_folds.csv` | LR（R0／R0d／R1）在 A 的各 fold 絕對 ROC-AUC 與 PR-AUC（長格式） | src/make_lr_fold_scores.py | 15 列 | 有：`src/make_report_tables.py`、`tests/test_lr_fold_scores.py` | 平均與標準差與 r1_lr_cv.csv 一致（差 < 1e-9）；補上資訊階梯圖需要的各 fold 絕對值 |
| `r1_lr_paired.csv` | LR 的 R0d−R0、R1−R0d 配對差異（A 逐 fold、C 配對 CI） | src/run_r1_lr.py | 6 列 | 有：`src/run_rf.py`、`src/run_xschool_r0.py` |  |
| `r1_minus_r0d_pr_auc.csv` | R1−R0d 的 ΔPR-AUC（LR 與 RF） | src/finalize_r1r_results.py（由 run_r1r.py 原始輸出切分） | 6 列 | 無 | 檔名是 r1_ 開頭但由 R1r 腳本產生 |
| `r1r_a_summary.csv` | R1r、R0d_rel（及 R1、R0d 參考）的 A ROC-AUC／PR-AUC 摘要 | src/run_r1r.py | 8 列 | 有：`src/finalize_r1r_results.py` | 被整理腳本當輸入 |
| `r1r_cv.csv` | R1r、R0d_rel 的 LR 與 RF 的 A／B／C | src/run_r1r.py | 96 列 | 無 | R1r 是事後變體 |
| `r1r_paired.csv` | R1r−R1、R0d_rel−R0d（LR 與 RF）的配對差異 | src/run_r1r.py（原始）→ src/finalize_r1r_results.py（只留 own_best） | 12 列 | 有：`src/finalize_r1r_results.py` | 整理腳本會讀它再覆寫自己 |
| `r1r_paired_same_config.csv` | 補充：RF 兩邊都用 R1 最佳設定的 R1r−R1 | src/finalize_r1r_results.py | 3 列 | 無 |  |
| `r1r_perm_importance.csv` | R1r 的 RF permutation importance（診斷用） | src/run_r1r.py | 8 列 | 無 |  |
| `r1r_verdict.csv` | R1r 判準（事先寫死）與結果 | src/run_r1r.py | 1 列 | 無 |  |
| `r1r_vs_r0d_rel.csv` | R1r 與 R0d_rel 的 A 比較（LR 與 RF 的 ROC-AUC、PR-AUC 與差值） | src/finalize_r1r_results.py | 4 列 | 無 | 只有各 fold 平均的差，沒有配對 CI |
| `rf_cv.csv` | RF 各組最佳設定的 A／B／C | src/run_rf.py | 72 列 | 無 |  |
| `rf_grid_auc.csv` | RF 9 組超參數 × R0／R0d／R1 的 A ROC-AUC 完整網格（含各 fold） | src/run_rf.py | 27 列 | 有：`src/run_r1r.py` | 依你的決定從 03_process 移到 02_analysis（被 `src/run_r1r.py` 讀取、報告的 RF 網格表也由它產生） |
| `rf_paired.csv` | RF 的 R0d−R0、R1−R0d 配對差異（各自最佳設定，及用 R0d 最佳設定的穩健性檢查） | src/run_rf.py | 12 列 | 無 | 穩健性列（analysis=R0d_best_cfg_for_all）與主要列在同一個檔案，未拆 |
| `rf_perm_importance.csv` | R1 最佳設定的 RF permutation importance（診斷用） | src/run_rf.py | 6 列 | 無 |  |
| `rf_vs_lr.csv` | R0d、R1 在 RF 與 LR 下的 A，以及 R1−R0d 並排 | src/run_rf.py | 4 列 | 無 |  |
| `xschool_a_folds.csv` | 跨學校各 fold 的 A 分數（ROC-AUC、PR-AUC、F1、Accuracy） | src/run_xschool.py | 20 列 | 有：`src/run_xschool_r0.py` | 兼具各 fold 原始分數與分析表，且被 run_xschool_r0.py 當輸入，依規則放 02 |
| `xschool_cv.csv` | 跨學校 5-fold 的 R0d／R1（LR 與 RF）A／B／C（C 的 CI 以組別為單位） | src/run_xschool.py | 96 列 | 無 |  |
| `xschool_fold0.csv` | 跨學校最大組所在 fold 的 R0d、R1 AUC 與差值 | src/run_xschool.py | 2 列 | 無 |  |
| `xschool_paired.csv` | 跨學校 R1−R0d 配對差異（LR 與 RF；A 逐 fold 與列數加權平均、C 的 CI） | src/run_xschool.py | 6 列 | 無 |  |
| `xschool_r0_a_folds.csv` | 跨學校 R0 各 fold 的 ROC-AUC、PR-AUC 與 R0d−R0 差值 | src/run_xschool_r0.py | 5 列 | 無 |  |
| `xschool_r0_a_summary.csv` | 跨學校 R0 的 A 彙總與 R0d−R0 的判讀（事先登記規則） | src/run_xschool_r0.py | 4 列 | 無 |  |
| `xschool_r0_fold0.csv` | 跨學校最大組所在 fold 的 R0、R0d 與差值 | src/run_xschool_r0.py | 1 列 | 無 |  |
| `xschool_r0_paired_c.csv` | 跨學校 C（t=6、t=11）的 R0d−R0 配對 ΔAUC 與 95% CI | src/run_xschool_r0.py | 2 列 | 無 |  |
| `xschool_r0_vs_student_split.csv` | 學生分組與學校群分組的 R0d−R0 並排（含各 fold 差值） | src/run_xschool_r0.py | 2 列 | 無 | 各 fold 差值欄位原本分成 diff_foldK 與 diff_foldK.0 兩套（兩套互補，每列只填一套；原因是 fold 編號一邊整數、一邊浮點）；已合併成 diff_fold0…4，數字與修前逐格相同 |

## 03_process — 過程與中間產物

| 檔案 | 內容 | 產生它的腳本 | 列數 | 有沒有被程式讀 | 備註 |
|---|---|---|---|---|---|
| `error_analysis_run.log` | run_error_analysis.py 的完整執行紀錄（含重現檢查與所有彙總表） | src/run_error_analysis.py | 167 行 | 無 | 由 shell 重導向存成（python3 … > 檔名）；腳本本身不寫這個檔 |
| `r1r_rf_grid.csv` | R1r、R0d_rel 的 RF 9 組網格（含各 fold） | src/run_r1r.py | 18 列 | 無 |  |
| `r1r_run.log` | run_r1r.py 的完整執行紀錄 | src/run_r1r.py | 105 行 | 無 | 由 shell 重導向存成（python3 … > 檔名）；腳本本身不寫這個檔 |
| `rf_run.log` | run_rf.py 的完整執行紀錄（含 27 組網格的進度） | src/run_rf.py | 124 行 | 無 | 由 shell 重導向存成（python3 … > 檔名）；腳本本身不寫這個檔 |
| `xschool_r0_run.log` | run_xschool_r0.py 的完整執行紀錄 | src/run_xschool_r0.py | 34 行 | 無 | 由 shell 重導向存成（python3 … > 檔名）；腳本本身不寫這個檔 |
| `xschool_run.log` | run_xschool.py 的完整執行紀錄 | src/run_xschool.py | 62 行 | 無 | 由 shell 重導向存成（python3 … > 檔名）；腳本本身不寫這個檔 |

## 04_exports — 給人看的匯出檔

| 檔案 | 內容 | 產生它的腳本 | 列數 | 有沒有被程式讀 | 備註 |
|---|---|---|---|---|---|
| `W3_LR結果.xlsx` | LR（R0／R0d／R1）A／B／C、配對差異、係數、ablation 的 Excel 匯出 | ~/scratch/make_w3_xlsx_all.py（不在 repo） | 11 張工作表 | 無 | 由 csv 整理；腳本不在 repo，要重現需要把它加進 src/ |
| `W3_RF結果.xlsx` | RF 網格、A／B／C、配對差異、permutation importance、R1r 與 R0d_rel 的 Excel 匯出 | ~/scratch/make_w3_xlsx_all.py（不在 repo） | 20 張工作表 | 無 | 同上 |
| `W3_跨學校結果.xlsx` | 跨學校分組與各 fold、配對差異、R0→R0d 並排、未見題目比例、C 的 CI 的 Excel 匯出 | ~/scratch/make_w3_xlsx_all.py（不在 repo） | 15 張工作表 | 無 | 同上 |

## 05_report_tables — 報告用寬表（由 02_analysis 的長格式檔案整理，不重算任何數字）

產生腳本：`src/make_report_tables.py`；測試：`tests/test_report_tables.py`（寬表裡每個數字都要能在來源檔案找到相同的值）。
列 = 比較對象，欄 = 指標；每個指標有獨立的數值欄（mean、sd，或 value、ci_low、ci_high）和一個格式化字串欄（`*_str`，4 位小數）。A、B、C 各自一張表（`c_ablation` 依規格同時有 A 與 C 的欄）。

| 檔案 | 內容 | 來源檔案（皆在 02_analysis） | 列數 | 對應報告候選圖表 | 備註 |
|---|---|---|---|---|---|
| `05_report_tables/a_main_A.csv` | 主結果表（A）：LR／RF × R0／R0d／R1 的 ROC-AUC、PR-AUC、F1、Accuracy | `r1_lr_cv.csv`、`rf_cv.csv` | 6 列 | 4. 主結果表 | RF 的 setting 欄是各特徵組自己的最佳設定 |
| `05_report_tables/a_main_C.csv` | 主結果表（C）：t=6、t=11 的 ROC-AUC、PR-AUC 與 95% CI | `r1_lr_cv.csv`、`rf_cv.csv` | 6 列 | 4. 主結果表 | C 的 CI 為 cluster bootstrap（學生為單位） |
| `05_report_tables/b_paired_A.csv` | 配對差異（A）：R0d−R0、R1−R0d；平均、標準差、各 fold 差值、較好 fold 數 | `r1_lr_paired.csv`、`rf_paired.csv` | 6 列 | 1. 資訊階梯圖（增量）；4. 主結果表 | RF 有 own_best 與 R0d_best_cfg_for_all（穩健性）兩種分析 |
| `05_report_tables/b_paired_C.csv` | 配對差異（C）：t=6、t=11 的 ΔAUC 與 95% CI | `r1_lr_paired.csv`、`rf_paired.csv` | 6 列 | 1. 資訊階梯圖（增量）；4. 主結果表 | 同上 |
| `05_report_tables/c_ablation.csv` | LR ablation：每組一列（對一個參考組）；A 的 AUC、ΔAUC、較好 fold 數、各 fold 差值，以及 C t=6／t=11 的 ΔAUC 與 CI | `ablation_lr_cv.csv`、`ablation_lr_paired.csv` | 14 列 | 2. 行為特徵群邊際貢獻圖 | 每列是『一個組對一個參考組』；A 與 C 的欄在同一張 |
| `05_report_tables/d_position_bins_LR.csv` | 位置分桶 B（LR）：桶為列，R0／R0d／R1 的 ROC-AUC、PR-AUC 為欄 | `r1_lr_cv.csv` | 4 列 | 5. 位置分桶 B 表 |  |
| `05_report_tables/d_position_bins_RF.csv` | 位置分桶 B（RF，各組最佳設定） | `rf_cv.csv` | 4 列 | 5. 位置分桶 B 表 | 三個特徵組的最佳設定不同，見 setting_* 欄 |
| `05_report_tables/e_cross_school.csv` | 跨學校：學生分組與學校群分組並排的 R0d−R0、R1−R0d（LR、RF），各 fold 差值 | `r1_lr_paired.csv`、`rf_paired.csv`、`xschool_paired.csv`、`xschool_r0_vs_student_split.csv`、`xschool_r0_a_summary.csv` | 7 列 | 3. 跨學校圖 | R0d−R0 的學校群分組只有 LR（RF 沒有跨學校 R0）；weighted_mean 只有學校群分組有 |
| `05_report_tables/g_fold_auc.csv` | 各 fold 的絕對 AUC：LR 與 RF（最佳設定）× R0／R0d／R1 | `r1_lr_folds.csv`、`rf_grid_auc.csv` | 6 列 | 1. 資訊階梯圖（各 fold 值） | LR 的 PR-AUC 有各 fold 值；RF 沒有逐 fold 的 PR-AUC（欄位為空） |
| `05_report_tables/f_rf_grid_R0.csv` | RF 網格（R0）：max_depth（列）× min_samples_leaf（欄）的 A ROC-AUC，* = 最佳 | `rf_grid_auc.csv` | 3 列 | （補充；無對應候選） | R1r、R0d_rel 的網格在 03_process，不納入 |
| `05_report_tables/f_rf_grid_R0d.csv` | RF 網格（R0d）：max_depth（列）× min_samples_leaf（欄）的 A ROC-AUC，* = 最佳 | `rf_grid_auc.csv` | 3 列 | （補充；無對應候選） |  |
| `05_report_tables/f_rf_grid_R1.csv` | RF 網格（R1）：max_depth（列）× min_samples_leaf（欄）的 A ROC-AUC，* = 最佳 | `rf_grid_auc.csv` | 3 列 | （補充；無對應候選） |  |
| `05_report_tables/h_error_segments_LR.csv` | 錯誤分析・分段表現（LR）：位置／技能／item_diff 三等分／提示歷史，R0d 與 R1 的 AUC、R1−R0d 的 ΔAUC 與 95% CI | `error_segments.csv` | 11 列 | 8. 錯誤分析 | 分段內單一類別時 AUC 為空並註記（本次沒有出現） |
| `05_report_tables/h_error_segments_RF.csv` | 錯誤分析・分段表現（RF）：位置／技能／item_diff 三等分／提示歷史，R0d 與 R1 的 AUC、R1−R0d 的 ΔAUC 與 95% CI | `error_segments.csv` | 11 列 | 8. 錯誤分析 | 同上 |
| `05_report_tables/i_error_confusion.csv` | 錯誤分析・混淆矩陣與 slip／guess（5 個 fold 合併；答錯為正類） | `error_confusion.csv` | 4 列 | 8. 錯誤分析 | slip = FN、guess = FP；比例占全部列與占實際同類列 |
| `05_report_tables/j_calibration_LR.csv` | 錯誤分析・校準（LR）：10 個等寬桶，R0d 與 R1 的列數、平均預測機率、實際答對率 | `error_calibration.csv` | 10 列 | 8. 錯誤分析 | 最後一桶含 1.0 |
| `05_report_tables/j_calibration_RF.csv` | 錯誤分析・校準（RF）：10 個等寬桶，R0d 與 R1 的列數、平均預測機率、實際答對率 | `error_calibration.csv` | 10 列 | 8. 錯誤分析 | 同上 |
| `05_report_tables/j_calibration_ece.csv` | 錯誤分析・ECE（R0d、R1 × LR、RF） | `error_calibration_ece.csv` | 4 列 | 8. 錯誤分析 |  |
| `05_report_tables/k_high_confidence.csv` | 錯誤分析・高信心錯誤兩群的列數、占比與特徵平均（vs 全部列） | `error_high_conf_groups.csv`、`error_high_conf_features.csv` | 8 列 | 8. 錯誤分析 | hint_rate 在 t=1 是訓練集平均填補值（與其他特徵一致，未特別處理） |

## 報告候選圖表對照表

| 候選 | 對應檔案 | 備註 |
|---|---|---|
| 1. 資訊階梯圖：R0／R0d／R1 的 A AUC（LR、RF），各 fold 值與增量 | 各 fold 絕對 AUC：`02_analysis/r1_lr_folds.csv`（LR，含 PR-AUC）、`02_analysis/rf_grid_auc.csv`（RF，各設定）；增量（逐 fold）：`02_analysis/r1_lr_paired.csv`（LR）、`02_analysis/rf_paired.csv`（RF）；平均：`02_analysis/r1_lr_cv.csv`、`02_analysis/rf_cv.csv` | 寬表：`05_report_tables/g_fold_auc.csv`（各 fold 值）、`b_paired_A.csv`（增量）、`a_main_A.csv`（平均）；RF 沒有逐 fold 的 PR-AUC |
| 2. 行為特徵群邊際貢獻圖：加入 R0d 的 ΔAUC，各 fold 值 | `02_analysis/ablation_lr_paired.csv`（LR；R0d+hint／time／att／scaffold 對 R0d，逐 fold） | 寬表：`05_report_tables/c_ablation.csv`；RF 版**尚無**（`02_analysis/rf_perm_importance.csv` 是 RF 的 permutation 診斷，不是加入式的邊際貢獻） |
| 3. 跨學校圖：學生分組與學校群分組的 R0d−R0、R1−R0d，各 fold 值 | R0d−R0：`02_analysis/xschool_r0_vs_student_split.csv`（兩種分組並排，逐 fold）；R1−R0d：`02_analysis/xschool_paired.csv`（學校群分組，LR、RF）、`02_analysis/r1_lr_paired.csv`（學生分組，LR）、`02_analysis/rf_paired.csv`（學生分組，RF） | 寬表：`05_report_tables/e_cross_school.csv`；學校群分組的 R0d−R0 只有 LR |
| 4. 主結果表：A（AUC、PR-AUC）與 C t=6／t=11 的點估計與 95% CI | `02_analysis/r1_lr_cv.csv`（LR：R0／R0d／R1）、`02_analysis/rf_cv.csv`（RF）、`02_analysis/r1r_cv.csv`（R1r、R0d_rel）、`02_analysis/xschool_cv.csv`（跨學校） | 寬表：`05_report_tables/a_main_A.csv`、`a_main_C.csv`（R0／R0d／R1 的 LR 與 RF）、`b_paired_A.csv`、`b_paired_C.csv`（配對差異）；R1r 與跨學校尚未做成寬表 |
| 5. 位置分桶 B 表 | `02_analysis/r1_lr_cv.csv`、`02_analysis/rf_cv.csv`、`02_analysis/xschool_cv.csv`、`02_analysis/r1r_cv.csv` 的 eval=B 列 | 寬表：`05_report_tables/d_position_bins_LR.csv`、`d_position_bins_RF.csv`；另有 `04_exports/` 內 Excel 的 *_B 工作表 |
| 6. 特徵清單／前處理表 | 尚無 | 部分：`01_preprocessing/r1r_feature_stats.csv` 只有 4 個新特徵的統計；R0d／R1 的 13 個特徵的統計表尚無（特徵定義在 `src/features_r1.py`） |
| 7. 資料集規模表 | 尚無（02_analysis 內沒有）；原始數字在 `01_preprocessing/make_subset_summary.txt`、`01_preprocessing/split_summary.txt`、`01_preprocessing/xschool_groups.csv`、`01_preprocessing/xschool_folds.csv` | 需要時可由這幾個檔案整理 |
| 8. 錯誤分析 | `02_analysis/error_segments.csv`（分段表現）、`error_confusion.csv`（混淆矩陣、slip／guess）、`error_calibration.csv`、`error_calibration_ece.csv`（校準）、`error_high_conf_groups.csv`、`error_high_conf_features.csv`（高信心錯誤）、`error_cutpoints.csv`、`error_reproduction.csv`（重現檢查） | 寬表：`05_report_tables/h_error_segments_LR.csv`、`h_error_segments_RF.csv`、`i_error_confusion.csv`、`j_calibration_LR.csv`、`j_calibration_RF.csv`、`j_calibration_ece.csv`、`k_high_confidence.csv`；只有彙總，OOF 逐列預測沒有存檔 |
