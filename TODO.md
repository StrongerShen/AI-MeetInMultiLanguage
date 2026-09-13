# 文件整理與後續工作

本文件供 AGY CLI 接手。開始前先閱讀 `README.md`、檢查 `git status` 與最近 5 筆 `git log`。未經使用者明確指示，不得執行 `git commit` 或 `git push`。

## P0：文件一致性

- [x] 統一 Web 上傳上限為 **200 MiB（209715200 bytes）**。
  - 同步 `README.md`、`.env.example`、設定、API、Web 提示與測試。
  - API 超限訊息必須依設定值動態產生，不可寫死容量。
  - 500 MB／120 分鐘是未來驗收目標，不是目前承諾。
- [x] 修正 README 內互相矛盾的 GPU 資料。
  - Breeze 60 秒實測峰值：約 2,303 MiB。
  - Breeze → Ollama 全流程峰值：6,972 MiB。
  - 完成卸載後閒置：約 213 MiB。
  - 不得宣稱 GPU 用量絕對恆定或 200 MiB 音訊對 RAM「完全無壓力」。
- [x] 將架構分成「目前已實作」與「未來方向」。
  - 目前：Python、FastAPI、原生 JavaScript、JSON、本機檔案、BackgroundTasks、Speaches、Ollama。
  - 未來：TypeScript、關聯式資料庫、物件儲存、正式工作佇列、多租戶權限。
- [x] 補齊 README 空白的「核心資料與 API」章節。
  - 說明 `EvaluationRun`、`raw_asr`、`revisions`、`result`、`summary`、工作狀態與不可變規則。
  - 列出主要 API，以及 400、404、409、413、415 錯誤契約。
- [x] 補齊快速開始、完整 pipeline、各 CLI 指令、設定與五種匯出格式。
- [x] 將歷次審查濃縮成時間軸，移除大段過期缺陷說明。
  - `a34ae49`、`621a55c`、`8b64a78`：未簽核。
  - `8e71e6d`：第五輪正式簽核通過。
  - 記錄 111 項測試及靜態檢查通過。
- [x] 擴充 README 目錄，確認章節順序及錨點可正確跳轉。

## P1：受控發布與實機驗收

- [x] 驗證 Breeze → Ollama 與 Ollama → Breeze 模型切換。
- [x] 驗證 Web → CLI 與 CLI → Web 交錯執行。
- [x] 模擬 Speaches／Ollama 離線，確認 fail closed 與錯誤訊息。
- [x] 確認工作完成後顯存回到約 213 MiB。
- [x] P1-1 Breeze 管線穩定性基線通過（2026-09-14）；逐字稿與摘要品質待改善驗收。
  - **指令**：`UV_CACHE_DIR=/tmp/ai-meet-uv-cache uv run meet-eval pipeline '/home/stronger/音樂/260909_1631.mp3' ./var/p1-baseline-260909 --engine breeze --summary-model qwen3.5:9b --segment-seconds 600 --export txt,srt,vtt,md,json`
  - **結果**：9 段切段、168 segments、19,514 字元、總耗時 498.9 秒、ASR 約 328 秒、摘要約 96 秒、GPU 閒置 213 MiB、五格式匯出完整、raw\_asr 未被修改、0 次失敗。
  - **品質待驗收**：65/168 段（38.7%）有品質警示且原始逐字稿實際存在重複文字；摘要 0 decisions / 0 action\_items 不得視為已通過；CER/WER 待人工標註。
- [ ] 人工抽樣至少 20 個有品質警示段落及 20 個無警示段落，量測重複幻覺警示的精確率與漏報率。
- [ ] 以相同音檔比較 Breeze 預設 VAD 關閉與 VAD 開啟的結果；記錄漏字、靜音誤判、重複幻覺段落數、處理時間與顯存。
- [ ] 人工檢核摘要的決議／待辦漏擷取；建立正確答案後量測召回率。
- [ ] 完成上述品質檢核後，才開始 Qwen3-ASR／TEA-ASR 同集對照。
- [ ] 建立至少 60 分鐘人工標註集，涵蓋臺灣華語、英語、日語、臺語、混語、重疊發言與噪音。
  - **實作建議（抽樣切片法）**：無需逐字聽打完整會議，建議挑選 4-5 段具代表性（如：純華語、國臺混語、背景噪音、多人搶話）的 1-2 分鐘切片進行人工聽打即可。
  - **聽打原則**：忠於原音（包含吃螺絲與重複），存成純文字檔（如 `ref.txt`），並將對應的 AI 產出存為 `hyp.txt`。
- [ ] 量測 CER、WER、DER、摘要證據有效率，以及決議／待辦召回率。
  - **執行指令**：使用 `uv run meet-eval score ref.txt hyp.txt` 計算中文字錯率（CER），或附加 `--unit word` 計算英文詞錯率（WER）。
- [ ] 驗證 120 分鐘、500 MB 與最多 8 位講者的規劃容量。
- [ ] 建立 Qwen3-ASR、TEA-ASR-1.1、Breeze-ASR-26 的同集對照評測；確認 Ubuntu 上可用的推論方式、授權、模型來源與資源需求。
- [ ] 擴充評測欄位：中文／臺語 CER、英語 WER、日語字形正確率、混語與臺語關鍵詞正確率、處理時間、GPU 顯存峰值、系統 RAM、失敗率。
- [ ] 評估可選 Forced Alignment 原型；僅產生衍生字級時間資料，失敗時保留段落級時間軸，絕不修改 raw_asr。
- [ ] 評估可選 ONNX 講者分離原型；量測 DER、重疊發言與跨片段講者一致性。
- [ ] 針對模型輸出建立日語字形保護與臺灣繁體中文用詞檢查，不得對原始逐字稿套用全域簡繁轉換。

## P2：正式多人部署

- [ ] 登入、授權與多租戶資料隔離。
- [ ] TLS、CSRF 防護與速率限制。
- [ ] 正式背景工作佇列與取消機制。
- [ ] 關聯式資料庫與物件儲存。
- [ ] 稽核日誌、備份、還原及資料保存期限。
- [ ] 自動清理錄音、逐字稿、摘要與衍生檔案。

## 待確認需求

- [ ] 一般及最長會議時間。
- [ ] 典型及最大講者數。
- [ ] 錄音設備與音訊來源。
- [ ] 每月錄音時數與同時工作數。
- [ ] 個人版或團隊版。
- [ ] 錄音、逐字稿與摘要保存期限。
- [ ] 本機、雲端或混合部署。
- [ ] 每月營運與模型預算。

## AGY 本輪驗證

完成 P0 文件整理後執行：

```bash
UV_CACHE_DIR=/tmp/ai-meet-uv-cache uv run pytest
python3 -m compileall src/ tests/
node --check src/meet_in_multi_language/static/app.js
git diff --check
git ls-files var/ '*.mp3' '*.wav' '*.m4a' '*.webm'
```

再檢查舊資訊：

```bash
rg -n '25 MB|26214400|3\.5 GB|完全無壓力|TypeScript|關聯式資料庫|物件儲存' \
  README.md TODO.md .env.example src tests docs
```

搜尋結果須逐筆判斷；未來架構詞彙可以保留，但必須明確標示為尚未實作。

完成後回報修改檔案、測試結果、尚存限制、建議 commit 訊息與 `git status --short`，然後等待使用者確認。
