# 系統部署與資源調校指南

本文件彙整 `AI-MeetInMultiLanguage` 在實機硬體環境（NVIDIA GeForce RTX 3050 8 GiB 顯存、Ubuntu Linux）之部署實測數據、顯存生命週期管理、模型推論調校建議與常見問題排除指南。

---

## 一、硬體基線與顯存互斥架構

### 1. 目標硬體環境
- **作業系統**：Ubuntu Linux 24.04
- **主機記憶體**：46.9 GiB RAM
- **GPU 規格**：NVIDIA GeForce RTX 3050 Laptop GPU（實體顯存 8,192 MiB）
- **驅動與 CUDA**：Driver 580.173.02 / CUDA 12.x

### 2. 顯存互斥佇列原理
在 8 GiB 顯存的主機上，若同時載入語音辨識模型與大語言模型，勢必觸發 CUDA OOM：
- **Breeze ASR**（`paulpengtw/faster-whisper-Breeze-ASR-26`）：轉錄推論峰值約 **2,303 MiB**。
- **Ollama 9B**（`qwen3.5:9b`）：摘要推論峰值約 **6,358 ~ 6,972 MiB**。
- **兩者相加**：至少需要 8,661 MiB，超出 8,192 MiB 上限。

因此，本系統設計了單一單例（Singleton）的非同步工作佇列 **`GpuWorkQueue`**（實作於 [`src/meet_in_multi_language/gpu.py`](../src/meet_in_multi_language/gpu.py)）：
1. **ASR 階段**：自 Speaches 發起轉錄前，主動呼叫 Ollama API（`keep_alive: 0`）卸載 Ollama 模型並釋放顯存。
2. **切換階段**：轉錄完成後，主動發送 HTTP DELETE 請求通知 Speaches 卸載 Whisper 模型（`DELETE /api/ps/{model}`），確保顯存降回基礎閒置水準（約 100 ~ 200 MiB）。
3. **摘要階段**：載入 Ollama 9B 模型執行結構化摘要，完成後立即發送 `keep_alive: 0` 歸零卸載，安全恢復顯存。
4. **同類別模型切換**：若前一任務為 `qwen3.8:latest`，下一任務切換為 `qwen3.5:9b`，佇列會精確追蹤實際執行的模型名稱，並於載入前主動將前一模型釋放。

---

## 二、Speaches 容器原生死鎖問題與修補

### 1. 死鎖根因（Root Cause）
官方 Docker 映像檔 `ghcr.io/speaches-ai/speaches:latest-cuda` 中的 Whisper 執行器模型管理器（`/home/ubuntu/speaches/src/speaches/executors/whisper/model_manager.py`）使用了不可重入鎖：
```python
self._lock = threading.Lock()
```
在執行 `unload_model()` 時，外層函式取得 `self._lock`，卸載成功後觸發回呼函式 `_handle_model_unloaded()`，該回呼在同一執行緒內再次嘗試取得 `self._lock`，造成 100% 永久死鎖，導致外部呼叫 `DELETE /api/ps/{model}` 永久卡死。

### 2. 修補方案（已實施並驗證）
將鎖更換為可重入鎖（`threading.RLock`）：
```python
self._lock = threading.RLock()
```
修復並重啟容器後，Speaches 外部卸載請求能在數百毫秒內回傳 `204 No Content`，徹底排除死鎖隱患。

---

## 三、Breeze ASR 轉錄品質與參數調校

針對真實財務會議語音（100 秒自發語音）以人工標準 Ground Truth 進行字詞錯誤率評測：

| 實驗配置 | CER（字元錯誤率） | WER（詞錯誤率） | 金額與數字表現 | 專有名詞表現 | 評估結論 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **預設 Breeze（無 prompt / 無 hotwords）** | **5.82%** | **28.57%** | 100% 精確 | 存在同音字（如「監視」/「監事」） | **最佳基線**，遠優於 P1 目標 ≤ 15% |
| **外部熱詞提示（注入自訂 hotwords）** | 13.36% | 34.29% | 劣化（引發「兩億萬塊」重複邊界錯誤） | 部分改善 | 未微調之熱詞會干擾解碼分佈，**不建議盲目注入** |

### 調校準則：
1. **提示詞（Prompt）與熱詞（Hotwords）**：
   - Breeze-ASR-26 在真實自發語音上，預設「不傳 prompt、不傳 hotwords」之機率分佈最為自然穩定。
2. **語音活動偵測（VAD）**：
   - 經多段實測，全域開啟 VAD 會將低音量或輕微停頓的發言（如財務報告前段）完全裁截。目前 Breeze profile 先以 **VAD 關閉** 為保守穩定預設；純靜音片段在關閉 VAD 下亦能輸出空白或短字詞。
3. **長音訊切段（Chunking）**：
   - 超過 600 秒（10 分鐘）之錄音，建議使用 `prepare_audio_chunks` 進行自動停頓切段，並維護時間戳偏移。

---

## 四、Ollama 摘要模型與顯存調校

1. **模型選擇**：
   - **`qwen3.5:9b`（預設推薦）**：在 8 GiB 顯存下占用約 6.3 GiB，推論速度快（約 15 ~ 25 秒），且能嚴格遵守 JSON Schema 與繁體中文用語規範。
   - **`qwen3.8:latest`（品質對照）**：內容細節更完整，但速度較慢，適合離線或品質對照時選用。
2. **長逐字稿分段摘要（Map-Reduce）**：
   - 單次上下文超過 12,000 字元時，系統自動啟用分段摘要抽取證據，最後整合時嚴格核對原文 evidence ID，禁止模型憑空捏造負責人或結論。
3. **推論釋放（Keep-Alive）**：
   - 每次推論請求一律附帶 `keep_alive: 0`，推論完畢後即刻通知 Ollama 卸載顯存，確保系統顯存隨時維持在安全餘裕（1.2 GiB 以上）。

---

## 五、系統常用指令速查

### 1. 系統自我診斷（Pre-flight Check）
```bash
# 終端文字報告
meet-eval doctor

# JSON 格式輸出
meet-eval doctor --json
```

### 2. 端到端批次處理（長音訊切段、轉錄、摘要、多格式匯出）
```bash
meet-eval pipeline /path/to/meeting.mp3 ./output_dir \
  --engine breeze \
  --summary-model qwen3.5:9b \
  --segment-seconds 600 \
  --export txt,srt,vtt,md,json
```

### 3. 字詞錯誤率（CER/WER）客觀評測
```bash
meet-eval score ground_truth.txt hypothesis.txt --unit character
```

### 4. 啟動 Web 檢閱服務
```bash
uv run uvicorn meet_in_multi_language.api:app --host 127.0.0.1 --port 8000
```
瀏覽器開啟 `http://127.0.0.1:8000` 即可操作錄音上傳、即時回聽、發音段落同步高亮、講者更名、關鍵字搜尋與多格式匯出。
