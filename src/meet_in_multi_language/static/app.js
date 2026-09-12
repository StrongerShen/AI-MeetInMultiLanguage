const form = document.querySelector("#upload-form");
const statusNode = document.querySelector("#service-status");
const messageNode = document.querySelector("#form-message");
const runList = document.querySelector("#run-list");
const refreshButton = document.querySelector("#refresh-button");
const submitButton = document.querySelector("#submit-button");
const engineSelect = document.querySelector("#engine-select");

const speachesStatusNode = document.querySelector("#speaches-status");
const ollamaStatusNode = document.querySelector("#ollama-status");
const openaiStatusNode = document.querySelector("#openai-status");
const gpuBadge = document.querySelector("#gpu-badge");
const gpuQueueDetail = document.querySelector("#gpu-queue-detail");

let systemHealth = null;
const selectedRevisionPerRun = {};

const statusLabels = {
  queued: "等待處理",
  transcribing: "轉錄中",
  summarizing: "摘要中",
  completed: "已完成",
  failed: "失敗",
};

const engineLabels = {
  breeze: "Breeze ASR（本機臺語／國臺混用）",
  "gpt-4o-transcribe-diarize": "OpenAI Diarize（含講者）",
  "gpt-transcribe": "OpenAI Transcribe（混語提示）",
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function formatMs(ms) {
  if (ms == null) return "";
  const totalSeconds = Math.floor(ms / 1000);
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${String(minutes).padStart(2, "0")}:${String(seconds).padStart(2, "0")}`;
}

async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || "發生未預期的錯誤";
  } catch {
    return "伺服器回應格式不正確";
  }
}

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) throw new Error("Health check failed");
    systemHealth = await response.json();

    speachesStatusNode.textContent = systemHealth.speaches_available ? "已連線 (127.0.0.1:8001)" : "未連線";
    ollamaStatusNode.textContent = systemHealth.ollama_available ? `已連線 (${systemHealth.ollama_default_model})` : "未連線";
    openaiStatusNode.textContent = systemHealth.openai_configured ? "已設定金鑰" : "未設定金鑰";

    const gpu = systemHealth.gpu_queue || {};
    if (gpu.is_busy) {
      gpuBadge.textContent = "GPU 忙碌中";
      gpuBadge.className = "badge busy";
      gpuQueueDetail.textContent = `執行：${escapeHtml(gpu.active_task || "處理中")}（排隊數：${gpu.queue_length || 0}）`;
    } else {
      gpuBadge.textContent = "GPU 閒置";
      gpuBadge.className = "badge idle";
      gpuQueueDetail.textContent = gpu.queue_length > 0 ? `等待排程（排隊數：${gpu.queue_length}）` : "無進行中工作";
    }

    const limitMb = Math.floor(systemHealth.max_upload_bytes / 1024 / 1024);
    statusNode.className = "status";
    statusNode.textContent = `伺服器已就緒；目前單檔限制 ${limitMb} MB。Breeze 與 Ollama 透過單一 GPU 互斥佇列保護顯存。`;
    validateEngineOption();
  } catch {
    statusNode.textContent = "無法連線至伺服器，請確認後端服務是否正在執行。";
    statusNode.className = "status error";
    gpuBadge.textContent = "斷線";
    gpuBadge.className = "badge error";
  }
}

function validateEngineOption() {
  if (!systemHealth) return;
  const chosen = engineSelect.value;
  if ((chosen === "gpt-4o-transcribe-diarize" || chosen === "gpt-transcribe") && !systemHealth.openai_configured) {
    submitButton.disabled = true;
    messageNode.textContent = "所選引擎需要 OPENAI_API_KEY，但伺服器尚未設定。請改選 Breeze ASR 或在伺服器設定金鑰。";
    messageNode.className = "error";
  } else {
    submitButton.disabled = false;
    messageNode.textContent = "";
    messageNode.className = "";
  }
}

engineSelect.addEventListener("change", validateEngineOption);

function renderSegments(segments) {
  if (!segments || !segments.length) return "<p>（無段落資訊）</p>";
  return segments.map((seg) => {
    const timeStr = seg.start_ms != null ? `[${formatMs(seg.start_ms)}]` : "";
    const speakerStr = seg.speaker ? `[${escapeHtml(seg.speaker)}]` : "";
    const flagsHtml = (seg.quality_flags || []).map(f => `<span class="quality-flag" title="品質警示">${escapeHtml(f)}</span>`).join("");
    return `
      <div class="segment-row" id="seg-row-${escapeHtml(seg.segment_id)}" data-seg-id="${escapeHtml(seg.segment_id)}">
        <span class="segment-meta">
          <span class="segment-tag">${escapeHtml(seg.segment_id)}</span>
          ${timeStr ? `<span>${escapeHtml(timeStr)}</span>` : ""}
          ${speakerStr ? `<span class="segment-speaker">${escapeHtml(speakerStr)}</span>` : ""}
        </span>
        <span class="segment-text">${escapeHtml(seg.text)}</span>
        ${flagsHtml}
      </div>
    `;
  }).join("");
}

function renderSummary(summary, runId) {
  if (!summary) {
    return `
      <div class="summary-container">
        <div class="summary-header">
          <h4>會議摘要分析</h4>
          <button type="button" class="small secondary" onclick="triggerSummary('${runId}')">立即以 Ollama 產生摘要</button>
        </div>
        <p class="hint">尚未產生摘要。點擊上方按鈕以 Ollama 進行繁體中文結構化摘要與證據提取。</p>
      </div>
    `;
  }

  const topicsHtml = (summary.topics || []).map(t => `
    <li>
      <strong>${escapeHtml(t.title)}：</strong>${escapeHtml(t.summary)}
      ${(t.evidence_ids || []).map(id => `<button type="button" class="evidence-link" data-run-id="${escapeHtml(runId)}" data-evidence-id="${escapeHtml(id)}">${escapeHtml(id)}</button>`).join(" ")}
    </li>
  `).join("");

  const decisionsHtml = (summary.decisions || []).map(d => `
    <li>
      ${escapeHtml(d.text)}
      ${(d.evidence_ids || []).map(id => `<button type="button" class="evidence-link" data-run-id="${escapeHtml(runId)}" data-evidence-id="${escapeHtml(id)}">${escapeHtml(id)}</button>`).join(" ")}
    </li>
  `).join("");

  const actionsHtml = (summary.action_items || []).map(a => `
    <li>
      <strong>任務：</strong>${escapeHtml(a.task)}
      · <strong>負責人：</strong>${escapeHtml(a.owner || "未指定")}
      · <strong>期限：</strong>${escapeHtml(a.due_date || a.original_due_text || "未指定")}
      ${(a.evidence_ids || []).map(id => `<button type="button" class="evidence-link" data-run-id="${escapeHtml(runId)}" data-evidence-id="${escapeHtml(id)}">${escapeHtml(id)}</button>`).join(" ")}
    </li>
  `).join("");

  const questionsHtml = (summary.open_questions || []).map(q => `
    <li>
      ${escapeHtml(q.text)}
      ${(q.evidence_ids || []).map(id => `<button type="button" class="evidence-link" data-run-id="${escapeHtml(runId)}" data-evidence-id="${escapeHtml(id)}">${escapeHtml(id)}</button>`).join(" ")}
    </li>
  `).join("");

  return `
    <div class="summary-container">
      <div class="summary-header">
        <h4>會議摘要與分析（模型：${escapeHtml(summary.model)}）</h4>
        <button type="button" class="small secondary" onclick="triggerSummary('${runId}')">重新產生摘要</button>
      </div>
      <div class="summary-overview">
        <strong>總覽：</strong>${escapeHtml(summary.overview)}
      </div>
      ${topicsHtml ? `<div class="summary-section"><h5>討論議題</h5><ul class="summary-list">${topicsHtml}</ul></div>` : ""}
      ${decisionsHtml ? `<div class="summary-section"><h5>重要決議</h5><ul class="summary-list">${decisionsHtml}</ul></div>` : ""}
      ${actionsHtml ? `<div class="summary-section"><h5>待辦事項</h5><ul class="summary-list">${actionsHtml}</ul></div>` : ""}
      ${questionsHtml ? `<div class="summary-section"><h5>未解問題與待確認事項</h5><ul class="summary-list">${questionsHtml}</ul></div>` : ""}
    </div>
  `;
}

window.highlightSegment = function(runId, segId) {
  const card = document.getElementById(`run-card-${runId}`);
  if (!card) return;
  const targetId = segId.replace(/^\[+|\]+$/g, "");
  // 嘗試匹配 seg-001 或 segment-1
  const segmentRows = Array.from(card.querySelectorAll(".segment-row"));
  let el = segmentRows.find(row => row.dataset.segId === targetId);
  if (!el) {
    const num = parseInt(targetId.replace(/[^0-9]/g, ""), 10);
    if (!isNaN(num)) {
      const aliases = new Set([`segment-${num}`, `seg-${String(num).padStart(3, "0")}`]);
      el = segmentRows.find(row => aliases.has(row.dataset.segId));
    }
  }
  if (el) {
    card.querySelectorAll(".segment-row").forEach(r => r.classList.remove("highlighted"));
    el.classList.add("highlighted");
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }
};

window.triggerSummary = async function(runId) {
  const model = document.querySelector("#summary-model-select")?.value || "qwen3.5:9b";
  const body = new FormData();
  body.append("model", model);
  try {
    const response = await fetch(`/api/runs/${runId}/summary`, { method: "POST", body });
    if (!response.ok) {
      alert(`要求摘要失敗：${await readError(response)}`);
    } else {
      await loadRuns();
    }
  } catch (err) {
    alert(`網路連線失敗：${err.message}`);
  }
};

const openCorrectionPanels = new Set();
const correctionDrafts = {};

window.toggleCorrectionPanel = function(runId) {
  const panel = document.getElementById(`correction-panel-${runId}`);
  if (!panel) return;
  const isCurrentlyOpen = openCorrectionPanels.has(runId);
  if (isCurrentlyOpen) {
    openCorrectionPanels.delete(runId);
    delete correctionDrafts[runId];
    panel.style.display = "none";
  } else {
    openCorrectionPanels.add(runId);
    panel.style.display = "grid";
    const textarea = document.getElementById(`correction-text-${runId}`);
    if (textarea) textarea.focus();
  }
};

window.onCorrectionInput = function(runId, text) {
  correctionDrafts[runId] = text;
};

window.submitCorrection = async function(runId) {
  const textarea = document.getElementById(`correction-text-${runId}`);
  const text = textarea ? textarea.value.trim() : "";
  if (!text) {
    alert("校訂文字不可為空");
    return;
  }
  const submitBtn = document.getElementById(`submit-correct-${runId}`);
  if (submitBtn) submitBtn.disabled = true;

  const body = new FormData();
  body.append("corrected_text", text);
  const sourceRevisionId = selectedRevisionPerRun[runId];
  if (sourceRevisionId) body.append("source_revision_id", sourceRevisionId);
  try {
    const response = await fetch(`/api/runs/${runId}/revisions/correct`, { method: "POST", body });
    if (!response.ok) {
      alert(`新增校訂版本失敗：${await readError(response)}`);
      if (submitBtn) submitBtn.disabled = false;
    } else {
      openCorrectionPanels.delete(runId);
      delete correctionDrafts[runId];
      await loadRuns();
    }
  } catch (err) {
    alert(`網路連線失敗：${err.message}`);
    if (submitBtn) submitBtn.disabled = false;
  }
};

window.switchRevision = function(runId, revisionId) {
  selectedRevisionPerRun[runId] = revisionId;
  delete correctionDrafts[runId];
  loadRuns();
};

async function loadRuns() {
  let activeTextareaId = null;
  let selStart = null;
  let selEnd = null;
  if (
    document.activeElement &&
    document.activeElement.tagName === "TEXTAREA" &&
    document.activeElement.id &&
    document.activeElement.id.startsWith("correction-text-")
  ) {
    activeTextareaId = document.activeElement.id;
    selStart = document.activeElement.selectionStart;
    selEnd = document.activeElement.selectionEnd;
  }

  let runs;
  try {
    const response = await fetch("/api/runs");
    if (!response.ok) throw new Error("伺服器回應錯誤");
    runs = await response.json();
  } catch (error) {
    runList.innerHTML = `<p class="error">無法取得工作紀錄：${escapeHtml(error.message)}</p>`;
    return;
  }
  if (!runs.length) {
    runList.innerHTML = "<p>目前尚無轉錄工作。請從上方上傳錄音檔案。</p>";
    return;
  }

  runList.innerHTML = runs.map((run) => {
    const revisions = run.revisions || (run.result ? [run.result] : []);
    const selectedRevId = selectedRevisionPerRun[run.run_id] || (run.result?.revision_id) || (run.raw_asr?.revision_id);
    const currentRev = revisions.find(r => r.revision_id === selectedRevId) || run.result || run.raw_asr;

    const revisionOptionsHtml = revisions.map(r => {
      const isRaw = r.revision_kind === "raw_asr";
      const isLlm = r.revision_kind === "llm_corrected";
      const kindLabel = isRaw ? "原始 ASR (raw_asr)" : (isLlm ? "LLM 校訂版" : "人工校正");
      const selectedAttr = r.revision_id === currentRev?.revision_id ? "selected" : "";
      return `<option value="${escapeHtml(r.revision_id)}" ${selectedAttr}>${kindLabel} - ${escapeHtml(r.revision_id)}</option>`;
    }).join("");

    const isPanelOpen = openCorrectionPanels.has(run.run_id);
    const draftText = correctionDrafts[run.run_id] !== undefined ? correctionDrafts[run.run_id] : currentRev?.text || "";

    return `
      <article class="run-card" id="run-card-${escapeHtml(run.run_id)}">
        <div class="card-header">
          <div>
            <h3 class="card-title">${escapeHtml(run.original_filename)}</h3>
            <p class="meta">
              引擎：${escapeHtml(engineLabels[run.engine] || run.engine)} ·
              建立於：${new Date(run.created_at).toLocaleString("zh-TW", { hour12: false })}
            </p>
          </div>
          <span class="status-pill ${escapeHtml(run.status)}">
            ${escapeHtml(statusLabels[run.status] || run.status)}
          </span>
        </div>

        ${run.error ? `<p class="status error">${escapeHtml(run.error)}</p>` : ""}

        ${currentRev ? `
          <div class="revision-toolbar">
            <div class="revision-selector-group">
              <label><strong>逐字稿版本：</strong></label>
              <select onchange="switchRevision('${escapeHtml(run.run_id)}', this.value)">
                ${revisionOptionsHtml}
              </select>
              ${currentRev.revision_kind === "raw_asr" ? `<span class="raw-badge" title="受保護版本，不可覆蓋">原始稿 (raw_asr)</span>` : ""}
            </div>
            <div class="revision-actions">
              <button type="button" class="small secondary" onclick="toggleCorrectionPanel('${escapeHtml(run.run_id)}')">建立人工校訂版</button>
            </div>
          </div>

          <div class="correct-modal" id="correction-panel-${escapeHtml(run.run_id)}" style="display: ${isPanelOpen ? 'grid' : 'none'};">
            <label><strong>編輯所選版本並另存人工校訂稿（保留原始 raw_asr）：</strong></label>
            <textarea id="correction-text-${escapeHtml(run.run_id)}" rows="4" oninput="onCorrectionInput('${escapeHtml(run.run_id)}', this.value)">${escapeHtml(draftText)}</textarea>
            <div>
              <button type="button" class="small" id="submit-correct-${escapeHtml(run.run_id)}" onclick="submitCorrection('${escapeHtml(run.run_id)}')">儲存為新版本</button>
              <button type="button" class="small secondary" onclick="toggleCorrectionPanel('${escapeHtml(run.run_id)}')">取消</button>
            </div>
          </div>

          <div class="transcript-box">
            ${renderSegments(currentRev.segments)}
          </div>
        ` : (run.status === "queued" || run.status === "transcribing" ? `<p class="hint">正在排隊／轉錄音訊中，請稍候…</p>` : "")}

        ${run.status === "completed" || run.status === "summarizing" || run.summary ? renderSummary(run.summary, run.run_id) : ""}
      </article>
    `;
  }).join("");

  if (activeTextareaId) {
    const activeEl = document.getElementById(activeTextareaId);
    if (activeEl) {
      activeEl.focus();
      if (typeof selStart === "number" && typeof selEnd === "number") {
        activeEl.setSelectionRange(selStart, selEnd);
      }
    }
  }
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  messageNode.textContent = "正在上傳音訊檔案並排入工作佇列…";
  messageNode.className = "status";

  const formData = new FormData(form);
  const autoSummaryCheck = document.querySelector("#auto-summary-checkbox");
  formData.set("auto_summary", autoSummaryCheck && autoSummaryCheck.checked ? "true" : "false");

  try {
    const response = await fetch("/api/runs", { method: "POST", body: formData });
    if (!response.ok) {
      messageNode.textContent = await readError(response);
      messageNode.className = "status error";
    } else {
      messageNode.textContent = "已成功上傳並加入佇列！";
      messageNode.className = "status";
      form.reset();
      // 重置預設值
      if (autoSummaryCheck) autoSummaryCheck.checked = true;
      await loadRuns();
      await loadHealth();
    }
  } catch (err) {
    messageNode.textContent = `網路連線發生問題：${err.message}`;
    messageNode.className = "status error";
  } finally {
    submitButton.disabled = false;
  }
});

refreshButton.addEventListener("click", () => {
  loadRuns();
  loadHealth();
});

runList.addEventListener("click", (event) => {
  const link = event.target.closest(".evidence-link");
  if (!link || !runList.contains(link)) return;
  highlightSegment(link.dataset.runId, link.dataset.evidenceId);
});

loadHealth();
loadRuns();
setInterval(() => {
  loadHealth();
  loadRuns();
}, 4000);
