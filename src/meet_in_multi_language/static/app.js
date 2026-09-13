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

    const limitMiB = Math.floor(systemHealth.max_upload_bytes / 1024 / 1024);
    statusNode.className = "status";
    statusNode.textContent = `伺服器已就緒；目前單檔限制 ${limitMiB} MiB。Breeze 與 Ollama 透過單一 GPU 互斥佇列保護顯存。`;
    const hintNode = document.querySelector("#upload-limit-hint");
    if (hintNode) {
      hintNode.textContent = `支援 WAV、MP3、M4A 等格式，目前原型單檔上限 ${limitMiB} MiB。`;
    }
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

const openSegmentEdits = {};

function renderSegments(segments, runId) {
  if (!segments || !segments.length) return "<p>（無段落資訊）</p>";
  return segments.map((seg, index) => {
    const seqNum = index + 1;
    const timeStr = seg.start_ms != null ? formatMs(seg.start_ms) : "";
    const timeButton = timeStr ? `
      <button type="button" class="segment-time-btn" title="點擊跳轉播放此段落" data-action="play-segment" data-run-id="${escapeHtml(runId)}" data-start-ms="${seg.start_ms || 0}" data-seg-id="${escapeHtml(seg.segment_id)}">
        ▶ ${escapeHtml(timeStr)}
      </button>
    ` : "";
    const rawSpeaker = seg.speaker || "";
    const speakerHtml = rawSpeaker ? `
      <span class="speaker-tag" role="button" tabindex="0" title="點擊更名此講者" data-action="open-rename-speaker" data-run-id="${escapeHtml(runId)}" data-speaker="${escapeHtml(rawSpeaker)}">[${escapeHtml(rawSpeaker)}]</span>
    ` : "";
    const flagsHtml = (seg.quality_flags || []).map(f => `<span class="quality-flag" title="品質警示">${escapeHtml(f)}</span>`).join("");
    const startMs = seg.start_ms != null ? seg.start_ms : "";
    const endMs = seg.end_ms != null ? seg.end_ms : (seg.start_ms != null ? seg.start_ms + 4000 : "");
    const segLabel = seg.segment_id.startsWith("chunk-") ? `#${seqNum}` : escapeHtml(seg.segment_id);

    const editKey = `${runId}:${seg.segment_id}`;
    const editState = openSegmentEdits[editKey];
    const isEditing = !!editState;

    const inlineEditHtml = isEditing ? `
      <div class="segment-inline-edit" id="inline-edit-${escapeHtml(runId)}-${escapeHtml(seg.segment_id)}">
        <label class="hint"><strong>快速校訂段落 #${seqNum}（另存新版本，保留原始 raw_asr）：</strong></label>
        <textarea id="inline-text-${escapeHtml(runId)}-${escapeHtml(seg.segment_id)}" rows="2" placeholder="輸入校訂後的段落文字...">${escapeHtml(editState.text)}</textarea>
        <div class="inline-edit-controls">
          <label class="hint">講者：</label>
          <input type="text" id="inline-spk-${escapeHtml(runId)}-${escapeHtml(seg.segment_id)}" value="${escapeHtml(editState.speaker)}" placeholder="講者名稱" />
          <button type="button" class="small" id="submit-inline-${escapeHtml(runId)}-${escapeHtml(seg.segment_id)}" data-action="submit-segment-edit" data-run-id="${escapeHtml(runId)}" data-seg-id="${escapeHtml(seg.segment_id)}">儲存校訂</button>
          <button type="button" class="small secondary" data-action="toggle-segment-edit" data-run-id="${escapeHtml(runId)}" data-seg-id="${escapeHtml(seg.segment_id)}">取消</button>
        </div>
      </div>
    ` : "";

    return `
      <div class="segment-row" id="seg-row-${escapeHtml(seg.segment_id)}" data-seg-id="${escapeHtml(seg.segment_id)}" data-seq-index="${seqNum}" data-start-ms="${startMs}" data-end-ms="${endMs}">
        <span class="segment-meta">
          <span class="segment-tag" title="段落 #${seqNum} (${escapeHtml(seg.segment_id)})">${segLabel}</span>
          ${timeButton}
          ${speakerHtml}
        </span>
        <span class="segment-text">${escapeHtml(seg.text)}</span>
        ${flagsHtml}
        <span class="segment-actions">
          <button type="button" class="segment-action-btn" title="快速校訂此段文字" data-action="toggle-segment-edit" data-run-id="${escapeHtml(runId)}" data-seg-id="${escapeHtml(seg.segment_id)}">✏️ 校訂</button>
          <button type="button" class="segment-action-btn" title="帶入此段文字至全篇校訂草稿" data-action="copy-to-draft" data-run-id="${escapeHtml(runId)}">📋 帶入草稿</button>
        </span>
        ${inlineEditHtml}
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
          <button type="button" class="small secondary" data-action="trigger-summary" data-run-id="${escapeHtml(runId)}">立即以 Ollama 產生摘要</button>
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
      <strong>${escapeHtml(a.task)}</strong>
      ${a.owner ? `（負責人：${escapeHtml(a.owner)}）` : ""}
      ${a.due_date ? `（期限：${escapeHtml(a.due_date)}）` : (a.original_due_text ? `（期限描述：${escapeHtml(a.original_due_text)}）` : "")}
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
        <button type="button" class="small secondary" data-action="trigger-summary" data-run-id="${escapeHtml(runId)}">重新產生摘要</button>
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
  const targetId = String(segId).replace(/^\[+|\]+$/g, "");
  // 嘗試匹配 seg-001、segment-1 或跨切段序號，完全使用 dataset 比對避免 selector 注入
  const segmentRows = Array.from(card.querySelectorAll(".segment-row"));
  let el = segmentRows.find(row => row.dataset.segId === targetId);
  if (!el) {
    const num = parseInt(targetId.replace(/[^0-9]/g, ""), 10);
    if (!isNaN(num)) {
      const aliases = new Set([`segment-${num}`, `seg-${String(num).padStart(3, "0")}`]);
      el = segmentRows.find(row => aliases.has(row.dataset.segId) || row.dataset.seqIndex === String(num));
      if (!el && num >= 1 && num <= segmentRows.length) {
        el = segmentRows[num - 1];
      }
    }
  }
  if (el) {
    segmentRows.forEach(r => r.classList.remove("highlighted"));
    el.classList.add("highlighted");
    el.scrollIntoView({ behavior: "smooth", block: "nearest" });

    // 若有時間資訊，同步更新播放器位置
    const startMs = Number(el.dataset.startMs);
    if (!isNaN(startMs) && startMs > 0) {
      const player = document.getElementById(`audio-player-${runId}`);
      if (player && player.paused) {
        player.currentTime = startMs / 1000;
      }
    }
  }
};

window.playAudioAt = function(runId, startMs, segId) {
  const player = document.getElementById(`audio-player-${runId}`);
  if (player) {
    player.currentTime = (startMs || 0) / 1000;
    player.play().catch(() => {});
  }
  if (segId) {
    window.highlightSegment(runId, segId);
  }
};

window.onAudioTimeUpdate = function(runId, currentTime) {
  const card = document.getElementById(`run-card-${runId}`);
  if (!card) return;
  const currentMs = Math.floor(currentTime * 1000);
  const rows = card.querySelectorAll(".segment-row");
  rows.forEach(row => {
    const start = Number(row.dataset.startMs);
    const end = Number(row.dataset.endMs);
    if (!isNaN(start) && !isNaN(end) && currentMs >= start && currentMs < end) {
      if (!row.classList.contains("active-playing")) {
        row.classList.add("active-playing");
      }
    } else {
      row.classList.remove("active-playing");
    }
  });
};

window.triggerSummary = async function(runId) {
  const model = document.querySelector("#summary-model-select")?.value || "qwen3.5:9b";
  const body = new FormData();
  body.append("model", model);
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/summary`, { method: "POST", body });
    if (!response.ok) {
      alert(`要求摘要失敗：${await readError(response)}`);
    } else {
      await loadRuns();
    }
  } catch (err) {
    alert(`網路連線失敗：${err.message}`);
  }
};

window.triggerExport = function(runId) {
  const select = document.getElementById(`export-format-${runId}`);
  const fmt = select ? select.value : "txt";
  const revId = selectedRevisionPerRun[runId] || "";
  const query = new URLSearchParams({ format: fmt });
  if (revId) query.set("revision_id", revId);
  window.open(`/api/runs/${encodeURIComponent(runId)}/export?${query.toString()}`, "_blank");
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
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/revisions/correct`, { method: "POST", body });
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

const openRenamePanels = new Set();
const renameSpeakerDefaults = {};

window.openRenameSpeakerModal = function(runId, defaultSpeaker) {
  openRenamePanels.add(runId);
  renameSpeakerDefaults[runId] = defaultSpeaker;
  loadRuns();
};

window.toggleRenamePanel = function(runId) {
  if (openRenamePanels.has(runId)) {
    openRenamePanels.delete(runId);
    delete renameSpeakerDefaults[runId];
  } else {
    openRenamePanels.add(runId);
  }
  loadRuns();
};

window.submitRenameSpeaker = async function(runId) {
  const oldSpkInput = document.getElementById(`rename-old-${runId}`);
  const newSpkInput = document.getElementById(`rename-new-${runId}`);
  const oldSpk = oldSpkInput ? oldSpkInput.value.trim() : "";
  const newSpk = newSpkInput ? newSpkInput.value.trim() : "";
  if (!newSpk) {
    alert("新講者名稱不可為空");
    return;
  }
  const submitBtn = document.getElementById(`submit-rename-${runId}`);
  if (submitBtn) submitBtn.disabled = true;

  const body = new FormData();
  body.append("old_speaker", oldSpk);
  body.append("new_speaker", newSpk);
  const sourceRevisionId = selectedRevisionPerRun[runId];
  if (sourceRevisionId) body.append("source_revision_id", sourceRevisionId);

  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/speakers/rename`, { method: "POST", body });
    if (!response.ok) {
      alert(`更名失敗：${await readError(response)}`);
      if (submitBtn) submitBtn.disabled = false;
    } else {
      openRenamePanels.delete(runId);
      delete renameSpeakerDefaults[runId];
      await loadRuns();
    }
  } catch (err) {
    alert(`網路連線失敗：${err.message}`);
    if (submitBtn) submitBtn.disabled = false;
  }
};

window.deleteRun = async function(runId, filename) {
  if (!confirm(`確定要刪除「${filename || runId}」這筆工作與錄音檔案嗎？此操作無法復原。`)) {
    return;
  }
  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}`, { method: "DELETE" });
    if (!response.ok) {
      alert(`刪除失敗：${await readError(response)}`);
    } else {
      delete selectedRevisionPerRun[runId];
      openCorrectionPanels.delete(runId);
      openRenamePanels.delete(runId);
      delete correctionDrafts[runId];
      delete renameSpeakerDefaults[runId];
      await loadRuns();
      await loadHealth();
    }
  } catch (err) {
    alert(`網路連線失敗：${err.message}`);
  }
};

const filterStatePerRun = {};

window.onFilterInput = function(runId) {
  const card = document.getElementById(`run-card-${runId}`);
  if (!card) return;
  const keyword = (document.getElementById(`search-input-${runId}`)?.value || "").toLowerCase().trim();
  const speaker = document.getElementById(`speaker-filter-${runId}`)?.value || "";
  const onlyFlagged = document.getElementById(`flag-filter-${runId}`)?.checked || false;

  filterStatePerRun[runId] = { keyword, speaker, onlyFlagged };

  const rows = card.querySelectorAll(".segment-row");
  rows.forEach(row => {
    const textNode = row.querySelector(".segment-text");
    const spkNode = row.querySelector(".speaker-tag");
    const flagsNode = row.querySelectorAll(".quality-flag");
    const text = (textNode?.textContent || "").toLowerCase();
    const spk = spkNode?.textContent?.replace(/[\[\]]/g, "") || "";
    const hasFlags = flagsNode.length > 0;

    let match = true;
    if (keyword && !text.includes(keyword)) match = false;
    if (speaker && spk !== speaker) match = false;
    if (onlyFlagged && !hasFlags) match = false;

    if (match) {
      row.classList.remove("hidden-by-filter");
    } else {
      row.classList.add("hidden-by-filter");
    }
  });
};

window.copySegmentToDraft = function(runId, text) {
  openCorrectionPanels.add(runId);
  const panel = document.getElementById(`correction-panel-${runId}`);
  const textarea = document.getElementById(`correction-text-${runId}`);
  if (panel) panel.style.display = "grid";
  if (textarea) {
    const current = textarea.value.trim();
    textarea.value = current ? `${current}\n${text}` : text;
    correctionDrafts[runId] = textarea.value;
    textarea.focus();
    textarea.scrollTop = textarea.scrollHeight;
  }
};

window.toggleSegmentEdit = function(runId, segId) {
  const key = `${runId}:${segId}`;
  if (openSegmentEdits[key]) {
    delete openSegmentEdits[key];
  } else {
    const card = document.getElementById(`run-card-${runId}`);
    const rows = card ? Array.from(card.querySelectorAll(".segment-row")) : [];
    const row = rows.find(r => r.dataset.segId === segId);
    const textNode = row ? row.querySelector(".segment-text") : null;
    const spkNode = row ? row.querySelector(".speaker-tag") : null;
    const text = textNode ? textNode.textContent.trim() : "";
    const spk = spkNode ? spkNode.textContent.replace(/[\[\]]/g, "").trim() : "";
    openSegmentEdits[key] = { text, speaker: spk };
  }
  loadRuns();
};

window.submitSegmentEdit = async function(runId, segId) {
  const textInput = document.getElementById(`inline-text-${runId}-${segId}`);
  const spkInput = document.getElementById(`inline-spk-${runId}-${segId}`);
  const submitBtn = document.getElementById(`submit-inline-${runId}-${segId}`);
  const text = textInput ? textInput.value.trim() : "";
  const speaker = spkInput ? spkInput.value.trim() : "";
  if (!text) {
    alert("校訂文字不可為空");
    return;
  }
  if (submitBtn) submitBtn.disabled = true;

  const currentRevId = selectedRevisionPerRun[runId];
  const formData = new URLSearchParams();
  formData.append("corrected_text", text);
  if (speaker) formData.append("speaker", speaker);
  if (currentRevId) formData.append("source_revision_id", currentRevId);

  try {
    const response = await fetch(`/api/runs/${encodeURIComponent(runId)}/segments/${encodeURIComponent(segId)}/correct`, {
      method: "POST",
      headers: { "Content-Type": "application/x-www-form-urlencoded" },
      body: formData.toString(),
    });
    if (!response.ok) {
      alert(`校訂失敗：${await readError(response)}`);
      if (submitBtn) submitBtn.disabled = false;
      return;
    }
    const updatedRun = await response.json();
    delete openSegmentEdits[`${runId}:${segId}`];
    if (updatedRun.result && updatedRun.result.revision_id) {
      selectedRevisionPerRun[runId] = updatedRun.result.revision_id;
    }
    await loadRuns();
  } catch (err) {
    alert(`連線失敗：${err.message}`);
    if (submitBtn) submitBtn.disabled = false;
  }
};

window.switchRevision = function(runId, revId) {
  selectedRevisionPerRun[runId] = revId;
  loadRuns();
};

async function loadRuns() {
  const activeEl = document.activeElement;
  const activeInputId = activeEl ? activeEl.id : null;
  const selStart = activeEl && typeof activeEl.selectionStart === "number" ? activeEl.selectionStart : null;
  const selEnd = activeEl && typeof activeEl.selectionEnd === "number" ? activeEl.selectionEnd : null;

  const playingStates = {};
  document.querySelectorAll("audio").forEach(a => {
    const rId = a.id.replace("audio-player-", "");
    if (rId) {
      playingStates[rId] = { currentTime: a.currentTime, paused: a.paused };
    }
  });

  try {
    const response = await fetch("/api/runs");
    if (!response.ok) throw new Error("載入工作失敗");
    const runs = await response.json();
    renderRuns(runs, activeInputId, selStart, selEnd, playingStates);
  } catch (err) {
    runList.innerHTML = `<p class="status error">載入清單時發生錯誤：${escapeHtml(err.message)}</p>`;
  }
}

function renderRuns(runs, activeInputId, selStart, selEnd, playingStates) {
  if (!runs.length) {
    runList.innerHTML = "<p>目前尚未有轉錄工作紀錄。</p>";
    return;
  }

  runList.innerHTML = runs.map(run => {
    const revisions = (run.result && run.result.revisions) || [];
    let currentRev = null;
    if (revisions.length > 0) {
      const savedRevId = selectedRevisionPerRun[run.run_id];
      currentRev = revisions.find(r => r.revision_id === savedRevId) || revisions[revisions.length - 1];
      selectedRevisionPerRun[run.run_id] = currentRev.revision_id;
    }

    const revisionOptionsHtml = revisions.map(r => {
      const isSelected = currentRev && r.revision_id === currentRev.revision_id;
      const label = r.revision_kind === "raw_asr"
        ? `版本 #${r.revision_number} (原始 ASR - ${r.revision_id})`
        : `版本 #${r.revision_number} (${r.description || r.revision_kind} - ${r.revision_id})`;
      return `<option value="${escapeHtml(r.revision_id)}" ${isSelected ? "selected" : ""}>${escapeHtml(label)}</option>`;
    }).join("");

    const isPanelOpen = openCorrectionPanels.has(run.run_id);
    const draftText = correctionDrafts[run.run_id] != null
      ? correctionDrafts[run.run_id]
      : (currentRev ? currentRev.segments.map(s => s.text).join("\n") : "");

    const isRenameOpen = openRenamePanels.has(run.run_id);

    const filterState = filterStatePerRun[run.run_id] || {};
    const allSegments = currentRev ? currentRev.segments : [];
    const uniqueSpeakers = Array.from(new Set(allSegments.map(s => s.speaker).filter(Boolean)));

    return `
      <article class="run-card" id="run-card-${escapeHtml(run.run_id)}">
        <div class="run-header">
          <div>
            <h3>${escapeHtml(run.original_filename)}</h3>
            <p class="hint">
              工作 ID：<code>${escapeHtml(run.run_id)}</code> ·
              引擎：${escapeHtml(engineLabels[run.engine] || run.engine)} ·
              建立於：${new Date(run.created_at).toLocaleString("zh-TW", { hour12: false })}
            </p>
          </div>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span class="status-pill ${escapeHtml(run.status)}">
              ${escapeHtml(statusLabels[run.status] || run.status)}
            </span>
            <button type="button" class="small danger-outline" title="刪除這筆工作紀錄與音訊" data-action="delete-run" data-run-id="${escapeHtml(run.run_id)}" data-filename="${escapeHtml(run.original_filename)}">刪除</button>
          </div>
        </div>

        ${run.stored_filename ? `
          <div class="audio-player-wrapper">
            <span class="player-label">音訊回聽：</span>
            <audio id="audio-player-${escapeHtml(run.run_id)}" class="audio-player" data-run-id="${escapeHtml(run.run_id)}" controls preload="metadata" src="/api/runs/${escapeHtml(run.run_id)}/audio"></audio>
          </div>
        ` : ""}

        ${run.error ? `<p class="status error">${escapeHtml(run.error)}</p>` : ""}

        ${currentRev ? `
          <div class="revision-toolbar">
            <div class="revision-selector-group">
              <label><strong>逐字稿版本：</strong></label>
              <select class="revision-select" data-run-id="${escapeHtml(run.run_id)}">
                ${revisionOptionsHtml}
              </select>
              ${currentRev.revision_kind === "raw_asr" ? `<span class="raw-badge" title="受保護版本，不可覆蓋">原始稿 (raw_asr)</span>` : ""}
            </div>
            <div class="revision-actions">
              <button type="button" class="small secondary" data-action="toggle-rename-panel" data-run-id="${escapeHtml(run.run_id)}">更名講者</button>
              <button type="button" class="small secondary" data-action="toggle-correction-panel" data-run-id="${escapeHtml(run.run_id)}">文字校訂</button>
              <div class="export-group">
                <select id="export-format-${escapeHtml(run.run_id)}">
                  <option value="txt">純文字 (.txt)</option>
                  <option value="srt">字幕檔 (.srt)</option>
                  <option value="vtt">WebVTT (.vtt)</option>
                  <option value="md">會議報告 (.md)</option>
                  <option value="json">結構資料 (.json)</option>
                </select>
                <button type="button" class="small" data-action="trigger-export" data-run-id="${escapeHtml(run.run_id)}">匯出下載</button>
              </div>
            </div>
          </div>

          <div class="rename-modal" id="rename-panel-${escapeHtml(run.run_id)}" style="display: ${isRenameOpen ? 'grid' : 'none'};">
            <label><strong>更名講者（另存為新版本，保護原始 ASR 逐字稿）：</strong></label>
            <div class="rename-grid">
              <div>
                <label class="hint">欲替換的原始講者：</label>
                <input type="text" id="rename-old-${escapeHtml(run.run_id)}" value="${escapeHtml(renameSpeakerDefaults[run.run_id] || '')}" placeholder="例如 SPEAKER_00" />
              </div>
              <div>
                <label class="hint">新講者名稱：</label>
                <input type="text" id="rename-new-${escapeHtml(run.run_id)}" placeholder="例如 主席 或 王經理" />
              </div>
              <div>
                <button type="button" class="small" id="submit-rename-${escapeHtml(run.run_id)}" data-action="submit-rename-speaker" data-run-id="${escapeHtml(run.run_id)}">確認更名</button>
                <button type="button" class="small secondary" data-action="toggle-rename-panel" data-run-id="${escapeHtml(run.run_id)}">取消</button>
              </div>
            </div>
          </div>

          <div class="correct-modal" id="correction-panel-${escapeHtml(run.run_id)}" style="display: ${isPanelOpen ? 'grid' : 'none'};">
            <label><strong>編輯所選版本並另存人工校訂稿（保留原始 raw_asr）：</strong></label>
            <textarea class="correction-textarea" data-run-id="${escapeHtml(run.run_id)}" id="correction-text-${escapeHtml(run.run_id)}" rows="4">${escapeHtml(draftText)}</textarea>
            <div>
              <button type="button" class="small" id="submit-correct-${escapeHtml(run.run_id)}" data-action="submit-correction" data-run-id="${escapeHtml(run.run_id)}">儲存為新版本</button>
              <button type="button" class="small secondary" data-action="toggle-correction-panel" data-run-id="${escapeHtml(run.run_id)}">取消</button>
            </div>
          </div>

          <div class="filter-toolbar">
            <div class="filter-group">
              <input type="search" class="filter-input search-input" data-run-id="${escapeHtml(run.run_id)}" id="search-input-${escapeHtml(run.run_id)}" placeholder="搜尋段落文字..." value="${escapeHtml(filterState.keyword || '')}" />
              <select class="filter-select speaker-select" data-run-id="${escapeHtml(run.run_id)}" id="speaker-filter-${escapeHtml(run.run_id)}">
                <option value="">全部講者</option>
                ${uniqueSpeakers.map(spk => `<option value="${escapeHtml(spk)}" ${spk === filterState.speaker ? "selected" : ""}>${escapeHtml(spk)}</option>`).join("")}
              </select>
            </div>
            <label class="checkbox-label" style="font-size: 0.85rem;">
              <input type="checkbox" class="flag-checkbox" data-run-id="${escapeHtml(run.run_id)}" id="flag-filter-${escapeHtml(run.run_id)}" ${filterState.onlyFlagged ? "checked" : ""} />
              僅顯示警示段落
            </label>
          </div>

          <div class="transcript-box">
            ${renderSegments(currentRev.segments, run.run_id)}
          </div>
        ` : (run.status === "queued" || run.status === "transcribing" ? `<p class="hint">正在排隊／轉錄音訊中，請稍候…</p>` : "")}

        ${run.status === "completed" || run.status === "summarizing" || run.summary ? renderSummary(run.summary, run.run_id) : ""}
      </article>
    `;
  }).join("");

  runs.forEach(run => {
    if (filterStatePerRun[run.run_id]) {
      onFilterInput(run.run_id);
    }
  });

  for (const [rId, state] of Object.entries(playingStates)) {
    const player = document.getElementById(`audio-player-${rId}`);
    if (player) {
      player.currentTime = state.currentTime;
      if (!state.paused) {
        player.play().catch(() => {});
      }
    }
  }

  if (activeInputId) {
    const activeEl = document.getElementById(activeInputId);
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

// 集中式事件委派，完全取代 inline 事件
runList.addEventListener("click", (event) => {
  // 1. 處理證據連結點擊
  const evidenceLink = event.target.closest(".evidence-link");
  if (evidenceLink && runList.contains(evidenceLink)) {
    highlightSegment(evidenceLink.dataset.runId, evidenceLink.dataset.evidenceId);
    return;
  }

  // 2. 處理各種操作按鈕
  const actionEl = event.target.closest("[data-action]");
  if (!actionEl || !runList.contains(actionEl)) return;

  const action = actionEl.dataset.action;
  const runId = actionEl.dataset.runId;

  switch (action) {
    case "play-segment": {
      const startMs = Number(actionEl.dataset.startMs) || 0;
      const segId = actionEl.dataset.segId || "";
      playAudioAt(runId, startMs, segId);
      break;
    }
    case "open-rename-speaker": {
      const speaker = actionEl.dataset.speaker || "";
      openRenameSpeakerModal(runId, speaker);
      break;
    }
    case "toggle-segment-edit": {
      const segId = actionEl.dataset.segId || "";
      toggleSegmentEdit(runId, segId);
      break;
    }
    case "submit-segment-edit": {
      const segId = actionEl.dataset.segId || "";
      submitSegmentEdit(runId, segId);
      break;
    }
    case "copy-to-draft": {
      const row = actionEl.closest(".segment-row");
      const textNode = row ? row.querySelector(".segment-text") : null;
      const text = textNode ? textNode.textContent.trim() : "";
      copySegmentToDraft(runId, text);
      break;
    }
    case "trigger-summary": {
      triggerSummary(runId);
      break;
    }
    case "delete-run": {
      const filename = actionEl.dataset.filename || "";
      deleteRun(runId, filename);
      break;
    }
    case "toggle-rename-panel": {
      toggleRenamePanel(runId);
      break;
    }
    case "toggle-correction-panel": {
      toggleCorrectionPanel(runId);
      break;
    }
    case "trigger-export": {
      triggerExport(runId);
      break;
    }
    case "submit-rename-speaker": {
      submitRenameSpeaker(runId);
      break;
    }
    case "submit-correction": {
      submitCorrection(runId);
      break;
    }
  }
});

// 支援講者標籤鍵盤 Enter/Space 觸發更名
runList.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    const actionEl = event.target.closest("[data-action='open-rename-speaker']");
    if (actionEl && runList.contains(actionEl)) {
      event.preventDefault();
      openRenameSpeakerModal(actionEl.dataset.runId, actionEl.dataset.speaker || "");
    }
  }
});

// 集中式 change 事件處理（版本切換與下拉篩選）
runList.addEventListener("change", (event) => {
  const target = event.target;
  if (target.classList.contains("revision-select")) {
    switchRevision(target.dataset.runId, target.value);
  } else if (target.classList.contains("speaker-select") || target.classList.contains("flag-checkbox")) {
    onFilterInput(target.dataset.runId);
  }
});

// 集中式 input 事件處理（草稿文字與關鍵字搜尋）
runList.addEventListener("input", (event) => {
  const target = event.target;
  if (target.classList.contains("correction-textarea")) {
    onCorrectionInput(target.dataset.runId, target.value);
  } else if (target.classList.contains("search-input")) {
    onFilterInput(target.dataset.runId);
  }
});

// 捕捉階段監聽音訊播放進度，更新高亮
runList.addEventListener("timeupdate", (event) => {
  const audioEl = event.target;
  if (audioEl && audioEl.classList && audioEl.classList.contains("audio-player")) {
    onAudioTimeUpdate(audioEl.dataset.runId, audioEl.currentTime);
  }
}, true);

loadHealth();
loadRuns();
setInterval(() => {
  loadHealth();
  loadRuns();
}, 4000);
