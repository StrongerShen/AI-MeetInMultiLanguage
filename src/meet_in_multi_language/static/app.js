const form = document.querySelector("#upload-form");
const statusNode = document.querySelector("#service-status");
const messageNode = document.querySelector("#form-message");
const runList = document.querySelector("#run-list");
const refreshButton = document.querySelector("#refresh-button");
const submitButton = form.querySelector('button[type="submit"]');

const statusLabels = {
  queued: "等待處理",
  transcribing: "轉錄中",
  completed: "已完成",
  failed: "失敗",
};

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

async function readError(response) {
  try {
    const body = await response.json();
    return body.detail || "發生未預期的錯誤";
  } catch {
    return "服務回應格式不正確";
  }
}

async function loadHealth() {
  const response = await fetch("/api/health");
  const health = await response.json();
  const limitMb = Math.floor(health.max_upload_bytes / 1024 / 1024);
  if (health.openai_configured) {
    statusNode.textContent = `服務已就緒；目前原型單檔上限 ${limitMb} MB。`;
    submitButton.disabled = false;
  } else {
    statusNode.textContent = "尚未設定 OPENAI_API_KEY，請先在伺服器環境加入金鑰。";
    statusNode.classList.add("error");
    submitButton.disabled = true;
  }
}

async function loadRuns() {
  const response = await fetch("/api/runs");
  if (!response.ok) {
    runList.innerHTML = '<p class="error">無法取得轉錄工作。</p>';
    return;
  }
  const runs = await response.json();
  if (!runs.length) {
    runList.innerHTML = "<p>尚無轉錄工作。</p>";
    return;
  }
  runList.innerHTML = runs.map((run) => `
    <article class="run-card">
      <h3>${escapeHtml(run.original_filename)}</h3>
      <p class="meta">${escapeHtml(statusLabels[run.status] || run.status)} · ${escapeHtml(run.engine)}</p>
      ${run.error ? `<p class="error">${escapeHtml(run.error)}</p>` : ""}
      ${run.result ? `<pre>${escapeHtml(run.result.text)}</pre>` : ""}
    </article>
  `).join("");
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  submitButton.disabled = true;
  messageNode.textContent = "正在上傳…";
  const response = await fetch("/api/runs", { method: "POST", body: new FormData(form) });
  if (!response.ok) {
    messageNode.textContent = await readError(response);
    messageNode.className = "error";
  } else {
    messageNode.textContent = "已建立轉錄工作。";
    messageNode.className = "";
    form.reset();
    await loadRuns();
  }
  submitButton.disabled = false;
});

refreshButton.addEventListener("click", loadRuns);
loadHealth().catch(() => {
  statusNode.textContent = "無法連線到服務。";
  statusNode.classList.add("error");
  submitButton.disabled = true;
});
loadRuns();
setInterval(loadRuns, 5000);

