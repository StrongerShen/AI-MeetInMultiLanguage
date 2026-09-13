from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import httpx


def check_host_memory() -> dict[str, Any]:
    """檢查主機實體記憶體狀態。"""
    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        available_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
        total_gib = round(total_bytes / (1024**3), 1)
        avail_gib = round(available_bytes / (1024**3), 1)

        is_ok = (total_gib >= 16) and (avail_gib >= 2.0)
        warn_reasons = []
        if total_gib < 16:
            warn_reasons.append(f"總記憶體僅 {total_gib} GiB（建議 16 GiB 以上）")
        if avail_gib < 2.0:
            warn_reasons.append(f"可用記憶體僅剩 {avail_gib} GiB（建議 2.0 GiB 以上）")

        detail = f"總記憶體 {total_gib} GiB，可用 {avail_gib} GiB"
        if warn_reasons:
            detail += f"（注意：{'；'.join(warn_reasons)}，記憶體過低可能引發 OOM）"

        return {
            "status": "ok" if is_ok else "warning",
            "total_gib": total_gib,
            "available_gib": avail_gib,
            "detail": detail,
        }
    except Exception as err:
        return {"status": "warning", "detail": f"無法偵測主機記憶體：{err}"}



def check_nvidia_gpu() -> dict[str, Any]:
    """檢查 NVIDIA GPU 與顯存狀態。"""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {
            "status": "warning",
            "gpu_found": False,
            "detail": "未偵測到 nvidia-smi，若僅使用雲端 API 則不受影響；若使用 Breeze 需具備 NVIDIA GPU。",
        }
    try:
        proc = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,memory.total,memory.free,memory.used,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            return {"status": "warning", "gpu_found": False, "detail": proc.stderr.strip()}

        lines = proc.stdout.strip().splitlines()
        if not lines:
            return {"status": "warning", "gpu_found": False, "detail": "無 GPU 回傳資訊"}

        parts = [p.strip() for p in lines[0].split(",")]
        name = parts[0]
        total_mib = int(parts[1])
        free_mib = int(parts[2])
        used_mib = int(parts[3])
        driver = parts[4] if len(parts) > 4 else "未知"

        # 建議 8 GiB 顯存門檻（約 7500 MiB 以上）
        status = "ok" if total_mib >= 7000 else "warning"
        return {
            "status": status,
            "gpu_found": True,
            "name": name,
            "total_mib": total_mib,
            "free_mib": free_mib,
            "used_mib": used_mib,
            "driver_version": driver,
            "detail": f"{name}（總顯存 {total_mib} MiB，目前可用 {free_mib} MiB，驅動 {driver}）",
        }
    except Exception as err:
        return {"status": "warning", "gpu_found": False, "detail": str(err)}


def check_binary_tools() -> dict[str, Any]:
    """檢查 ffmpeg 與 ffprobe 工具。"""
    ffmpeg_path = shutil.which("ffmpeg")
    ffprobe_path = shutil.which("ffprobe")
    all_ok = bool(ffmpeg_path and ffprobe_path)
    return {
        "status": "ok" if all_ok else "error",
        "ffmpeg": bool(ffmpeg_path),
        "ffprobe": bool(ffprobe_path),
        "detail": (
            "ffmpeg 與 ffprobe 已安裝就緒"
            if all_ok
            else "缺少音訊處理核心工具：請確認已安裝 ffmpeg 與 ffprobe"
        ),
    }


def check_speaches_service(speaches_url: str = "http://127.0.0.1:8001/v1") -> dict[str, Any]:
    """檢查本機 Speaches 轉錄服務連線與模型狀態。"""
    api_root = speaches_url.removesuffix("/v1")
    try:
        with httpx.Client(timeout=3.0) as client:
            resp = client.get(f"{api_root}/health")
            if resp.status_code != 200:
                return {"status": "error", "available": False, "detail": f"HTTP {resp.status_code}"}
            models_resp = client.get(f"{speaches_url}/models")
            models: list[str] = []
            if models_resp.is_success:
                data = models_resp.json()
                models = [m.get("id", "") for m in data.get("data", [])]

            breeze_ready = any("Breeze" in m for m in models)
            return {
                "status": "ok",
                "available": True,
                "breeze_ready": breeze_ready,
                "models": models,
                "detail": (
                    f"Speaches 正常運作（已就緒模型數：{len(models)}）"
                    + ("，包含 Breeze ASR" if breeze_ready else "，尚未載入 Breeze ASR")
                ),
            }
    except Exception as err:
        return {
            "status": "warning",
            "available": False,
            "detail": f"無法連線至 Speaches ({api_root})：{err}",
        }


def check_ollama_service(
    ollama_url: str = "http://127.0.0.1:11434",
    target_model: str = "qwen3.5:9b",
) -> dict[str, Any]:
    """檢查本機 Ollama 摘要服務連線與模型清單。"""
    try:
        with httpx.Client(timeout=3.0) as client:
            ver_resp = client.get(f"{ollama_url}/api/version")
            if not ver_resp.is_success:
                return {"status": "error", "available": False, "detail": f"HTTP {ver_resp.status_code}"}
            version = ver_resp.json().get("version", "未知")

            tags_resp = client.get(f"{ollama_url}/api/tags")
            models: list[str] = []
            if tags_resp.is_success:
                data = tags_resp.json()
                models = [m.get("name", "") for m in data.get("models", [])]

            target_ready = any(target_model in m for m in models)
            return {
                "status": "ok" if target_ready else "warning",
                "available": True,
                "version": version,
                "models": models,
                "target_model": target_model,
                "target_ready": target_ready,
                "detail": (
                    f"Ollama {version} 正常運作；模型 {target_model} 已就緒"
                    if target_ready
                    else f"Ollama {version} 正常運作，但尚未拉取 {target_model} 模型"
                ),
            }
    except Exception as err:
        return {
            "status": "warning",
            "available": False,
            "detail": f"無法連線至 Ollama ({ollama_url})：{err}",
        }


def check_storage_directory(data_dir: Path) -> dict[str, Any]:
    """檢查資料儲存目錄的權限與磁碟可用空間。"""
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        test_file = data_dir / ".write_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink()

        stat = shutil.disk_usage(data_dir)
        free_gib = round(stat.free / (1024**3), 1)
        total_gib = round(stat.total / (1024**3), 1)
        status = "ok" if free_gib >= 5.0 else "warning"
        return {
            "status": status,
            "writable": True,
            "free_gib": free_gib,
            "total_gib": total_gib,
            "detail": f"目錄可讀寫，磁碟可用空間 {free_gib} GiB（總空間 {total_gib} GiB）",
        }
    except Exception as err:
        return {"status": "error", "writable": False, "detail": f"資料目錄不可寫入：{err}"}


def diagnose_system(
    data_dir: Path,
    speaches_url: str = "http://127.0.0.1:8001/v1",
    ollama_url: str = "http://127.0.0.1:11434",
    ollama_model: str = "qwen3.5:9b",
) -> dict[str, Any]:
    """執行全系統健康診斷並回傳結構化成果。"""
    memory = check_host_memory()
    gpu = check_nvidia_gpu()
    tools = check_binary_tools()
    speaches = check_speaches_service(speaches_url)
    ollama = check_ollama_service(ollama_url, target_model=ollama_model)
    storage = check_storage_directory(data_dir)

    all_statuses = [
        memory["status"],
        gpu["status"],
        tools["status"],
        speaches["status"],
        ollama["status"],
        storage["status"],
    ]
    if "error" in all_statuses:
        overall = "error"
    elif "warning" in all_statuses:
        overall = "warning"
    else:
        overall = "ok"

    return {
        "overall_status": overall,
        "items": {
            "host_memory": memory,
            "nvidia_gpu": gpu,
            "binary_tools": tools,
            "speaches_service": speaches,
            "ollama_service": ollama,
            "storage": storage,
        },
    }


def format_doctor_report(diag: dict[str, Any]) -> str:
    """將診斷字典排版為臺灣繁體中文終端文字報告。"""
    lines: list[str] = [
        "==================================================",
        "  AI Meet In Multi-Language 系統環境檢查報告",
        "==================================================",
    ]
    symbols = {"ok": "✅ [正常]", "warning": "⚠️  [注意]", "error": "❌ [錯誤]"}
    items = diag.get("items", {})

    labels = [
        ("host_memory", "主機記憶體"),
        ("nvidia_gpu", "NVIDIA GPU 顯存"),
        ("binary_tools", "音訊處理工具 (ffmpeg)"),
        ("speaches_service", "Speaches ASR 服務"),
        ("ollama_service", "Ollama 摘要服務"),
        ("storage", "資料儲存與磁碟空間"),
    ]

    for key, name in labels:
        data = items.get(key, {})
        status = data.get("status", "warning")
        sym = symbols.get(status, "[未知]")
        detail = data.get("detail", "")
        lines.append(f"{sym} {name}：{detail}")

    overall = diag.get("overall_status", "ok")
    lines.append("--------------------------------------------------")
    if overall == "ok":
        lines.append("系統狀態整體良好，所有本機與外部相依元件均已就緒！")
    elif overall == "warning":
        lines.append("系統具備基本運作能力，但部分項目存在警示（如未連線本機服務或顯存偏低），請參閱上方提示。")
    else:
        lines.append("系統存在關鍵錯誤（如缺少 ffmpeg 或儲存目錄不可寫入），請先修復後再啟動。")
    lines.append("==================================================")
    return "\n".join(lines)
