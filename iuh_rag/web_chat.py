from __future__ import annotations

import argparse
import json
import os
import socket
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .config import (
    CHUNKS_PATH,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_HISTORY_K,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QDRANT_COLLECTION_PREFIX,
    DEFAULT_QDRANT_MODE,
    DEFAULT_QDRANT_PATH,
    DEFAULT_QDRANT_URL,
    DEFAULT_TOP_K,
    DEFAULT_VECTOR_STORE,
    RESULTS_DIR,
)
from .device import torch_device_status
from .pipeline import AgenticGraphRAG


DEFAULT_MODELS = "miniLM,e5,bge_m3,vietnamese_bi"
DEFAULT_SYSTEM = "S8_FULL"


@dataclass
class WebChatConfig:
    host: str = "127.0.0.1"
    port: int = 7860
    chunks: Path = CHUNKS_PATH
    history_file: Path = RESULTS_DIR / "web_chat_history.json"
    system: str = DEFAULT_SYSTEM
    models: str = DEFAULT_MODELS
    role: str = "general"
    top_k: int = DEFAULT_TOP_K
    history_k: int = DEFAULT_HISTORY_K
    device: str = DEFAULT_EMBEDDING_DEVICE
    vector_store: str = DEFAULT_VECTOR_STORE
    qdrant_mode: str = DEFAULT_QDRANT_MODE
    qdrant_path: str = DEFAULT_QDRANT_PATH
    qdrant_url: str = DEFAULT_QDRANT_URL
    qdrant_collection_prefix: str = DEFAULT_QDRANT_COLLECTION_PREFIX
    openai: bool = False
    openai_model: str = DEFAULT_OPENAI_MODEL
    reranker: str | None = None

    @property
    def model_keys(self) -> List[str]:
        return [item.strip() for item in self.models.split(",") if item.strip()]

    @property
    def dashboard_url(self) -> str | None:
        if self.qdrant_mode != "http":
            return None
        return self.qdrant_url.rstrip("/") + "/dashboard"


class WebChatService:
    def __init__(self, config: WebChatConfig) -> None:
        self.config = config
        self._rag: AgenticGraphRAG | None = None
        self._rag_lock = threading.RLock()
        self._history_lock = threading.Lock()
        self.histories = self._load_histories()

    def _load_histories(self) -> Dict[str, List[Dict[str, Any]]]:
        path = self.config.history_file
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if isinstance(payload, list):
            return {"default": payload}
        if isinstance(payload, dict):
            return {str(key): value for key, value in payload.items() if isinstance(value, list)}
        return {}

    def _save_histories(self) -> None:
        self.config.history_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.history_file.write_text(
            json.dumps(self.histories, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def history(self, session_id: str) -> List[Dict[str, Any]]:
        with self._history_lock:
            return list(self.histories.get(session_id, []))

    def history_context(self, session_id: str) -> List[Dict[str, Any]]:
        history = self.history(session_id)
        if self.config.history_k <= 0:
            return []
        return history[-self.config.history_k * 2 :]

    def clear_history(self, session_id: str) -> None:
        with self._history_lock:
            self.histories[session_id] = []
            self._save_histories()

    @staticmethod
    def _assistant_history_item(answer: str, result: Dict[str, Any]) -> Dict[str, Any]:
        item: Dict[str, Any] = {"role": "assistant", "content": answer}
        constraints = result.get("evidence_constraints") or {}
        if isinstance(constraints, dict):
            if constraints.get("major"):
                item["active_major"] = constraints.get("major")
                item["active_entity"] = constraints.get("major")
            if constraints.get("topic"):
                item["topic"] = constraints.get("topic")
            if constraints.get("year"):
                item["year"] = constraints.get("year")
        for key in ("rewritten_question", "retrieval_query", "intent", "topic", "year"):
            if result.get(key) is not None:
                item[key] = result.get(key)
        return item

    def _append_turn(self, session_id: str, question: str, answer: str, result: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        with self._history_lock:
            history = self.histories.setdefault(session_id, [])
            history.append({"role": "user", "content": question})
            history.append(self._assistant_history_item(answer, result or {}))
            self._save_histories()
            return list(history)

    def rag(self) -> AgenticGraphRAG:
        if self._rag is not None:
            return self._rag
        with self._rag_lock:
            if self._rag is None:
                self._rag = AgenticGraphRAG.from_path(
                    self.config.chunks,
                    enable_dense=True,
                    dense_model_keys=self.config.model_keys,
                    answer_mode="openai" if self.config.openai else "extractive",
                    openai_model=self.config.openai_model,
                    reranker_model=self.config.reranker,
                    embedding_device=self.config.device,
                    vector_store=self.config.vector_store,
                    qdrant_mode=self.config.qdrant_mode,
                    qdrant_path=self.config.qdrant_path,
                    qdrant_url=self.config.qdrant_url,
                    qdrant_collection_prefix=self.config.qdrant_collection_prefix,
                )
            return self._rag

    def qdrant_status(self) -> Dict[str, Any]:
        if self.config.vector_store != "qdrant":
            return {"enabled": False, "mode": None, "path": None, "url": None, "dashboard": None, "ok": None, "error": None}
        if self.config.qdrant_mode == "local":
            path = Path(self.config.qdrant_path)
            return {
                "enabled": True,
                "mode": "local",
                "path": str(path),
                "path_exists": path.exists(),
                "url": None,
                "dashboard": None,
                "ok": True,
                "error": None,
            }
        try:
            with urllib.request.urlopen(self.config.qdrant_url.rstrip("/") + "/collections", timeout=2) as response:
                body = response.read().decode("utf-8")
            return {
                "enabled": True,
                "mode": "http",
                "path": None,
                "ok": True,
                "url": self.config.qdrant_url,
                "dashboard": self.config.dashboard_url,
                "error": None,
                "collections_response": body[:500],
            }
        except Exception as exc:
            return {
                "enabled": True,
                "mode": "http",
                "path": None,
                "ok": False,
                "url": self.config.qdrant_url,
                "dashboard": self.config.dashboard_url,
                "error": str(exc),
            }

    def status(self, include_dense: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "system": self.config.system,
            "models": self.config.model_keys,
            "role": self.config.role,
            "top_k": self.config.top_k,
            "history_k": self.config.history_k,
            "answer_mode": "openai" if self.config.openai else "extractive",
            "openai_model": self.config.openai_model if self.config.openai else None,
            "openai_key_set": bool(os.getenv("OPENAI_API_KEY")),
            "device": torch_device_status(self.config.device),
            "vector_store": self.config.vector_store,
            "qdrant_mode": self.config.qdrant_mode if self.config.vector_store == "qdrant" else None,
            "qdrant_path": self.config.qdrant_path if self.config.vector_store == "qdrant" and self.config.qdrant_mode == "local" else None,
            "qdrant_url": self.config.qdrant_url if self.config.vector_store == "qdrant" and self.config.qdrant_mode == "http" else None,
            "qdrant": self.qdrant_status(),
            "history_file": str(self.config.history_file),
            "rag_loaded": self._rag is not None,
        }
        if include_dense:
            try:
                payload["dense_statuses"] = self.rag().dense_statuses()
            except Exception as exc:
                payload["dense_status_error"] = str(exc)
        return payload

    def ask(self, message: str, session_id: str, role: str | None = None, top_k: int | None = None) -> Dict[str, Any]:
        message = message.strip()
        if not message:
            raise ValueError("Message is empty.")
        selected_role = role or self.config.role
        selected_top_k = top_k or self.config.top_k
        history_context = self.history_context(session_id)
        payload = {"question": message, "history": history_context, "role": selected_role}
        with self._rag_lock:
            result = self.rag().answer_for_system(self.config.system, payload, top_k=selected_top_k)
        result["system"] = self.config.system
        result["history"] = self._append_turn(session_id, message, result.get("answer", ""), result)
        result["history_k"] = self.config.history_k
        result["history_context_messages"] = len(history_context)
        result["session_id"] = session_id
        result["status"] = self.status(include_dense=False)
        return result


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _html() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>IUH Agentic Graph-RAG Chat</title>
  <style>
    :root {
      --bg: #f6f7f9;
      --panel: #ffffff;
      --ink: #17202a;
      --muted: #647084;
      --line: #d9dee7;
      --accent: #1264a3;
      --accent-dark: #0b4f82;
      --danger: #a3322b;
      --ok: #16794c;
      --warn: #9a6700;
      --chip: #eef2f7;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app {
      min-height: 100vh;
      display: grid;
      grid-template-columns: minmax(240px, 310px) minmax(0, 1fr);
    }
    aside {
      border-right: 1px solid var(--line);
      background: #fbfcfd;
      padding: 18px;
      overflow: auto;
    }
    main {
      display: grid;
      grid-template-rows: auto 1fr auto;
      min-width: 0;
      min-height: 100vh;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 14px 18px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
    }
    h1, h2 {
      margin: 0;
      letter-spacing: 0;
    }
    h1 { font-size: 18px; }
    h2 { font-size: 13px; margin: 20px 0 8px; color: var(--muted); text-transform: uppercase; }
    .sub { color: var(--muted); font-size: 13px; }
    .status-line { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; }
    .chip {
      display: inline-flex;
      align-items: center;
      min-height: 26px;
      padding: 3px 8px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: var(--chip);
      color: var(--ink);
      font-size: 12px;
      white-space: nowrap;
    }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .bad { color: var(--danger); }
    label { display: block; margin: 10px 0 5px; color: var(--muted); font-size: 12px; }
    select, input, textarea, button {
      font: inherit;
      border-radius: 6px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--ink);
    }
    select, input {
      width: 100%;
      height: 36px;
      padding: 0 10px;
    }
    button {
      height: 38px;
      padding: 0 12px;
      cursor: pointer;
      border-color: var(--accent);
      color: #fff;
      background: var(--accent);
      font-weight: 600;
    }
    button:hover { background: var(--accent-dark); }
    button.secondary {
      color: var(--ink);
      border-color: var(--line);
      background: #fff;
      font-weight: 500;
    }
    button.secondary:hover { background: #f2f5f9; }
    button:disabled {
      cursor: wait;
      opacity: 0.65;
    }
    .side-text {
      color: var(--muted);
      font-size: 12px;
      overflow-wrap: anywhere;
      margin: 6px 0;
    }
    .messages {
      overflow: auto;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .message {
      max-width: 920px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px 14px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .message.user {
      align-self: flex-end;
      background: #edf6ff;
      border-color: #c7dff5;
      max-width: 760px;
    }
    .message.assistant { align-self: flex-start; }
    .message.meta {
      align-self: center;
      max-width: 780px;
      color: var(--muted);
      background: transparent;
      border-style: dashed;
      white-space: normal;
    }
    .message-title {
      font-size: 12px;
      color: var(--muted);
      margin-bottom: 6px;
      white-space: normal;
    }
    details {
      margin-top: 10px;
      border-top: 1px solid var(--line);
      padding-top: 8px;
      white-space: normal;
    }
    summary { cursor: pointer; color: var(--accent); }
    .source {
      margin-top: 8px;
      padding: 8px;
      background: #f8fafc;
      border: 1px solid var(--line);
      border-radius: 6px;
      font-size: 12px;
      overflow-wrap: anywhere;
    }
    .composer {
      border-top: 1px solid var(--line);
      background: var(--panel);
      padding: 12px 18px;
    }
    .composer-row {
      display: grid;
      grid-template-columns: 1fr auto auto;
      gap: 10px;
      align-items: end;
    }
    textarea {
      min-height: 54px;
      max-height: 180px;
      resize: vertical;
      padding: 10px;
      width: 100%;
    }
    .controls { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }
    @media (max-width: 860px) {
      .app { grid-template-columns: 1fr; }
      aside { border-right: 0; border-bottom: 1px solid var(--line); }
      main { min-height: 70vh; }
      .composer-row { grid-template-columns: 1fr; }
      button { width: 100%; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>IUH RAG Chat</h1>
      <div class="sub">S8_FULL localhost chatbot</div>

      <h2>Runtime</h2>
      <div id="runtime"></div>

      <h2>Chat Settings</h2>
      <label for="role">Role</label>
      <select id="role">
        <option value="general">general</option>
        <option value="thi_sinh">thi_sinh</option>
        <option value="phu_huynh">phu_huynh</option>
        <option value="sinh_vien">sinh_vien</option>
        <option value="can_bo_tu_van">can_bo_tu_van</option>
      </select>

      <label for="session">Session</label>
      <input id="session" value="default" />

      <label for="topk">Top K</label>
      <input id="topk" type="number" min="1" max="20" value="5" />

      <div class="controls" style="margin-top: 12px;">
        <button class="secondary" id="reload">Reload</button>
        <button class="secondary" id="clear">Clear</button>
      </div>

      <h2>Links</h2>
      <div class="side-text"><a id="qdrantLink" href="#" target="_blank" rel="noreferrer">Qdrant dashboard</a></div>
      <div class="side-text" id="historyFile"></div>
    </aside>

    <main>
      <header>
        <div>
          <h1>Real-time Chatbot</h1>
          <div class="sub">Persistent history, Qdrant retrieval, OpenAI or extractive answers</div>
        </div>
        <div class="status-line" id="headerStatus"></div>
      </header>

      <section class="messages" id="messages"></section>

      <section class="composer">
        <div class="composer-row">
          <textarea id="message" placeholder="Ask about IUH admissions, programs, regulations, scholarships..."></textarea>
          <button id="send">Send</button>
          <button class="secondary" id="stop" disabled>Busy</button>
        </div>
      </section>
    </main>
  </div>

  <script>
    const messages = document.getElementById("messages");
    const messageBox = document.getElementById("message");
    const sendButton = document.getElementById("send");
    const stopButton = document.getElementById("stop");
    const roleSelect = document.getElementById("role");
    const sessionInput = document.getElementById("session");
    const topkInput = document.getElementById("topk");
    const runtime = document.getElementById("runtime");
    const headerStatus = document.getElementById("headerStatus");

    function sessionId() {
      return sessionInput.value.trim() || "default";
    }

    function addMessage(role, content, meta) {
      const div = document.createElement("div");
      div.className = "message " + role;
      const title = document.createElement("div");
      title.className = "message-title";
      title.textContent = role === "user" ? "You" : role === "assistant" ? "Assistant" : "Status";
      if (role === "assistant" && meta) {
        const parts = [];
        if (meta.answer_mode) parts.push(meta.answer_mode);
        if (meta.llm_model) parts.push(meta.llm_model);
        if (meta.llm_used === true) parts.push("llm used");
        if (meta.llm_used === false && meta.llm_skip_reason) parts.push("skip: " + meta.llm_skip_reason);
        if (meta.llm_error) parts.push("error: " + meta.llm_error);
        if (parts.length) title.textContent = "Assistant · " + parts.join(" · ");
      }
      div.appendChild(title);
      const body = document.createElement("div");
      body.textContent = content;
      div.appendChild(body);

      if (meta && meta.evidence && meta.evidence.length) {
        const details = document.createElement("details");
        const summary = document.createElement("summary");
        summary.textContent = "Sources (" + meta.evidence.length + ")";
        details.appendChild(summary);
        meta.evidence.forEach((item) => {
          const source = document.createElement("div");
          source.className = "source";
          source.textContent = item.chunk_id + " | " + (item.file_name || "") + " | " + (item.relative_path || "");
          details.appendChild(source);
        });
        div.appendChild(details);
      }

      messages.appendChild(div);
      messages.scrollTop = messages.scrollHeight;
      return div;
    }

    function setBusy(isBusy) {
      sendButton.disabled = isBusy;
      stopButton.disabled = !isBusy;
      stopButton.textContent = isBusy ? "Running" : "Busy";
    }

    function chip(text, cls) {
      const span = document.createElement("span");
      span.className = "chip " + (cls || "");
      span.textContent = text;
      return span;
    }

    async function loadStatus() {
      const res = await fetch("/api/status");
      const data = await res.json();
      headerStatus.innerHTML = "";
      headerStatus.appendChild(chip(data.system || "S8_FULL"));
      headerStatus.appendChild(chip(data.answer_mode || "extractive", data.answer_mode === "openai" ? "ok" : ""));
      headerStatus.appendChild(chip("device " + ((data.device || {}).resolved_device || "unknown")));
      headerStatus.appendChild(chip("Qdrant " + (data.qdrant && data.qdrant.ok ? "ok" : "off"), data.qdrant && data.qdrant.ok ? "ok" : "warn"));

      runtime.innerHTML = "";
      const lines = [
        "System: " + data.system,
        "Models: " + (data.models || []).join(", "),
        "Vector store: " + data.vector_store,
        "Qdrant mode: " + (data.qdrant_mode || "n/a"),
        "Qdrant path: " + (data.qdrant_path || "n/a"),
        "Device: " + ((data.device || {}).resolved_device || "unknown"),
        "History k: " + data.history_k + " turns",
        "OpenAI key: " + (data.openai_key_set ? "set" : "not set"),
        "Qdrant: " + (data.qdrant && data.qdrant.ok ? "ready" : "not ready")
      ];
      lines.forEach((line) => {
        const div = document.createElement("div");
        div.className = "side-text";
        div.textContent = line;
        runtime.appendChild(div);
      });
      const qLink = document.getElementById("qdrantLink");
      qLink.href = data.qdrant && data.qdrant.dashboard ? data.qdrant.dashboard : "#";
      qLink.textContent = data.qdrant && data.qdrant.dashboard ? "Qdrant dashboard" : "Embedded Qdrant: no dashboard";
      document.getElementById("historyFile").textContent = "History: " + data.history_file;
    }

    async function loadHistory() {
      const url = "/api/history?session_id=" + encodeURIComponent(sessionId());
      const res = await fetch(url);
      const data = await res.json();
      messages.innerHTML = "";
      if (!data.history || !data.history.length) {
        addMessage("meta", "No history yet. Ask a question to start this session.");
        return;
      }
      data.history.forEach((item) => {
        addMessage(item.role === "assistant" ? "assistant" : "user", item.content || "");
      });
    }

    async function sendMessage() {
      const message = messageBox.value.trim();
      if (!message) return;
      messageBox.value = "";
      addMessage("user", message);
      const placeholder = addMessage("meta", "Retrieving evidence and generating answer...");
      setBusy(true);
      try {
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify({
            message,
            session_id: sessionId(),
            role: roleSelect.value,
            top_k: Number(topkInput.value || 5)
          })
        });
        const data = await res.json();
        placeholder.remove();
        if (!res.ok) {
          addMessage("assistant", data.error || "Request failed.");
        } else {
          addMessage("assistant", data.answer || "", data);
          await loadStatus();
        }
      } catch (err) {
        placeholder.remove();
        addMessage("assistant", "Request failed: " + err);
      } finally {
        setBusy(false);
      }
    }

    async function clearHistory() {
      await fetch("/api/clear", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({session_id: sessionId()})
      });
      await loadHistory();
    }

    sendButton.addEventListener("click", sendMessage);
    document.getElementById("reload").addEventListener("click", async () => {
      await loadStatus();
      await loadHistory();
    });
    document.getElementById("clear").addEventListener("click", clearHistory);
    sessionInput.addEventListener("change", loadHistory);
    messageBox.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.metaKey || event.ctrlKey)) {
        sendMessage();
      }
    });

    loadStatus().then(loadHistory);
  </script>
</body>
</html>
"""


class WebChatHandler(BaseHTTPRequestHandler):
    server_version = "IUHRAGChat/1.0"

    @property
    def service(self) -> WebChatService:
        return self.server.service  # type: ignore[attr-defined]

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Any) -> None:
        self._send(status, "application/json; charset=utf-8", _json_bytes(payload))

    def _read_json(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            return {}
        data = self.rfile.read(length)
        if not data:
            return {}
        return json.loads(data.decode("utf-8"))

    def do_GET(self) -> None:
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self._send(HTTPStatus.OK, "text/html; charset=utf-8", _html().encode("utf-8"))
            return
        if parsed.path == "/api/status":
            query = urllib.parse.parse_qs(parsed.query)
            include_dense = query.get("dense", ["0"])[0] in {"1", "true", "yes"}
            self._send_json(HTTPStatus.OK, self.service.status(include_dense=include_dense))
            return
        if parsed.path == "/api/history":
            query = urllib.parse.parse_qs(parsed.query)
            session_id = query.get("session_id", ["default"])[0] or "default"
            self._send_json(HTTPStatus.OK, {"session_id": session_id, "history": self.service.history(session_id)})
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def do_POST(self) -> None:
        try:
            payload = self._read_json()
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid JSON: {exc}"})
            return

        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/chat":
            try:
                result = self.service.ask(
                    message=str(payload.get("message", "")),
                    session_id=str(payload.get("session_id") or "default"),
                    role=payload.get("role") or None,
                    top_k=int(payload.get("top_k") or self.service.config.top_k),
                )
            except Exception as exc:
                self._send_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": str(exc)})
                return
            self._send_json(HTTPStatus.OK, result)
            return

        if parsed.path == "/api/clear":
            session_id = str(payload.get("session_id") or "default")
            self.service.clear_history(session_id)
            self._send_json(HTTPStatus.OK, {"session_id": session_id, "history": []})
            return

        self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

    def log_message(self, format: str, *args: Any) -> None:
        print("%s - %s" % (self.address_string(), format % args))


class WebChatHTTPServer(ThreadingHTTPServer):
    service: WebChatService


def _port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def find_available_port(host: str, start_port: int) -> int:
    port = start_port
    while port < start_port + 100:
        if _port_available(host, port):
            return port
        port += 1
    raise RuntimeError(f"No available port found from {start_port} to {start_port + 99}.")


def serve_web_chat(config: WebChatConfig) -> Tuple[WebChatHTTPServer, str]:
    port = find_available_port(config.host, config.port)
    config.port = port
    service = WebChatService(config)
    server = WebChatHTTPServer((config.host, config.port), WebChatHandler)
    server.service = service
    url = f"http://{config.host}:{config.port}"
    print(f"IUH RAG web chatbot running at {url}")
    print(f"System: {config.system} | models: {config.models} | vector_store: {config.vector_store}")
    print(f"History file: {config.history_file}")
    if config.vector_store == "qdrant":
        if config.qdrant_mode == "local":
            print(f"Qdrant embedded local path: {config.qdrant_path}")
        else:
            print(f"Qdrant HTTP URL: {config.qdrant_url}")
            print(f"Qdrant dashboard: {config.dashboard_url}")
    if config.openai:
        print(f"OpenAI mode: {config.openai_model} | OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
    server.serve_forever()
    return server, url


def config_from_args(args: argparse.Namespace) -> WebChatConfig:
    return WebChatConfig(
        host=args.host,
        port=args.port,
        chunks=Path(args.chunks),
        history_file=Path(args.history_file),
        system=args.system,
        models=args.models,
        role=args.role,
        top_k=args.top_k,
        history_k=args.history_k,
        device=args.device,
        vector_store=args.vector_store,
        qdrant_mode=args.qdrant_mode,
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        qdrant_collection_prefix=args.qdrant_collection_prefix,
        openai=args.openai,
        openai_model=args.openai_model,
        reranker=args.reranker,
    )
