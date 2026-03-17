import { useEffect, useState } from "react";

import { Composer } from "./components/Composer";
import { ContextPanel } from "./components/ContextPanel";
import { Conversation } from "./components/Conversation";
import { ShellHeader } from "./components/ShellHeader";
import { StatusPanel } from "./components/StatusPanel";
import { api } from "./lib/api";
import type {
  BuildIndexResponse,
  ChatResponse,
  LocalMessage,
  StatusResponse,
} from "./types";


const SAMPLE_QUESTIONS = [
  "Thong tin tuyen sinh dai hoc chinh quy moi nhat la gi?",
  "Khoa Cong nghe Thong tin dao tao nhung nganh nao?",
  "Dieu kien xet hoc bong duoc neu nhu the nao?",
  "Sinh vien khong duoc lam nhung hanh vi nao theo quy che?",
];


function createMessage(role: LocalMessage["role"], content: string): LocalMessage {
  return {
    id: `${role}-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    role,
    content,
  };
}


function parseSources(value: string): string[] {
  return value
    .split(/[\n,]+/)
    .map((part) => part.trim())
    .filter(Boolean);
}


export default function App() {
  const [buildMessage, setBuildMessage] = useState<string | null>(null);
  const [buildResult, setBuildResult] = useState<BuildIndexResponse | null>(null);
  const [building, setBuilding] = useState(false);
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [lastResponse, setLastResponse] = useState<ChatResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [messages, setMessages] = useState<LocalMessage[]>([]);
  const [refreshPending, setRefreshPending] = useState(false);
  const [sourceDraft, setSourceDraft] = useState("");
  const [status, setStatus] = useState<StatusResponse | null>(null);

  async function refreshStatus() {
    setRefreshPending(true);
    try {
      const nextStatus = await api.getStatus();
      setStatus(nextStatus);
      setError(null);
    } catch (requestError) {
      const nextMessage =
        requestError instanceof Error ? requestError.message : "Khong the tai trang thai he thong.";
      setError(nextMessage);
    } finally {
      setRefreshPending(false);
    }
  }

  useEffect(() => {
    void refreshStatus();
  }, []);

  async function handleBuildIndex() {
    setBuilding(true);
    setBuildMessage(null);
    try {
      const result = await api.buildIndex(parseSources(sourceDraft));
      setBuildResult(result);
      setBuildMessage(`Da build ${result.chunk_count} chunks vao index hien tai.`);
      setError(null);
      await refreshStatus();
    } catch (requestError) {
      const nextMessage =
        requestError instanceof Error ? requestError.message : "Build index that bai.";
      setError(nextMessage);
      setBuildMessage(nextMessage);
    } finally {
      setBuilding(false);
    }
  }

  async function handleSubmit() {
    const question = draft.trim();
    if (!question) {
      return;
    }

    const nextUserMessage = createMessage("user", question);
    const history = messages.map(({ role, content }) => ({ role, content }));

    setMessages((current) => current.concat(nextUserMessage));
    setDraft("");
    setLoading(true);
    setError(null);

    try {
      const response = await api.sendChat(question, history);
      setLastResponse(response);
      setMessages((current) => current.concat(createMessage("assistant", response.answer)));
      await refreshStatus();
    } catch (requestError) {
      const nextMessage =
        requestError instanceof Error ? requestError.message : "Gui cau hoi that bai.";
      setError(nextMessage);
      setMessages((current) =>
        current.concat(
          createMessage(
            "assistant",
            "Yeu cau that bai. Kiem tra backend hoac build index roi thu lai.",
          ),
        ),
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page-shell">
      <aside className="left-rail">
        <ShellHeader />
        <StatusPanel
          buildMessage={buildMessage}
          buildResult={buildResult}
          building={building}
          onBuild={handleBuildIndex}
          onRefresh={refreshStatus}
          refreshPending={refreshPending}
          sourceDraft={sourceDraft}
          setSourceDraft={setSourceDraft}
          status={status}
        />

        <section className="surface-card sample-card">
          <div className="section-heading">
            <div>
              <p className="eyebrow">Quick Start</p>
              <h2>Cau hoi mau</h2>
            </div>
          </div>
          <div className="sample-list">
            {SAMPLE_QUESTIONS.map((sample) => (
              <button
                key={sample}
                className="sample-chip"
                onClick={() => setDraft(sample)}
              >
                {sample}
              </button>
            ))}
          </div>
        </section>
      </aside>

      <main className="main-stage">
        {error ? <div className="alert-banner">{error}</div> : null}
        <Conversation loading={loading} messages={messages} />
        <Composer busy={loading} draft={draft} onChange={setDraft} onSubmit={handleSubmit} />
        <ContextPanel response={lastResponse} />
      </main>
    </div>
  );
}
