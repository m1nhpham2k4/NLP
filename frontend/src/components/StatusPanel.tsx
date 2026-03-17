import type { BuildIndexResponse, StatusResponse } from "../types";


interface StatusPanelProps {
  buildMessage: string | null;
  buildResult: BuildIndexResponse | null;
  building: boolean;
  onBuild: () => void;
  onRefresh: () => void;
  refreshPending: boolean;
  sourceDraft: string;
  setSourceDraft: (value: string) => void;
  status: StatusResponse | null;
}


export function StatusPanel({
  buildMessage,
  buildResult,
  building,
  onBuild,
  onRefresh,
  refreshPending,
  sourceDraft,
  setSourceDraft,
  status,
}: StatusPanelProps) {
  return (
    <section className="surface-card status-card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">System Status</p>
          <h2>Back-end Runtime</h2>
        </div>
        <button className="ghost-button" onClick={onRefresh} disabled={refreshPending}>
          {refreshPending ? "Đang tải..." : "Làm mới"}
        </button>
      </div>

      <div className="status-grid">
        <article>
          <span>Index</span>
          <strong>{status?.index_ready ? "Sẵn sàng" : "Chưa build"}</strong>
        </article>
        <article>
          <span>Generation</span>
          <strong>{status?.gemini_configured ? "Gemini" : "Fallback"}</strong>
        </article>
        <article>
          <span>Top K</span>
          <strong>{status?.top_k ?? "--"}</strong>
        </article>
      </div>

      <dl className="tech-specs">
        <div>
          <dt>Embedding</dt>
          <dd>{status?.embedding_model ?? "Chưa kết nối"}</dd>
        </div>
        <div>
          <dt>Rewrite</dt>
          <dd>{status?.query_rewrite_enabled ? "Bật" : "Tắt"}</dd>
        </div>
        <div>
          <dt>Gemini</dt>
          <dd>{status?.gemini_model ?? "Chưa kết nối"}</dd>
        </div>
      </dl>

      <label className="field-block" htmlFor="sources">
        Nguồn build index
      </label>
      <textarea
        id="sources"
        className="source-input"
        value={sourceDraft}
        onChange={(event) => setSourceDraft(event.target.value)}
        placeholder="Để trống để dùng nguồn mặc định, hoặc nhập mỗi dòng một source."
        rows={5}
      />

      <button className="primary-button" onClick={onBuild} disabled={building}>
        {building ? "Đang build index..." : "Build / Refresh Index"}
      </button>

      {buildMessage ? <p className="inline-note">{buildMessage}</p> : null}

      {buildResult ? (
        <div className="result-card">
          <span>{buildResult.message}</span>
          <strong>{buildResult.chunk_count} chunks</strong>
        </div>
      ) : null}
    </section>
  );
}
