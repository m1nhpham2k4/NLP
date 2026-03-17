import type { ChatResponse } from "../types";


interface ContextPanelProps {
  response: ChatResponse | null;
}


export function ContextPanel({ response }: ContextPanelProps) {
  return (
    <section className="surface-card context-card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Retrieved Context</p>
          <h2>Ngữ cảnh và tín hiệu truy xuất</h2>
        </div>
      </div>

      {!response ? (
        <p className="empty-copy">
          Sau khi gửi câu hỏi, normalized query, trích dẫn và các đoạn context
          sẽ hiển thị ở đây.
        </p>
      ) : (
        <>
          <div className="response-meta">
            <article>
              <span>Normalized query</span>
              <strong>{response.rewritten_question}</strong>
            </article>
            <article>
              <span>Generation</span>
              <strong>{response.used_generation ? "Gemini" : "Fallback"}</strong>
            </article>
            <article>
              <span>Timestamp</span>
              <strong>{response.timestamp_utc}</strong>
            </article>
          </div>

          <div className="citations-block">
            <p className="field-block">Citations</p>
            <pre>{response.citations}</pre>
          </div>

          <div className="context-stack">
            {response.contexts.map((context, index) => (
              <article key={`${context.source}-${index}`} className="context-item">
                <div className="context-topline">
                  <strong>#{index + 1}</strong>
                  <span>{context.score.toFixed(3)}</span>
                </div>
                <p className="context-title">
                  {context.metadata.title ?? "Untitled"}
                </p>
                <p className="context-meta">
                  {context.metadata.section_path ?? "root"} · {context.source}
                </p>
                <p>{context.text}</p>
              </article>
            ))}
          </div>
        </>
      )}
    </section>
  );
}
