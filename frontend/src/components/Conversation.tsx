import type { LocalMessage } from "../types";


interface ConversationProps {
  loading: boolean;
  messages: LocalMessage[];
}


export function Conversation({ loading, messages }: ConversationProps) {
  return (
    <section className="surface-card chat-card">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Conversation</p>
          <h2>Chat với IUH Assistant</h2>
        </div>
        <span className="chat-count">{messages.length} tin nhắn</span>
      </div>

      <div className="message-stack">
        {messages.length === 0 ? (
          <article className="message message-assistant">
            <p className="message-label">Assistant</p>
            <p>
              Chưa có hội thoại. Hãy hỏi về tuyển sinh, khoa, viện, trung tâm
              hoặc quy chế trong dữ liệu IUH.
            </p>
          </article>
        ) : null}

        {messages.map((message) => (
          <article
            key={message.id}
            className={`message ${
              message.role === "user" ? "message-user" : "message-assistant"
            }`}
          >
            <p className="message-label">
              {message.role === "user" ? "Bạn" : "Assistant"}
            </p>
            <p>{message.content}</p>
          </article>
        ))}

        {loading ? (
          <article className="message message-assistant loading-message">
            <p className="message-label">Assistant</p>
            <p>Đang truy xuất ngữ cảnh và tổng hợp câu trả lời...</p>
          </article>
        ) : null}
      </div>
    </section>
  );
}
