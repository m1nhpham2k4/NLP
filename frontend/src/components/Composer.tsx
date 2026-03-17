interface ComposerProps {
  busy: boolean;
  draft: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
}


export function Composer({ busy, draft, onChange, onSubmit }: ComposerProps) {
  return (
    <section className="surface-card composer-card">
      <label className="field-block" htmlFor="question">
        Câu hỏi
      </label>
      <textarea
        id="question"
        className="composer-input"
        value={draft}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Ví dụ: Khoa Công nghệ Thông tin đào tạo những ngành nào?"
        rows={4}
      />
      <button
        className="primary-button"
        onClick={onSubmit}
        disabled={busy || draft.trim().length === 0}
      >
        {busy ? "Đang gửi..." : "Gửi câu hỏi"}
      </button>
    </section>
  );
}
