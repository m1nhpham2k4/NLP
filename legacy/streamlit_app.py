from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from nlp_rag.config import Settings
from nlp_rag.ingest import build_index
from nlp_rag.service import ChatTurn, answer_question, citations_markdown, index_exists


st.set_page_config(
    page_title="IUH Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ink: #172033;
        --muted: #5b6678;
        --line: rgba(23, 32, 51, 0.10);
        --surface: rgba(255, 250, 244, 0.84);
        --surface-strong: rgba(255, 255, 255, 0.78);
        --brand: #0f766e;
        --brand-2: #c2410c;
        --user: #172033;
    }
    .stApp {
        background:
            radial-gradient(circle at 0% 0%, rgba(15,118,110,0.16), transparent 24%),
            radial-gradient(circle at 100% 0%, rgba(194,65,12,0.16), transparent 22%),
            linear-gradient(180deg, #f8f2e7 0%, #ece2d1 100%);
        color: var(--ink);
    }
    .block-container {
        max-width: 1250px;
        padding-top: 1.2rem;
        padding-bottom: 1.5rem;
    }
    .app-shell {
        display: grid;
        grid-template-columns: 340px minmax(0, 1fr);
        gap: 1rem;
    }
    .card {
        background: var(--surface);
        border: 1px solid var(--line);
        border-radius: 24px;
        box-shadow: 0 16px 40px rgba(23, 32, 51, 0.08);
        backdrop-filter: blur(8px);
    }
    .sidebar-card { padding: 1.1rem; }
    .chat-card { padding: 1.1rem 1.1rem 0.8rem 1.1rem; }
    .hero {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.2rem 0.2rem 1rem 0.2rem;
        border-bottom: 1px solid var(--line);
        margin-bottom: 1rem;
    }
    .logo {
        width: 66px;
        height: 66px;
        border-radius: 20px;
        position: relative;
        background: linear-gradient(145deg, #0f766e 0%, #115e59 100%);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.24), 0 14px 24px rgba(15,118,110,0.22);
        overflow: hidden;
        flex: 0 0 auto;
    }
    .logo:before {
        content: "IUH";
        position: absolute;
        left: 11px;
        top: 10px;
        color: #ffffff;
        font-size: 1.1rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        font-family: Georgia, "Times New Roman", serif;
    }
    .logo:after {
        content: "AI";
        position: absolute;
        right: 10px;
        bottom: 9px;
        width: 24px;
        height: 24px;
        border-radius: 999px;
        display: grid;
        place-items: center;
        color: #172033;
        background: linear-gradient(145deg, #facc15 0%, #fb923c 100%);
        font-size: 0.72rem;
        font-weight: 900;
    }
    .hero-title h1 {
        font-family: Georgia, "Times New Roman", serif;
        font-size: 2.05rem;
        line-height: 1.05;
        margin: 0;
        color: var(--ink);
    }
    .hero-title p {
        margin: 0.3rem 0 0 0;
        color: var(--muted);
        font-size: 0.96rem;
    }
    .kpis {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.7rem;
        margin-top: 1rem;
    }
    .kpi {
        background: var(--surface-strong);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.8rem 0.9rem;
    }
    .kpi .label {
        color: var(--muted);
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .kpi .value {
        font-size: 1.15rem;
        font-weight: 800;
        margin-top: 0.15rem;
    }
    .sample-chip {
        background: rgba(255,255,255,0.72);
        border: 1px solid var(--line);
        border-radius: 999px;
        padding: 0.5rem 0.8rem;
        margin: 0.2rem 0;
        font-size: 0.93rem;
    }
    .meta-box {
        margin-top: 1rem;
        background: rgba(255,255,255,0.62);
        border: 1px dashed var(--line);
        border-radius: 18px;
        padding: 0.8rem 0.9rem;
        color: var(--muted);
        font-size: 0.92rem;
    }
    .ctx {
        background: rgba(255,255,255,0.66);
        border: 1px solid var(--line);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        margin-top: 0.75rem;
    }
    @media (max-width: 980px) {
        .app-shell { grid-template-columns: 1fr; }
        .kpis { grid-template-columns: 1fr; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = Settings.from_env()
if "chat_turns" not in st.session_state:
    st.session_state.chat_turns = []
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "draft_question" not in st.session_state:
    st.session_state.draft_question = ""


def _history_for_service() -> list[ChatTurn]:
    return [ChatTurn(role=turn["role"], content=turn["content"]) for turn in st.session_state.chat_turns]


def _push_question(question: str) -> None:
    st.session_state.draft_question = question


with st.sidebar:
    ready = index_exists(settings)
    st.markdown("## Điều khiển")
    st.write(f"Gemini: `{settings.gemini_model}`")
    st.write(f"Embedding: `{settings.embedding_model}`")
    st.write(f"Rewrite: {'on' if settings.enable_query_rewrite else 'off'}")
    st.write(f"API key: {'configured' if settings.google_api_key else 'missing'}")
    st.write(f"Index: {'ready' if ready else 'missing'}")

    if st.button("Build / Refresh Index", use_container_width=True):
        with st.spinner("Đang build index..."):
            total_chunks = build_index(
                project_root=settings.project_root,
                index_path=settings.index_path,
                embeddings_path=settings.embeddings_path,
                model_name=settings.embedding_model,
            )
        st.success(f"Indexed {total_chunks} chunks")
        st.rerun()

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.chat_turns = []
        st.session_state.last_response = None
        st.session_state.draft_question = ""
        st.rerun()

left_col, right_col = st.columns([0.82, 1.78], gap="large")

with left_col:
    st.markdown('<div class="card sidebar-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="logo"></div>
            <div class="hero-title">
                <h1>IUH Assistant</h1>
                <p>Chatbot demo cho dữ liệu tuyển sinh, đào tạo, học vụ, khoa, viện và trung tâm.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="kpis">
            <div class="kpi"><div class="label">Nguồn</div><div class="value">iuh_data</div></div>
            <div class="kpi"><div class="label">Truy xuất</div><div class="value">Hybrid</div></div>
            <div class="kpi"><div class="label">LLM</div><div class="value">Gemini 2.5</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### Gợi ý hỏi nhanh")
    samples = [
        "Thông tin tuyển sinh đại học chính quy mới nhất là gì?",
        "Khoa Công nghệ Thông tin đào tạo những ngành nào?",
        "Điều kiện xét học bổng được nêu như thế nào?",
        "Sinh viên không được làm những hành vi nào theo quy chế?",
        "Trung tâm thư viện có các quy định lưu hành tài liệu nào?",
    ]
    for idx, sample in enumerate(samples):
        if st.button(sample, key=f"sample_{idx}", use_container_width=True):
            _push_question(sample)
    st.markdown(
        f"""
        <div class="meta-box">
            <strong>Trạng thái demo</strong><br/>
            Normalized query sẽ được hiển thị sau mỗi lần hỏi.<br/>
            Chế độ phản hồi hiện tại: {'Gemini' if settings.google_api_key else 'Fallback retrieval'}.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card chat-card">', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="hero">
            <div class="logo"></div>
            <div class="hero-title">
                <h1>IUH Chat</h1>
                <p>Hỏi trực tiếp như một chatbot. Lịch sử hội thoại sẽ được dùng để chuẩn hóa câu hỏi tiếp theo.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.chat_turns:
        for turn in st.session_state.chat_turns:
            with st.chat_message("user" if turn["role"] == "user" else "assistant"):
                st.write(turn["content"])
    else:
        with st.chat_message("assistant"):
            st.write("Chào bạn. Tôi là IUH Assistant. Bạn có thể hỏi về tuyển sinh, học vụ, khoa, viện, trung tâm hoặc các quy chế trong dữ liệu IUH.")

    prompt = st.chat_input("Nhập câu hỏi của bạn...")
    if not prompt and st.session_state.draft_question:
        prompt = st.session_state.draft_question
        st.session_state.draft_question = ""

    if prompt:
        st.session_state.chat_turns.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if not index_exists(settings):
            answer_text = "Chưa có index. Hãy bấm 'Build / Refresh Index' ở thanh bên trái trước khi hỏi."
            st.session_state.chat_turns.append({"role": "assistant", "content": answer_text})
            with st.chat_message("assistant"):
                st.error(answer_text)
        else:
            with st.chat_message("assistant"):
                with st.spinner("Đang truy xuất và tổng hợp câu trả lời..."):
                    response = answer_question(
                        settings,
                        prompt,
                        top_k=settings.top_k,
                        chat_history=_history_for_service()[:-1],
                    )
                st.write(response.answer)
                st.caption(f"Normalized query: {response.rewritten_question}")
                st.caption(f"Citations: {citations_markdown(response.results)}")
            st.session_state.chat_turns.append({"role": "assistant", "content": response.answer})
            st.session_state.last_response = response

    response = st.session_state.last_response
    if response:
        st.markdown("### Context được truy xuất")
        for idx, item in enumerate(response.results, start=1):
            st.markdown(
                f"""
                <div class="ctx">
                    <strong>#{idx} | score={item.score:.3f}</strong><br/>
                    <span style="color:#5b6678;font-size:0.92rem;">{item.metadata.get('title', 'Untitled')} | {item.metadata.get('section_path', 'root')}</span><br/>
                    <span style="color:#5b6678;font-size:0.88rem;">semantic={item.metadata.get('semantic_score', 'n/a')} | lexical={item.metadata.get('lexical_score', 'n/a')}</span>
                    <p>{item.text}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)
