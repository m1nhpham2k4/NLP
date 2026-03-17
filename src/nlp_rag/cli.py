from __future__ import annotations

import argparse
from pathlib import Path

from .config import Settings
from .ingest import build_index
from .service import ChatTurn, RagResponse, answer_question, index_exists


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IUH NLP RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build the local vector index")
    ingest_parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        help="Optional source path relative to the project root. Repeat for multiple values.",
    )

    query_parser = subparsers.add_parser("query", help="Search the local RAG index")
    query_parser.add_argument("question", help="Question to ask")
    query_parser.add_argument("--top-k", type=int, default=None, help="Number of passages to retrieve")

    demo_parser = subparsers.add_parser("demo", help="Run an interactive terminal chat demo")
    demo_parser.add_argument("--top-k", type=int, default=None, help="Number of passages to retrieve")

    return parser


def _print_sources(response: RagResponse) -> None:
    print("\nRetrieved contexts:")
    if not response.results:
        print("- No matching passages found.")
        return

    for item in response.results:
        print(
            f"- {item.score:.3f} | {item.metadata.get('title', 'Untitled')} | "
            f"{item.metadata.get('section_path', 'root')} | {item.source}"
        )


def run_demo(settings: Settings, top_k: int | None = None) -> None:
    if not index_exists(settings):
        raise FileNotFoundError("RAG index files are missing. Build the index first.")

    history: list[ChatTurn] = []
    last_response: RagResponse | None = None
    print("IUH Assistant terminal demo")
    print("Commands: /help, /clear, /sources, /exit")

    while True:
        try:
            question = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting terminal demo.")
            return

        if not question:
            continue

        lowered = question.lower()
        if lowered in {"/exit", "/quit", "exit", "quit"}:
            print("Exiting terminal demo.")
            return
        if lowered == "/help":
            print("Ask a question about IUH data. Use /clear to reset chat history.")
            print("Use /sources to show the latest retrieved contexts.")
            continue
        if lowered == "/clear":
            history.clear()
            last_response = None
            print("Chat history cleared.")
            continue
        if lowered == "/sources":
            if last_response is None:
                print("No retrieved contexts yet.")
            else:
                _print_sources(last_response)
            continue

        response = answer_question(settings, question, top_k=top_k, chat_history=history)
        history.append(ChatTurn(role="user", content=question))
        history.append(ChatTurn(role="assistant", content=response.answer))
        last_response = response

        if response.rewritten_question != question:
            print(f"\nNormalized query: {response.rewritten_question}")
        print(f"\nAssistant> {response.answer}")
        _print_sources(response)


def main(project_root: Path | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args()
    settings = Settings.from_env(project_root=project_root)

    if args.command == "ingest":
        total_chunks = build_index(
            project_root=settings.project_root,
            index_path=settings.index_path,
            embeddings_path=settings.embeddings_path,
            model_name=settings.embedding_model,
            requested_sources=args.sources,
        )
        print(f"Indexed {total_chunks} chunks into {settings.index_path}")
        return

    if args.command == "demo":
        run_demo(settings, top_k=args.top_k)
        return

    response = answer_question(settings, args.question, top_k=args.top_k)
    if response.rewritten_question != args.question:
        print(f"Normalized query: {response.rewritten_question}\n")
    print(response.answer)
    _print_sources(response)


if __name__ == "__main__":
    main()
