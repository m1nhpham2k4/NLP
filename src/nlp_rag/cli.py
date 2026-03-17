from __future__ import annotations

import argparse

from .config import Settings
from .ingest import build_index
from .service import answer_question


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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    settings = Settings.from_env()

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

    response = answer_question(settings, args.question, top_k=args.top_k)
    if response.rewritten_question != args.question:
        print(f"Normalized query: {response.rewritten_question}\n")
    print(response.answer)
    print("\nRetrieved contexts:")
    for item in response.results:
        print(
            f"- {item.score:.3f} | {item.metadata.get('title', 'Untitled')} | "
            f"{item.metadata.get('section_path', 'root')} | {item.source}"
        )


if __name__ == "__main__":
    main()
