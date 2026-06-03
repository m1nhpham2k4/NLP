from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

from .audit import audit_data
from .config import (
    CANDIDATE_QUESTIONS_PATH,
    CHUNKS_PATH,
    DATA_DIR,
    DEFAULT_EMBEDDING_DEVICE,
    DEFAULT_HISTORY_K,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_QDRANT_COLLECTION_PREFIX,
    DEFAULT_QDRANT_MODE,
    DEFAULT_QDRANT_PATH,
    DEFAULT_QDRANT_URL,
    DEFAULT_VECTOR_STORE,
    EMBEDDING_MODELS,
    FINAL_QUESTIONS_PATH,
    RESULTS_DIR,
    TRIPLES_PATH,
)
from .data import build_chunks, ensure_questions_final, load_questions, save_json
from .device import torch_device_status
from .embeddings import build_embedding_token_report_from_path
from .evaluation import by_type_summary, by_difficulty_summary, evaluate_answers, evaluate_retrieval, evaluate_abstention, evaluate_conflict_detection, evaluate_self_correction
from .pipeline import AgenticGraphRAG
from .web_chat import config_from_args, serve_web_chat


ASK_SYSTEM_CHOICES = [
    "S8_FULL",
]

DENSE_ASK_SYSTEMS = {
    "S8_FULL",
}


def _load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_model_keys(raw: str | None, default: Sequence[str] | None = None) -> List[str] | None:
    if raw:
        return [key.strip() for key in raw.split(",") if key.strip()]
    return list(default) if default is not None else None


def _answer_mode(args: argparse.Namespace) -> str:
    return "openai" if getattr(args, "openai", False) else "extractive"


def _build_rag_from_args(
    args: argparse.Namespace,
    *,
    enable_dense: bool,
    model_keys: Sequence[str] | None = None,
    answer_mode: str | None = None,
) -> AgenticGraphRAG:
    return AgenticGraphRAG.from_path(
        Path(args.chunks),
        enable_dense=enable_dense,
        dense_model_keys=model_keys,
        answer_mode=answer_mode or _answer_mode(args),
        openai_model=getattr(args, "openai_model", DEFAULT_OPENAI_MODEL),
        reranker_model=getattr(args, "reranker", None),
        embedding_device=args.device,
        vector_store=args.vector_store,
        qdrant_mode=args.qdrant_mode,
        qdrant_path=args.qdrant_path,
        qdrant_url=args.qdrant_url,
        qdrant_collection_prefix=args.qdrant_collection_prefix,
    )


def _qdrant_runtime_payload(args: argparse.Namespace) -> Dict[str, Any]:
    if args.vector_store != "qdrant":
        return {"qdrant_mode": None, "qdrant_path": None, "qdrant_url": None, "qdrant_dashboard": None}
    return {
        "qdrant_mode": args.qdrant_mode,
        "qdrant_path": args.qdrant_path if args.qdrant_mode == "local" else None,
        "qdrant_url": args.qdrant_url if args.qdrant_mode == "http" else None,
        "qdrant_dashboard": args.qdrant_url.rstrip("/") + "/dashboard" if args.qdrant_mode == "http" else None,
    }


def _load_history(raw: str | None) -> List[Dict[str, Any]]:
    if not raw:
        return []
    stripped = raw.strip()
    if stripped.startswith("[") or stripped.startswith("{"):
        return json.loads(stripped)
    path = Path(raw)
    try:
        exists = path.exists()
    except OSError:
        exists = False
    if exists:
        return json.loads(path.read_text(encoding="utf-8"))
    return json.loads(raw)


def _history_context(history: List[Dict[str, Any]], history_k: int = DEFAULT_HISTORY_K) -> List[Dict[str, Any]]:
    """Use the last k chat turns as RAG context while still saving full history."""
    if history_k <= 0:
        return []
    return history[-history_k * 2 :]


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


def cmd_preprocess(args: argparse.Namespace) -> None:
    chunks = build_chunks(
        data_dir=Path(args.data_dir),
        output_path=Path(args.output),
        max_chars=args.max_chars,
        overlap_chars=args.overlap_chars,
    )
    print(f"Created {len(chunks)} chunks -> {args.output}")


def cmd_finalize_questions(args: argparse.Namespace) -> None:
    output = ensure_questions_final(Path(args.input), Path(args.output), overwrite=args.overwrite)
    print(f"questions_final ready: {output}")


def cmd_build_graph(args: argparse.Namespace) -> None:
    rag = AgenticGraphRAG.from_path(Path(args.chunks), enable_dense=False)
    rag.graph.save(Path(args.output))
    print(f"Created {len(rag.graph.triples)} triples -> {args.output}")
    conflicts = rag.graph.detect_metadata_conflicts()
    if conflicts:
        print(f"Metadata conflicts detected: {len(conflicts)}")


def _token_stats_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "chunks_path": report.get("chunks_path"),
        "num_chunks": report.get("num_chunks"),
        "models": {},
    }
    for key, model_report in report.get("models", {}).items():
        summary["models"][key] = {
            "tokenizer_loaded": model_report.get("tokenizer_loaded"),
            "tokenizer_error": model_report.get("tokenizer_error"),
            "embedding_model_max_tokens": model_report.get("embedding_model_max_tokens"),
            "total_tokens": model_report.get("total", {}).get("tokens"),
            "over_embedding_limit_chunks": model_report.get("total", {}).get("over_embedding_limit_chunks"),
            "by_category_tokens": {
                category: stats.get("tokens")
                for category, stats in model_report.get("by_category", {}).items()
            },
        }
    return summary


def _write_token_stats(args: argparse.Namespace, model_keys: Sequence[str] | None) -> None:
    output = getattr(args, "token_stats_output", None) or getattr(args, "output", None)
    if not output:
        return
    report = build_embedding_token_report_from_path(
        Path(args.chunks),
        model_keys=model_keys,
        include_chunks=getattr(args, "token_stats_include_chunks", False),
        local_files_only=getattr(args, "token_stats_local_files_only", False),
    )
    save_json(Path(output), report)
    print(json.dumps({"token_stats_saved": output, **_token_stats_summary(report)}, ensure_ascii=False, indent=2))


def cmd_build_dense(args: argparse.Namespace) -> None:
    model_keys = _parse_model_keys(args.models)
    rag = _build_rag_from_args(args, enable_dense=True, model_keys=model_keys, answer_mode="extractive")
    statuses = rag.build_dense_indexes(force=args.force)
    print(
        json.dumps(
            {
                "device": torch_device_status(args.device),
                "vector_store": args.vector_store,
                **_qdrant_runtime_payload(args),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    for name, status in statuses.items():
        print(f"{name}: {status}")
    for key, status in rag.dense_statuses().items():
        print(f"{key}: requested_device={status['requested_device']} resolved_device={status['resolved_device']}")
    _write_token_stats(args, model_keys)


def cmd_embedding_stats(args: argparse.Namespace) -> None:
    model_keys = _parse_model_keys(args.models, default=EMBEDDING_MODELS.keys())
    report = build_embedding_token_report_from_path(
        Path(args.chunks),
        model_keys=model_keys,
        include_chunks=args.include_chunks,
        local_files_only=args.local_files_only,
    )
    save_json(Path(args.output), report)
    print(json.dumps({"saved": args.output, **_token_stats_summary(report)}, ensure_ascii=False, indent=2))


def cmd_audit_data(args: argparse.Namespace) -> None:
    report = audit_data(Path(args.data_dir), Path(args.chunks))
    save_json(Path(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved data audit -> {args.output}")


def cmd_ask(args: argparse.Namespace) -> None:
    model_keys = _parse_model_keys(args.models)
    answer_mode = _answer_mode(args)
    enable_dense = args.dense or args.system in DENSE_ASK_SYSTEMS
    rag = _build_rag_from_args(args, enable_dense=enable_dense, model_keys=model_keys, answer_mode=answer_mode)
    history = _history_context(_load_history(args.history), args.history_k)
    if args.system:
        result = rag.answer_for_system(
            args.system,
            {"question": args.question, "history": history, "role": args.role},
            top_k=args.top_k,
        )
        result["system"] = args.system
    else:
        result = rag.answer(args.question, history=history, role=args.role, top_k=args.top_k)
    if enable_dense and result.get("llm_skip_reason") != "direct_conversation":
        try:
            result["dense_statuses"] = rag.dense_statuses()
        except Exception as exc:
            result["dense_statuses_error"] = str(exc)
    result["device"] = torch_device_status(args.device)
    result["vector_store"] = args.vector_store
    result.update(_qdrant_runtime_payload(args))
    result["reranker_status"] = rag.reranker_status()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(result["answer"])
    if "intent" in result and "role" in result:
        print(f"\nIntent: {result['intent']} | Role: {result['role']} | Confidence: {result['confidence']}")
    else:
        print(f"\nSystem: {result.get('system', 'default')} | Confidence: {result['confidence']}")
    if result.get("llm_model"):
        print(f"LLM: {result['llm_model']}")
    if enable_dense:
        devices = {key: status["resolved_device"] for key, status in rag.dense_statuses().items()}
        print(f"Embedding devices: {devices}")
        print(f"Vector store: {args.vector_store}")
        if args.vector_store == "qdrant":
            if args.qdrant_mode == "local":
                print(f"Qdrant local path: {args.qdrant_path}")
            else:
                print(f"Qdrant dashboard: {args.qdrant_url.rstrip('/')}/dashboard")
    if result["claim_verification"]:
        print("\nClaim verification:")
        for item in result["claim_verification"]:
            print(f"- {item['status']} | {item['score']} | {item['claim']}")


def _save_history(path: str | None, history: List[Dict[str, Any]]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def cmd_chat(args: argparse.Namespace) -> None:
    model_keys = _parse_model_keys(args.models)
    answer_mode = _answer_mode(args)
    enable_dense = args.dense or args.system in DENSE_ASK_SYSTEMS
    rag = _build_rag_from_args(args, enable_dense=enable_dense, model_keys=model_keys, answer_mode=answer_mode)

    if args.history_file and Path(args.history_file).exists():
        history = _load_history(args.history_file)
    else:
        history = _load_history(args.history)
    _save_history(args.history_file, history)

    print("IUH RAG chat đang chạy. Gõ q hoặc /exit để thoát, /history để xem số lượt hội thoại.")
    print(f"History context mặc định: {args.history_k} lượt gần nhất.")
    if enable_dense:
        print(f"Vector store: {args.vector_store}")
        if args.vector_store == "qdrant":
            if args.qdrant_mode == "local":
                print(f"Qdrant local path: {args.qdrant_path}")
            else:
                print(f"Qdrant dashboard: {args.qdrant_url.rstrip('/')}/dashboard")
    while True:
        try:
            question = input("\nBạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in {"/exit", "exit", "quit", "q", ":q"}:
            break
        if question.lower() == "/history":
            print(f"History hiện có {len(history)} message; RAG đang dùng {len(_history_context(history, args.history_k))} message gần nhất.")
            continue
        if question.lower() == "/clear":
            history = []
            _save_history(args.history_file, history)
            print("Đã xóa history của session.")
            continue

        payload = {"question": question, "history": _history_context(history, args.history_k), "role": args.role}
        if args.system:
            result = rag.answer_for_system(args.system, payload, top_k=args.top_k)
            result["system"] = args.system
        else:
            result = rag.answer(question, history=_history_context(history, args.history_k), role=args.role, top_k=args.top_k)

        print("\nBot:")
        print(result["answer"])
        print(f"\nConfidence: {result['confidence']} | Answer mode: {result['answer_mode']}")
        if result.get("llm_model"):
            print(f"LLM: {result['llm_model']}")

        history.append({"role": "user", "content": question})
        history.append(_assistant_history_item(result["answer"], result))
        _save_history(args.history_file, history)


def cmd_serve_chat(args: argparse.Namespace) -> None:
    serve_web_chat(config_from_args(args))


def cmd_openai_test(args: argparse.Namespace) -> None:
    key_set = bool(os.getenv("OPENAI_API_KEY"))
    payload: Dict[str, Any] = {
        "openai_key_set": key_set,
        "model": args.openai_model,
        "ok": False,
        "error": None,
        "response_text": None,
    }
    if not key_set:
        payload["error"] = "OPENAI_API_KEY is not set. Put it in .env or export it in your shell."
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    try:
        from openai import OpenAI

        client = OpenAI()
        response = client.responses.create(
            model=args.openai_model,
            input="Trả lời chính xác chuỗi này: OpenAI OK",
            temperature=0,
        )
        payload["ok"] = True
        payload["response_text"] = response.output_text.strip()
    except Exception as exc:
        payload["error"] = str(exc)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def cmd_evaluate(args: argparse.Namespace) -> None:
    questions = load_questions(Path(args.questions))
    if args.limit:
        questions = questions[: args.limit]

    model_keys = _parse_model_keys(args.models, default=EMBEDDING_MODELS.keys())
    answer_mode = _answer_mode(args)
    rag = _build_rag_from_args(args, enable_dense=args.dense, model_keys=model_keys, answer_mode=answer_mode)
    dense_build_statuses: Dict[str, str] = {}
    if args.dense and args.build_dense:
        dense_build_statuses = rag.build_dense_indexes(force=args.force_dense)

    retrieval_summary = {
        "S8_FULL": evaluate_retrieval(
            questions,
            lambda question: rag.retrieve_for_system("S8_FULL", question, top_k=args.top_k),
        )
    }

    output: Dict[str, Any] = {
        "num_questions": len(questions),
        "top_k": args.top_k,
        "dense_enabled": args.dense,
        "embedding_device": args.device,
        "device": torch_device_status(args.device),
        "vector_store": args.vector_store,
        "answer_mode": answer_mode,
        "openai_model": getattr(args, "openai_model", DEFAULT_OPENAI_MODEL) if answer_mode == "openai" else None,
        "openai_key_set": bool(os.getenv("OPENAI_API_KEY")) if answer_mode == "openai" else None,
        **_qdrant_runtime_payload(args),
        "embedding_models": {key: EMBEDDING_MODELS.get(key, key) for key in model_keys},
        "dense_build_statuses": dense_build_statuses,
        "dense_statuses": rag.dense_statuses(),
        "reranker_status": rag.reranker_status(),
        "systems": rag.system_descriptions(),
        "retrieval": retrieval_summary,
    }

    if args.by_type:
        output["by_type"] = {
            "S8_FULL": by_type_summary(
                questions,
                lambda question: rag.retrieve_for_system("S8_FULL", question, top_k=args.top_k),
            )
        }

    if getattr(args, "by_difficulty", False):
        output["by_difficulty"] = {
            "S8_FULL": by_difficulty_summary(
                questions,
                lambda question: rag.retrieve_for_system("S8_FULL", question, top_k=args.top_k),
            )
        }

    if args.answer_metrics:
        answer_limit = questions[: args.answer_limit] if args.answer_limit else questions
        cached_answer_results = {
            id(question): rag.answer_for_system("S8_FULL", question, top_k=args.top_k)
            for question in answer_limit
        }
        output["answer_metrics"] = {
            "S8_FULL": evaluate_answers(
                answer_limit,
                lambda question: cached_answer_results[id(question)],
            )
        }
        output["abstention_metrics"] = {
            "S8_FULL": evaluate_abstention(
                answer_limit,
                lambda question: cached_answer_results[id(question)],
            )
        }
        output["conflict_metrics"] = {
            "S8_FULL": evaluate_conflict_detection(
                answer_limit,
                lambda question: cached_answer_results[id(question)],
            )
        }
        output["self_correction_metrics"] = {
            "S8_FULL": evaluate_self_correction(
                answer_limit,
                lambda question: cached_answer_results[id(question)],
            )
        }
        output["answer_call_summary"] = {
            "S8_FULL": {
                "n_answers": len(cached_answer_results),
                "llm_used": sum(1 for result in cached_answer_results.values() if result.get("llm_used") is True),
                "llm_skipped": sum(1 for result in cached_answer_results.values() if result.get("llm_skip_reason")),
                "llm_errors": sum(1 for result in cached_answer_results.values() if result.get("llm_error")),
            }
        }

    if args.details:
        detail_rows = []
        for question in questions:
            predictions = {
                "S8_FULL": rag.retrieve_for_system("S8_FULL", question, top_k=args.top_k)
            }
            detail_rows.append(
                {
                    "id": question.get("id"),
                    "question": question.get("question"),
                    "type": question.get("type"),
                    "difficulty": question.get("difficulty"),
                    "relevant_chunk_ids": question.get("relevant_chunk_ids", []),
                    "predictions": predictions,
                }
            )
        output["details"] = detail_rows

    output_path = Path(args.output)
    save_json(output_path, output)
    print(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"Saved evaluation -> {output_path}")


def _add_vector_store_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--vector-store",
        default=DEFAULT_VECTOR_STORE,
        choices=["qdrant", "joblib"],
        help="Dense vector backend. Default uses embedded local Qdrant storage.",
    )
    parser.add_argument(
        "--qdrant-mode",
        default=DEFAULT_QDRANT_MODE,
        choices=["local", "http"],
        help="Qdrant mode. local uses QdrantClient(path=...); http uses a running Qdrant server.",
    )
    parser.add_argument(
        "--qdrant-path",
        default=DEFAULT_QDRANT_PATH,
        help="Local embedded Qdrant storage path. Used only when --qdrant-mode local.",
    )
    parser.add_argument(
        "--qdrant-url",
        default=DEFAULT_QDRANT_URL,
        help="Qdrant HTTP URL. Used only when --qdrant-mode http.",
    )
    parser.add_argument(
        "--qdrant-collection-prefix",
        default=DEFAULT_QDRANT_COLLECTION_PREFIX,
        help="Prefix for Qdrant collections, one collection per embedding model.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="IUH Agentic Graph-RAG CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess = subparsers.add_parser("preprocess", help="Build processed/chunks.json from iuh_data")
    preprocess.add_argument("--data-dir", default=str(DATA_DIR))
    preprocess.add_argument("--output", default=str(CHUNKS_PATH))
    preprocess.add_argument("--max-chars", type=int, default=1400)
    preprocess.add_argument("--overlap-chars", type=int, default=220)
    preprocess.set_defaults(func=cmd_preprocess)

    finalize = subparsers.add_parser("finalize-questions", help="Copy reviewed/candidate questions to evaluation/questions_final.json")
    finalize.add_argument("--input", default=str(CANDIDATE_QUESTIONS_PATH))
    finalize.add_argument("--output", default=str(FINAL_QUESTIONS_PATH))
    finalize.add_argument("--overwrite", action="store_true")
    finalize.set_defaults(func=cmd_finalize_questions)

    graph = subparsers.add_parser("build-graph", help="Build knowledge graph triples")
    graph.add_argument("--chunks", default=str(CHUNKS_PATH))
    graph.add_argument("--output", default=str(TRIPLES_PATH))
    graph.set_defaults(func=cmd_build_graph)

    audit = subparsers.add_parser("audit-data", help="Audit source data and processed chunks for rebuild issues")
    audit.add_argument("--data-dir", default=str(DATA_DIR))
    audit.add_argument("--chunks", default=str(CHUNKS_PATH))
    audit.add_argument("--output", default=str(RESULTS_DIR / "data_audit.json"))
    audit.set_defaults(func=cmd_audit_data)

    dense = subparsers.add_parser("build-dense", help="Build optional dense embedding indexes")
    dense.add_argument("--chunks", default=str(CHUNKS_PATH))
    dense.add_argument("--models", default=None, help="Comma-separated model keys, e.g. miniLM,e5,bge_m3,vietnamese_bi")
    dense.add_argument("--device", default=DEFAULT_EMBEDDING_DEVICE, help="Embedding device: auto uses CUDA if available, otherwise CPU. You can also pass cpu, cuda, or cuda:0.")
    _add_vector_store_args(dense)
    dense.add_argument("--force", action="store_true")
    dense.add_argument("--token-stats-output", default=None, help="Write embedding tokenizer token statistics to this JSON file after building dense indexes.")
    dense.add_argument("--token-stats-include-chunks", action="store_true", help="Include per-chunk token counts in --token-stats-output.")
    dense.add_argument("--token-stats-local-files-only", action="store_true", help="Load tokenizers only from the local Hugging Face cache when writing token stats.")
    dense.set_defaults(func=cmd_build_dense)

    embedding_stats = subparsers.add_parser("embedding-stats", help="Count tokenizer tokens per embedding model and data category")
    embedding_stats.add_argument("--chunks", default=str(CHUNKS_PATH))
    embedding_stats.add_argument("--models", default="miniLM,e5,bge_m3,vietnamese_bi", help="Comma-separated model keys.")
    embedding_stats.add_argument("--output", default=str(RESULTS_DIR / "embedding_token_stats.json"))
    embedding_stats.add_argument("--include-chunks", action="store_true", help="Include token count for every chunk.")
    embedding_stats.add_argument("--local-files-only", action="store_true", help="Load tokenizers only from the local Hugging Face cache.")
    embedding_stats.set_defaults(func=cmd_embedding_stats)

    ask = subparsers.add_parser("ask", help="Ask the Agentic Graph-RAG system")
    ask.add_argument("question")
    ask.add_argument("--chunks", default=str(CHUNKS_PATH))
    ask.add_argument("--role", default=None, choices=["thi_sinh", "phu_huynh", "sinh_vien", "can_bo_tu_van", "general", None])
    ask.add_argument("--history", default=None, help="JSON string or path to JSON chat history")
    ask.add_argument("--history-k", type=int, default=DEFAULT_HISTORY_K, help="Use only the last k chat turns as history context. Default: 3.")
    ask.add_argument("--top-k", type=int, default=5)
    ask.add_argument("--dense", action="store_true")
    ask.add_argument("--models", default=None)
    ask.add_argument("--device", default=DEFAULT_EMBEDDING_DEVICE, help="Embedding/reranker device: auto uses CUDA if available, otherwise CPU. You can also pass cpu, cuda, or cuda:0.")
    _add_vector_store_args(ask)
    ask.add_argument("--system", default=None, choices=ASK_SYSTEM_CHOICES, help="Run a specific retrieval/answer system such as S8_FULL.")
    ask.add_argument("--openai", action="store_true", help="Use OpenAI LLM for answer generation over retrieved evidence.")
    ask.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    ask.add_argument("--reranker", default=None, help="Cross-encoder reranker model, e.g. BAAI/bge-reranker-base. Optional.")
    ask.add_argument("--json", action="store_true")
    ask.set_defaults(func=cmd_ask)

    chat = subparsers.add_parser("chat", help="Run a persistent terminal chatbot with in-session history")
    chat.add_argument("--chunks", default=str(CHUNKS_PATH))
    chat.add_argument("--role", default=None, choices=["thi_sinh", "phu_huynh", "sinh_vien", "can_bo_tu_van", "general", None])
    chat.add_argument("--history", default=None, help="Initial JSON string or path to JSON chat history")
    chat.add_argument("--history-file", default=str(RESULTS_DIR / "chat_history.json"), help="Persist chat history to this JSON file.")
    chat.add_argument("--history-k", type=int, default=DEFAULT_HISTORY_K, help="Use only the last k chat turns as history context. Default: 3.")
    chat.add_argument("--top-k", type=int, default=5)
    chat.add_argument("--dense", action="store_true", default=True)
    chat.add_argument("--models", default="miniLM,e5,bge_m3,vietnamese_bi")
    chat.add_argument("--device", default=DEFAULT_EMBEDDING_DEVICE, help="Embedding/reranker device: auto uses CUDA if available, otherwise CPU. You can also pass cpu, cuda, or cuda:0.")
    _add_vector_store_args(chat)
    chat.add_argument("--system", default="S8_FULL", choices=ASK_SYSTEM_CHOICES, help="Run a specific retrieval/answer system. Default: S8_FULL.")
    chat.add_argument("--openai", action="store_true", help="Use OpenAI LLM for answer generation over retrieved evidence.")
    chat.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    chat.add_argument("--reranker", default=None, help="Cross-encoder reranker model, e.g. BAAI/bge-reranker-base. Optional.")
    chat.set_defaults(func=cmd_chat)

    serve_chat = subparsers.add_parser("serve-chat", help="Run a real-time localhost web chatbot")
    serve_chat.add_argument("--host", default="127.0.0.1")
    serve_chat.add_argument("--port", type=int, default=7860)
    serve_chat.add_argument("--chunks", default=str(CHUNKS_PATH))
    serve_chat.add_argument("--history-file", default=str(RESULTS_DIR / "web_chat_history.json"), help="Persist browser chat histories to this JSON file.")
    serve_chat.add_argument("--history-k", type=int, default=DEFAULT_HISTORY_K, help="Use only the last k chat turns as history context. Default: 3.")
    serve_chat.add_argument("--role", default="general", choices=["thi_sinh", "phu_huynh", "sinh_vien", "can_bo_tu_van", "general"])
    serve_chat.add_argument("--top-k", type=int, default=5)
    serve_chat.add_argument("--models", default="miniLM,e5,bge_m3,vietnamese_bi")
    serve_chat.add_argument("--device", default=DEFAULT_EMBEDDING_DEVICE, help="Embedding/reranker device: auto uses CUDA if available, otherwise CPU. You can also pass cpu, cuda, or cuda:0.")
    _add_vector_store_args(serve_chat)
    serve_chat.add_argument("--system", default="S8_FULL", choices=ASK_SYSTEM_CHOICES, help="Run a specific retrieval/answer system. Default: S8_FULL.")
    serve_chat.add_argument("--openai", action="store_true", help="Use OpenAI LLM for answer generation over retrieved evidence.")
    serve_chat.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    serve_chat.add_argument("--reranker", default=None, help="Cross-encoder reranker model, e.g. BAAI/bge-reranker-base. Optional.")
    serve_chat.set_defaults(func=cmd_serve_chat)

    openai_test = subparsers.add_parser("openai-test", help="Check whether OPENAI_API_KEY and the OpenAI API call work")
    openai_test.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    openai_test.set_defaults(func=cmd_openai_test)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate retrieval/answer metrics")
    evaluate.add_argument("--chunks", default=str(CHUNKS_PATH))
    evaluate.add_argument("--questions", default=str(FINAL_QUESTIONS_PATH))
    evaluate.add_argument("--output", default=str(RESULTS_DIR / "evaluation.json"))
    evaluate.add_argument("--top-k", type=int, default=5)
    evaluate.add_argument("--limit", type=int, default=None)
    evaluate.add_argument("--dense", dest="dense", action="store_true", default=True, help="Enable E1-E4 embedding systems. This is on by default for the full comparison.")
    evaluate.add_argument("--no-dense", dest="dense", action="store_false", help="Disable dense embedding systems for a fast lexical-only smoke test.")
    evaluate.add_argument("--models", default=None)
    evaluate.add_argument("--device", default=DEFAULT_EMBEDDING_DEVICE, help="Embedding/reranker device: auto uses CUDA if available, otherwise CPU. You can also pass cpu, cuda, or cuda:0.")
    _add_vector_store_args(evaluate)
    evaluate.add_argument("--systems", default=None, help="Comma-separated systems to evaluate, e.g. S8_FULL or E1_MINILM,E2_E5.")
    evaluate.add_argument("--build-dense", action="store_true", help="Build/load dense indexes before evaluation instead of lazily during search.")
    evaluate.add_argument("--force-dense", action="store_true", help="Rebuild dense indexes when used with --build-dense.")
    evaluate.add_argument("--by-type", action="store_true")
    evaluate.add_argument("--by-difficulty", action="store_true", help="Compute retrieval metrics grouped by question difficulty (easy/medium/hard).")
    evaluate.add_argument("--answer-metrics", action="store_true")
    evaluate.add_argument("--answer-limit", type=int, default=50)
    evaluate.add_argument("--openai", action="store_true", help="Use OpenAI LLM for S7/S8 answer metrics.")
    evaluate.add_argument("--openai-model", default=DEFAULT_OPENAI_MODEL)
    evaluate.add_argument("--details", action="store_true")
    evaluate.add_argument("--reranker", default=None, help="Cross-encoder reranker model to use during evaluation, e.g. BAAI/bge-reranker-base.")
    evaluate.set_defaults(func=cmd_evaluate)

    return parser


def main() -> None:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
