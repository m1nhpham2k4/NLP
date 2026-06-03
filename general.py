import json
import time
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from tqdm import tqdm


# =========================
# CONFIG
# =========================

CHUNKS_PATH = Path("processed/chunks.json")
OUTPUT_PATH = Path("processed/candidate_questions.json")
FAILED_PATH = Path("processed/failed_chunks.json")

# Có thể đổi model tùy tài khoản/cost.
# gpt-4.1-mini: thường đủ tốt cho sinh benchmark.
# gpt-4.1: chất lượng cao hơn nhưng tốn hơn.
MODEL_NAME = "gpt-4.1-mini"

MIN_CHUNK_LENGTH = 300
MAX_CHUNKS = None  # ví dụ đặt 50 để test trước, None là chạy toàn bộ
QUESTIONS_PER_CHUNK = "3-5"

client = OpenAI()


# =========================
# PROMPT
# =========================

def build_prompt(chunk: Dict[str, Any]) -> str:
    return f"""
Bạn là người tạo bộ câu hỏi đánh giá cho hệ thống RAG hỏi đáp thông tin Trường Đại học Công nghiệp TP.HCM.

Nhiệm vụ:
Từ đoạn tài liệu bên dưới, hãy tạo {QUESTIONS_PER_CHUNK} câu hỏi kiểm thử.

Yêu cầu bắt buộc:
1. Mỗi câu hỏi phải trả lời được trực tiếp từ đoạn tài liệu.
2. Không được bịa thông tin ngoài đoạn tài liệu.
3. Câu trả lời phải ngắn gọn, chính xác.
4. Nếu đoạn tài liệu không đủ thông tin để tạo câu hỏi chất lượng, trả về [].
5. Không tạo câu hỏi quá chung chung như:
   - "Đoạn trên nói về điều gì?"
   - "Tóm tắt nội dung trên."
   - "Thông tin chính là gì?"
6. Câu hỏi nên đa dạng:
   - câu hỏi fact đơn giản
   - câu hỏi có keyword rõ ràng
   - câu hỏi diễn đạt tự nhiên
   - câu hỏi liệt kê nếu phù hợp
7. Output bắt buộc là JSON array hợp lệ.
8. Không markdown, không giải thích, không thêm text ngoài JSON.

Metadata:
- chunk_id: {chunk["chunk_id"]}
- file_name: {chunk["file_name"]}
- relative_path: {chunk.get("relative_path", "")}
- type: {chunk["type"]}

Đoạn tài liệu:
\"\"\"
{chunk["text"]}
\"\"\"

Schema output:
[
  {{
    "question": "...",
    "answer": "...",
    "type": "{chunk["type"]}",
    "difficulty": "easy | medium | hard",
    "relevant_files": ["{chunk["file_name"]}"],
    "relevant_chunk_ids": ["{chunk["chunk_id"]}"],
    "expected_keywords": ["...", "..."]
  }}
]
""".strip()


# =========================
# HELPERS
# =========================

def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def parse_json_array(text: str) -> List[Dict[str, Any]]:
    """
    Cố gắng parse JSON array từ output của model.
    Nếu model lỡ bọc markdown thì vẫn xử lý được.
    """
    text = text.strip()

    if text.startswith("```json"):
        text = text.replace("```json", "", 1).strip()
    if text.startswith("```"):
        text = text.replace("```", "", 1).strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("Không tìm thấy JSON array trong output.")

    json_text = text[start:end + 1]
    data = json.loads(json_text)

    if not isinstance(data, list):
        raise ValueError("Output không phải JSON array.")

    return data


def validate_question_item(item: Dict[str, Any], chunk: Dict[str, Any]) -> bool:
    required_fields = [
        "question",
        "answer",
        "type",
        "difficulty",
        "relevant_files",
        "relevant_chunk_ids",
        "expected_keywords",
    ]

    for field in required_fields:
        if field not in item:
            return False

    if not item["question"] or not item["answer"]:
        return False

    if chunk["chunk_id"] not in item["relevant_chunk_ids"]:
        return False

    if chunk["file_name"] not in item["relevant_files"]:
        return False

    return True


def call_openai_generate(chunk: Dict[str, Any], max_retries: int = 3) -> List[Dict[str, Any]]:
    prompt = build_prompt(chunk)

    for attempt in range(max_retries):
        try:
            response = client.responses.create(
                model=MODEL_NAME,
                input=prompt,
                temperature=0.2,
            )

            output_text = response.output_text
            questions = parse_json_array(output_text)

            valid_questions = []
            for item in questions:
                if validate_question_item(item, chunk):
                    valid_questions.append(item)

            return valid_questions

        except Exception as e:
            wait_time = 2 ** attempt
            print(f"\nError at chunk {chunk['chunk_id']} | attempt {attempt + 1}: {e}")
            time.sleep(wait_time)

    raise RuntimeError(f"Failed after {max_retries} retries: {chunk['chunk_id']}")


# =========================
# MAIN
# =========================

def main():
    chunks = load_json(CHUNKS_PATH)

    selected_chunks = [
        c for c in chunks
        if len(c.get("text", "")) >= MIN_CHUNK_LENGTH
    ]

    if MAX_CHUNKS is not None:
        selected_chunks = selected_chunks[:MAX_CHUNKS]

    all_questions = []
    failed_chunks = []

    print(f"Total chunks loaded: {len(chunks)}")
    print(f"Selected chunks: {len(selected_chunks)}")
    print(f"Model: {MODEL_NAME}")

    # Resume nếu đã có output trước đó
    if OUTPUT_PATH.exists():
        all_questions = load_json(OUTPUT_PATH)
        done_chunk_ids = set()

        for q in all_questions:
            for cid in q.get("relevant_chunk_ids", []):
                done_chunk_ids.add(cid)

        selected_chunks = [
            c for c in selected_chunks
            if c["chunk_id"] not in done_chunk_ids
        ]

        print(f"Existing questions: {len(all_questions)}")
        print(f"Remaining chunks: {len(selected_chunks)}")

    question_counter = len(all_questions) + 1

    for chunk in tqdm(selected_chunks, desc="Generating questions"):
        try:
            questions = call_openai_generate(chunk)

            for q in questions:
                q["id"] = f"q{question_counter:04d}"
                question_counter += 1
                all_questions.append(q)

            # Lưu liên tục để tránh mất dữ liệu nếu bị lỗi giữa chừng
            save_json(OUTPUT_PATH, all_questions)

        except Exception as e:
            failed_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "file_name": chunk["file_name"],
                "relative_path": chunk.get("relative_path", ""),
                "error": str(e),
            })
            save_json(FAILED_PATH, failed_chunks)

    save_json(OUTPUT_PATH, all_questions)
    save_json(FAILED_PATH, failed_chunks)

    print("\nDone.")
    print(f"Total generated questions: {len(all_questions)}")
    print(f"Failed chunks: {len(failed_chunks)}")
    print(f"Saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()