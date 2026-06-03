# S8_FULL OpenAI Evaluation

- OpenAI model: `gpt-4.1-mini`
- OPENAI_API_KEY set: `True`
- Device requested/resolved: `auto` / `cpu`
- Vector store: `qdrant`
- Qdrant mode/path: `local` / `./qdrant_storage`
- Runtime seconds: `384.45`

| Limit | LLM Used | LLM Errors | Recall@5 | Answer Correctness | Citation Accuracy | Hallucination Rate |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 5 | 0 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| 10 | 10 | 0 | 0.9000 | 0.9417 | 0.9000 | 0.0000 |
| 50 | 50 | 0 | 0.7000 | 0.9233 | 0.7000 | 0.0000 |
| 100 | 100 | 0 | 0.7500 | 0.9192 | 0.7500 | 0.0000 |

## Files

- `run_manifest.json`
- `s8_openai_prefix_metrics.json`
- `s8_openai_predictions.json`