# Final NLP Evaluation Summary

- Questions evaluated: `2359`
- Top K: `5`
- Device requested/resolved: `auto` / `cpu`
- Runtime seconds: `1024.83`
- Best single embedding: `bge_m3`
- Vector store: Qdrant embedded local at `./qdrant_storage`

## Model / Weight Manifest

This project does not train new neural weights. It uses pretrained embedding model weights plus a built Qdrant vector index.

| Key | Model | Qdrant collection | Points | Vector size |
|---|---|---|---:|---:|
| `miniLM` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | `iuh_rag_minilm_f9fc7e6f` | 2465 | 384 |
| `e5` | `intfloat/multilingual-e5-base` | `iuh_rag_e5_321b015f` | 2465 | 768 |
| `bge_m3` | `BAAI/bge-m3` | `iuh_rag_bge_m3_2022e1cf` | 2465 | 1024 |
| `vietnamese_bi` | `bkai-foundation-models/vietnamese-bi-encoder` | `iuh_rag_vietnamese_bi_9091474f` | 2465 | 768 |

## Overall Retrieval Metrics

Metrics are kept at 6 decimals and include raw hit counts so the denominator is visible.

| System | N | Hit@1 | Hit@3 | Hit@5 | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | nDCG@5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `B1_BM25` | 2359 | 750 | 1094 | 1219 | 0.317931 | 0.463756 | 0.516744 | 0.103349 | 0.394397 | 0.425038 |
| `B2_TFIDF` | 2359 | 514 | 775 | 882 | 0.217889 | 0.328529 | 0.373887 | 0.074777 | 0.277370 | 0.301491 |
| `B3_BM25_TFIDF_4EMB_RRF` | 2359 | 857 | 1264 | 1412 | 0.363290 | 0.535820 | 0.598559 | 0.119712 | 0.454889 | 0.490908 |
| `E1_MINILM` | 2359 | 461 | 735 | 867 | 0.195422 | 0.311573 | 0.367529 | 0.073506 | 0.259149 | 0.286139 |
| `E2_E5` | 2359 | 726 | 1087 | 1240 | 0.307758 | 0.460788 | 0.525646 | 0.105129 | 0.390038 | 0.423912 |
| `E3_BGE_M3` | 2359 | 873 | 1222 | 1352 | 0.370072 | 0.518016 | 0.573124 | 0.114625 | 0.447739 | 0.479114 |
| `E4_VIETNAMESE_BI` | 2359 | 700 | 1050 | 1190 | 0.296736 | 0.445104 | 0.504451 | 0.100890 | 0.375964 | 0.408112 |
| `S1_BEST_SINGLE_EMBEDDING` | 2359 | 873 | 1222 | 1352 | 0.370072 | 0.518016 | 0.573124 | 0.114625 | 0.447739 | 0.479114 |
| `S2_MULTI_EMBEDDING` | 2359 | 812 | 1218 | 1387 | 0.344214 | 0.516320 | 0.587961 | 0.117592 | 0.436739 | 0.474548 |
| `S3_MULTI_EMBEDDING_HYDE` | 2359 | 881 | 1266 | 1399 | 0.373463 | 0.536668 | 0.593048 | 0.118610 | 0.458881 | 0.492534 |
| `S4_MULTI_EMBEDDING_HYDE_HISTORY` | 2359 | 889 | 1270 | 1404 | 0.376855 | 0.538364 | 0.595167 | 0.119033 | 0.461318 | 0.494871 |
| `S5_MULTI_EMBEDDING_HYDE_HISTORY_METADATA` | 2359 | 809 | 1116 | 1225 | 0.342942 | 0.473082 | 0.519288 | 0.103858 | 0.411495 | 0.438528 |
| `S6_S5_GRAPH` | 2359 | 804 | 1113 | 1230 | 0.340822 | 0.471810 | 0.521407 | 0.104281 | 0.410103 | 0.437959 |
| `S7_AGENT_PLANNER` | 2359 | 802 | 1091 | 1198 | 0.339975 | 0.462484 | 0.507842 | 0.101568 | 0.405878 | 0.431454 |
| `S8_FULL` | 2359 | 802 | 1091 | 1198 | 0.339975 | 0.462484 | 0.507842 | 0.101568 | 0.405878 | 0.431454 |

## Generated Artifacts

- `metrics_all_systems.json`
- `predictions_all_questions.json`
- `by_type_metrics.json`
- `by_difficulty_metrics.json`
- `charts/retrieval_metrics_overall.png`
- `charts/embedding_comparison.png`
- `charts/s8_recall_by_type.png`
- `charts/s8_pipeline.png`