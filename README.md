# IUH Assistant

Day la cau truc final de test san pham sau khi da tach va don lai repo.

## Thu muc chinh

```text
backend/         FastAPI API
frontend/        React + Vite client
src/nlp_rag/     RAG core logic
data/sources/    Du lieu nguon cho ingest
data/index/      Vector index sinh ra khi build
scripts/dev/     Script chay nhanh va smoke test
legacy/          Ma va tai nguyen cu da duoc dua ra khoi luong chay chinh
```

## Luong chay chinh

- Backend: [backend/app/main.py](backend/app/main.py)
- Frontend: [frontend/src/App.tsx](frontend/src/App.tsx)
- RAG core: [src/nlp_rag/service.py](src/nlp_rag/service.py)
- Ingest CLI: [scripts/build_index.py](scripts/build_index.py)

## Cai dat

### Python

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

### Frontend

```bash
cd frontend
copy .env.example .env
npm install
cd ..
```

## Chay de test

### 1. Build index

```bash
powershell -ExecutionPolicy Bypass -File scripts/dev/build_index.ps1
```

### 2. Chay backend

```bash
powershell -ExecutionPolicy Bypass -File scripts/dev/start_backend.ps1
```

API mac dinh:

- `GET /api/v1/health`
- `GET /api/v1/status`
- `POST /api/v1/index/build`
- `POST /api/v1/chat`

### 3. Chay frontend

```bash
powershell -ExecutionPolicy Bypass -File scripts/dev/start_frontend.ps1
```

Frontend mac dinh chay tai `http://localhost:5173`.

### 4. Smoke test backend

```bash
python scripts/dev/smoke_backend.py
```

## Bien moi truong

```env
GOOGLE_API_KEY=
RAG_INDEX_PATH=data/index/rag_index.json
RAG_EMBEDDINGS_PATH=data/index/rag_embeddings.npz
RAG_TOP_K=5
GEMINI_MODEL=gemini-2.5-flash
QUERY_REWRITE_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENABLE_QUERY_REWRITE=true
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

## Ghi chu

- Du lieu ingest mac dinh da duoc chuyen vao `data/sources/`.
- Phan Streamlit cu khong con nam trong luong chay chinh; no da duoc chuyen sang `legacy/streamlit_app.py`.
- Neu may chua co model embedding, lan build index hoac lan query dau tien se tai model tu Hugging Face.
