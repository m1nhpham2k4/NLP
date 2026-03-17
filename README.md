# IUH Assistant

IUH Assistant la he thong hoi dap RAG cho du lieu IUH, gom:

- `backend/`: FastAPI API
- `frontend/`: React + Vite client
- `src/nlp_rag/`: phan logic ingest, retrieve va generate
- `data/sources/`: du lieu nguon de build index
- `data/index/`: vector index sau khi ingest

Repo nay ho tro 2 cach su dung:

- Chay web app voi backend + frontend
- Chay terminal demo bang CLI

## 1. Yeu cau moi truong

- Python `3.10+`
- Node.js `18+`
- npm

Khuyen nghi:

- Tao virtual environment rieng cho Python
- Dung Node.js `20` neu co

## 2. Cau truc repo

```text
backend/         FastAPI API
frontend/        React + Vite client
src/nlp_rag/     RAG core logic
scripts/         CLI wrappers
scripts/dev/     Script ho tro dev va smoke test
data/sources/    Du lieu nguon
data/index/      Index sinh ra sau ingest
legacy/          Ma va tai nguyen cu, khong nam trong luong chay chinh
```

## 3. Cai dat

### 3.1. Clone repo

```bash
git clone <your-repo-url>
cd NLP
```

### 3.2. Tao va kich hoat virtual environment

Tao virtual environment:

```bash
python -m venv .venv
```

Kich hoat:

Windows CMD:

```bash
.venv\Scripts\activate
```

Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

### 3.3. Cai dat Python dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4. Cai dat frontend dependencies

```bash
cd frontend
npm install
cd ..
```

## 4. Cau hinh moi truong

### 4.1. Backend

Tao file `.env` tu mau:

```bash
copy .env.example .env
```

Neu ban khong dung Windows:

```bash
cp .env.example .env
```

Noi dung mac dinh:

```env
GOOGLE_API_KEY=your_api_key_here
RAG_INDEX_PATH=data/index/rag_index.json
RAG_EMBEDDINGS_PATH=data/index/rag_embeddings.npz
RAG_TOP_K=5
GEMINI_MODEL=gemini-2.5-flash
QUERY_REWRITE_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
ENABLE_QUERY_REWRITE=true
FRONTEND_ORIGINS=http://localhost:5173,http://127.0.0.1:5173
```

Luu y:

- `GOOGLE_API_KEY` can neu ban muon su dung Gemini de rewrite/generate.
- Neu khong co API key, mot so chuc nang sinh cau tra loi se khong day du.
- Lan ingest/query dau tien co the tai model embedding tu Hugging Face.

### 4.2. Frontend

Tao file `frontend/.env` tu mau:

```bash
copy frontend\.env.example frontend\.env
```

Neu ban khong dung Windows:

```bash
cp frontend/.env.example frontend/.env
```

Gia tri mac dinh:

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

## 5. Build index

Truoc khi chat, ban nen build index tu du lieu trong `data/sources/`.

```bash
python scripts/rag_cli.py ingest
```

Neu chi muon ingest mot so nguon cu the:

```bash
python scripts/rag_cli.py ingest --source data/sources/path-a --source data/sources/path-b
```

Sau khi ingest thanh cong, index se duoc ghi ra:

- `data/index/rag_index.json`
- `data/index/rag_embeddings.npz`

## 6. Tu chay web app

Mo **2 terminal** rieng.

### Terminal 1: chay backend

Tu thu muc goc repo:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Backend mac dinh chay tai:

- `http://127.0.0.1:8000`
- `http://127.0.0.1:8000/api/v1/health`
- `http://127.0.0.1:8000/api/v1/status`

Neu muon auto reload trong luc dev:

```bash
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Terminal 2: chay frontend

```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

Frontend mac dinh chay tai:

- `http://127.0.0.1:5173`

## 7. Cach dung sau khi chay

1. Mo frontend tai `http://127.0.0.1:5173`
2. Kiem tra trang thai backend trong giao dien
3. Neu index chua co, build index bang CLI hoac nut build trong giao dien
4. Dat cau hoi va xem context duoc retrieve

## 8. Terminal demo

Neu ban khong muon chay frontend, co the dung terminal demo:

```bash
python scripts/rag_cli.py demo
```

Lenh ho tro trong demo:

- `/help`: xem tro giup nhanh
- `/clear`: xoa lich su chat hien tai
- `/sources`: hien context vua retrieve
- `/exit`: thoat demo

Query mot cau hoi khong vao che do interactive:

```bash
python scripts/rag_cli.py query "Khoa Cong nghe Thong tin dao tao nhung nganh nao?"
```

## 9. Smoke test backend

Script sau dung `TestClient`, khong can backend chay san:

```bash
python scripts/dev/smoke_backend.py
```

## 10. API chinh

Backend expose cac route:

- `GET /api/v1/health`
- `GET /api/v1/status`
- `POST /api/v1/index/build`
- `POST /api/v1/chat`

## 11. Build frontend cho production

```bash
cd frontend
npm run build
```

## 12. Cac file entrypoint quan trong

- Backend app: `backend/app/main.py`
- API routes: `backend/app/api/routes.py`
- Frontend app: `frontend/src/App.tsx`
- Frontend API client: `frontend/src/lib/api.ts`
- RAG service: `src/nlp_rag/service.py`
- CLI: `src/nlp_rag/cli.py`
- Wrapper CLI: `scripts/rag_cli.py`

## 13. Loi thuong gap

### Frontend khong goi duoc backend

Kiem tra:

- Backend da chay o `127.0.0.1:8000`
- File `frontend/.env` co `VITE_API_BASE_URL=http://localhost:8000/api/v1`
- CORS trong `.env` backend cho phep `http://localhost:5173` va `http://127.0.0.1:5173`

### Chat bao thieu index

Hay chay:

```bash
python scripts/rag_cli.py ingest
```

### Ingest/query dau tien cham

Dieu nay thuong do model embedding dang duoc tai lan dau.

### Khong co `GOOGLE_API_KEY`

Repo van co the retrieve, nhung cac buoc rewrite/generate dung Gemini se bi anh huong.

## 14. Cach dung de push len GitHub

Truoc khi push:

- Khong commit file `.env`
- Khong commit cac file log tam
- Kiem tra lai `README.md`
- Neu can, them `data/index/` vao `.gitignore` neu ban khong muon push file index sinh ra

Quy trinh co ban:

```bash
git status
git add .
git commit -m "Update README and run instructions"
git push
```

## 15. Ghi chu

- `legacy/` chi dung de tham khao, khong nam trong luong chay chinh.
- `scripts/dev/*.ps1` van duoc giu lai cho moi truong Windows cu, nhung khong con la huong dan chay chinh trong README nay.
