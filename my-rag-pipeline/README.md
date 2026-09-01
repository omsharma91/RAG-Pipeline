# Local RAG Pipeline — my-rag-pipeline

## Overview
A small, local Retrieval-Augmented Generation (RAG) pipeline with a Streamlit UI for offline PDF ingestion, vector storage (Chroma), retrieval, and CPU-friendly synthesis. The app is designed to run locally (CPU) and supports optional local LLMs (`llama-cpp`) and an optional cloud Gemini integration (guarded and explicit).

## Quickstart
1. Create and activate a Python virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install core dependencies (adjust as needed):

```powershell
pip install streamlit sentence-transformers chromadb transformers scikit-learn pypdf
```

Optional (only if you want the features):

```powershell
pip install llama-cpp-python google-generativeai
```

3. Run the app:

```powershell
streamlit run app.py
```

## Files of Interest
- `app.py` — Streamlit UI (ingest, query, synthesis, pre-warm models, Gemini toggle).
- `services/` — `embedder.py`, `chunker.py`, `parser.py` (text extraction, chunking, embeddings).
- `database/vector_store.py` — Chroma wrapper with ingestion telemetry.

## How to Use
- Sidebar -> Ingestion: upload a PDF -> `Inject into Vector DB`.
- Sidebar -> Pre-warm models: loads selected summarizer/generator and optional local or cloud clients to reduce first-call latency.
- Query: enter a question and press Enter. Use `Synthesis method` = `summarize` for chunk-level extraction + polishing (recommended).

## Optional Backends
- Local LLM (`llama-cpp`): provide the local ggml model file path in the sidebar and check `Enable llama-cpp backend`. Install `llama-cpp-python` and obtain a compatible ggml model (e.g., from gpt4all or other providers). Pre-warm to initialize the client.

- Gemini (cloud): enable `Enable Gemini (cloud)` in the sidebar and paste your Gemini API key and model name. WARNING: enabling Gemini sends selected document context and the query to Google — do not upload sensitive/confidential content.

## Recommendations & Tuning
- Chunk size / overlap: tuned in `services/chunker.py`; larger chunks reduce the number of embeddings and speed up ingestion but may reduce granularity.
- TF-IDF top-k prefilter reduces embedding work for large documents — tune `TOP_K` or chunk size for your dataset.
- For better synthesis quality: use `Synthesis: number of top chunks` = 3–6, pre-warm models, and prefer Gemini or a local LLM if available.

## Troubleshooting
- First run may download model weights (transformers); this can take time on CPU.
- If Streamlit shows a `SyntaxError` after edits, restart Streamlit to clear caches: `streamlit run app.py` (stop and re-run).
- If local `llama-cpp` fails to initialize, ensure the ggml model path is correct and `llama-cpp-python` is installed.

## Privacy
Using local models keeps data on your machine. Enabling Gemini sends content to Google; do not upload confidential data when Gemini is enabled.

## Next Steps
- Add a `requirements.txt` with pinned versions (I can add this if you want).
- Add a benchmark script to measure embedding and generation latency.

---

If you'd like, I can add `requirements.txt` and a short benchmark script next.