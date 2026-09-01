import logging
import time
import uuid
from datetime import datetime
from typing import Callable, List, Optional

import streamlit as st

from config import COLLECTION_NAME, DB_PATH
from database.vector_store import VectorStore
from services.chunker import sliding_window_chunking
from services.embedder import LocalEmbedder
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    _HAVE_SKLEARN = True
except Exception:
    _HAVE_SKLEARN = False
try:
    from transformers import pipeline
    _HAVE_TRANSFORMERS = True
except Exception:
    _HAVE_TRANSFORMERS = False

try:
    import google.generativeai as genai
    _HAVE_GA = True
except Exception:
    _HAVE_GA = False

try:
    from llama_cpp import Llama
    _HAVE_LLAMA_CPP = True
except Exception:
    _HAVE_LLAMA_CPP = False

# Lazy generator holder
_GEN_PIPELINE = None
_SUMMARIZER_PIPELINE = None
_LLAMA_CLIENT = None
from services.parser import extract_text_from_pdf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@st.cache_resource
def get_embedder() -> LocalEmbedder:
    return LocalEmbedder()


@st.cache_resource
def get_vector_store() -> VectorStore:
    return VectorStore()


def get_vector_store_safe() -> VectorStore:
    store = get_vector_store()
    if not hasattr(store, "get_all_logs"):
        logger.warning("Cached VectorStore instance is stale; recreating resource.")
        get_vector_store.clear()
        store = get_vector_store()
    return store


def inject_pdf(
    file_bytes: bytes,
    filename: str,
    embedder: LocalEmbedder,
    store: VectorStore,
    status_callback: Optional[Callable[[str, int], None]] = None,
) -> int:
    if status_callback:
        status_callback("Extracting text from PDF...", 5)
    text = extract_text_from_pdf(file_bytes)
    if not text:
        raise ValueError("No text extracted from PDF")

    if status_callback:
        status_callback("Chunking text into embeddings...", 20)
    chunks = sliding_window_chunking(text)
    if not chunks:
        raise ValueError("No chunks created from extracted text")

    # Two-stage ingestion: pre-select top-k chunks to embed using TF-IDF
    TOP_K = min(200, max(10, len(chunks)))

    if status_callback:
        status_callback(f"Selecting top {TOP_K} chunks to embed...", 30)

    selected_indices = list(range(len(chunks)))
    if len(chunks) > TOP_K:
        if _HAVE_SKLEARN:
            try:
                vec = TfidfVectorizer(max_features=8192)
                X = vec.fit_transform(chunks)
                # document centroid in TF-IDF space
                centroid = X.mean(axis=0)
                sims = cosine_similarity(X, centroid)
                # sims is (n_chunks, 1) — flatten and take top-k
                sims = sims.reshape(-1)
                selected_indices = sims.argsort()[::-1][:TOP_K].tolist()
            except Exception:
                # fallback to simple sampling if TF-IDF fails
                selected_indices = list(range(0, len(chunks), max(1, len(chunks)//TOP_K)))[:TOP_K]
        else:
            # sklearn not available: coarse sampling evenly across document
            selected_indices = list(range(0, len(chunks), max(1, len(chunks)//TOP_K)))[:TOP_K]

    selected_chunks = [chunks[i] for i in selected_indices]

    start_time = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    start_perf = time.perf_counter()

    if status_callback:
        status_callback("Computing embeddings for selected chunks...", 40)
    # Use a larger batch size to improve throughput
    embeddings = embedder.get_embeddings_batch(selected_chunks, batch_size=256)

    end_perf = time.perf_counter()
    end_time = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    embedding_duration = end_perf - start_perf

    if status_callback:
        status_callback("Saving vectors to the database...", 70)
    # Persist only the selected (embedded) chunks into the vector DB to save time
    ids: List[str] = [f"{uuid.uuid4().hex}" for _ in selected_chunks]
    metadata = [{"source": filename, "chunk_index": int(selected_indices[i])} for i in range(len(selected_chunks))]
    store.upsert_documents(documents=selected_chunks, embeddings=embeddings, metadata=metadata, ids=ids)

    if status_callback:
        status_callback("Recording ingestion telemetry...", 90)
    store.add_log(
        doc_name=filename,
        start_time=start_time,
        end_time=end_time,
        total_chunks=len(chunks),
        embedding_duration=embedding_duration,
    )

    if status_callback:
        status_callback("Injection complete.", 100)

    return len(chunks)


def main() -> None:
    global _LLAMA_CLIENT, _GEN_PIPELINE, _SUMMARIZER_PIPELINE
    st.set_page_config(page_title="Local RAG Pipeline", layout="wide")
    st.title("Local RAG Pipeline — Offline PDF Ingestion")

    embedder = get_embedder()
    store = get_vector_store_safe()

    with st.sidebar:
        st.header("Ingestion")
        uploaded = st.file_uploader("Upload a PDF to ingest", type=["pdf"])
        inject = st.button("Inject into Vector DB")

        status_placeholder = st.empty()
        progress_bar = st.progress(0)

        st.markdown("---")
        st.write("Storage:", DB_PATH)
        st.write("Collection:", COLLECTION_NAME)

        # Local LLM options (optional): allow user to point to a local ggml model
        if _HAVE_LLAMA_CPP:
            st.markdown("**Local LLM options (llama-cpp)**")
            llama_model_path = st.text_input("Local llama model path (ggml)", value="", help="Path to a local ggml model file for llama-cpp-python")
            enable_llama = st.checkbox("Enable llama-cpp backend", value=False)
        else:
            llama_model_path = ""
            enable_llama = False

        # Summarizer options (CPU-friendly)
        st.markdown("**Summarizer options**")
        summarizer_model = st.selectbox(
            "Summarizer model",
            options=["sshleifer/distilbart-cnn-12-6", "facebook/bart-large-cnn"],
            index=0,
            key="summarizer_model",
        )
        summ_min_length = st.number_input("Summarizer: min length", min_value=5, max_value=200, value=10, step=5, key="summ_min")
        summ_max_length = st.number_input("Summarizer: max length", min_value=20, max_value=1024, value=60, step=10, key="summ_max")
        prewarm = st.button("Pre-warm models", key="prewarm_models")

        if prewarm:
            st.sidebar.info("Pre-warming selected models...")
            # Pre-warm transformers pipelines
            if _HAVE_TRANSFORMERS:
                try:
                    if _SUMMARIZER_PIPELINE is None:
                        _SUMMARIZER_PIPELINE = pipeline("summarization", model=summarizer_model, device=-1)
                    if _GEN_PIPELINE is None:
                        _GEN_PIPELINE = pipeline("text-generation", model="distilgpt2", device=-1)
                    st.sidebar.success("Transformers models loaded")
                except Exception as e:
                    logger.exception("Failed to pre-warm transformers: %s", e)
                    st.sidebar.error(f"Transformers pre-warm failed: {e}")
            else:
                st.sidebar.warning("Transformers not available; skipping transformer pre-warm")

            # Pre-warm llama-cpp if requested
            if enable_llama and _HAVE_LLAMA_CPP and llama_model_path:
                try:
                    if _LLAMA_CLIENT is None:
                        _LLAMA_CLIENT = Llama(model_path=llama_model_path)
                    st.sidebar.success("llama-cpp model client initialized")
                except Exception as e:
                    logger.exception("Failed to initialize llama-cpp: %s", e)
                    st.sidebar.error(f"llama-cpp init failed: {e}")
            elif enable_llama:
                st.sidebar.warning("llama-cpp requested but library/model not available or path missing")

            # Configure Gemini client if requested
            if use_gemini:
                if gemini_api_key:
                    try:
                        if _HAVE_GA:
                            try:
                                genai.configure(api_key=gemini_api_key)
                            except Exception:
                                pass
                            st.sidebar.success("Gemini client configured")
                        else:
                            st.sidebar.warning("google.generativeai not installed; cannot pre-warm Gemini client")
                    except Exception as e:
                        logger.exception("Failed to configure Gemini client: %s", e)
                        st.sidebar.error(f"Gemini config failed: {e}")
                else:
                    st.sidebar.warning("Enable Gemini and provide API key to pre-warm")

            st.sidebar.info("Pre-warm completed")

        # Cloud Gemini (optional)
        st.markdown("**Cloud LLM (Gemini) options**")
        use_gemini = st.checkbox("Enable Gemini (cloud)", value=False, key="use_gemini")
        gemini_api_key = ""
        gemini_model = ""
        if use_gemini:
            gemini_api_key = st.text_input("Gemini API key", type="password", key="gem_api")
            gemini_model = st.text_input("Gemini model name", value="gemini-1.0", key="gem_model")
            st.warning(
                "Privacy: Enabling Gemini sends selected document context and your query to Google's servers. "
                "Do not upload sensitive or confidential data. Ensure you have permission to share the content.",
                icon="⚠️",
            )

    def update_status(message: str, percent: int) -> None:
        status_placeholder.info(message)
        progress_bar.progress(min(max(percent, 0), 100))

    if inject:
        if uploaded is None:
            st.sidebar.warning("Please upload a PDF file first.")
        else:
            try:
                status_placeholder.info("Starting injection...")
                progress_bar.progress(0)
                file_bytes = uploaded.read()
                count = inject_pdf(
                    file_bytes=file_bytes,
                    filename=uploaded.name or "uploaded.pdf",
                    embedder=embedder,
                    store=store,
                    status_callback=update_status,
                )
                st.sidebar.success(f"Successfully injected {count} chunks from {uploaded.name}")
            except Exception as e:
                logger.exception("Injection failed: %s", e)
                status_placeholder.error("Injection failed.")
                st.sidebar.error(f"Injection failed: {e}")
            finally:
                progress_bar.progress(100)

    st.markdown("---")
    st.header("Query")
    query = st.text_input("Enter a text query and press Enter")

    # CPU-friendly synthesis controls
    synth_top_k = st.slider("Synthesis: number of top chunks to use", 1, 8, 3)
    synth_max_length = st.slider("Synthesis: max output tokens", 20, 200, 80)

    if query:
        try:
            q_emb = embedder.get_embedding(query)
            # Request more results if user wants to synthesize from more chunks
            n_results = max(3, synth_top_k)
            results = store.query_similar_documents(q_emb, n_results=n_results)

            # Chroma returns lists nested per query; handle defensively
            docs = results.get("documents")
            dists = results.get("distances")
            metas = results.get("metadatas")

            if not docs:
                st.info("No matches found.")
            else:
                items = docs[0] if isinstance(docs[0], (list, tuple)) else docs
                distances = dists[0] if dists and isinstance(dists[0], (list, tuple)) else dists
                metadatas = metas[0] if metas and isinstance(metas[0], (list, tuple)) else metas

                for i, chunk in enumerate(items):
                    score = None
                    try:
                        score = float(distances[i]) if distances else None
                    except Exception:
                        score = None

                    title = f"Match {i+1}"
                    if score is not None:
                        title += f" (score: {score:.4f})"

                    with st.expander(title):
                        st.write(chunk)
                        md = metadatas[i] if metadatas and i < len(metadatas) else {}
                        if md:
                            st.json(md)

                # end of matches loop

            # Option: synthesize a concise answer from top retrieved chunks (outside the loop)
            synth_method = st.selectbox("Synthesis method", ["summarize", "generate"], index=0, key="synth_method")

            def synthesize_answer(question: str, context_chunks: list[str], gen_max_length: int, method: str) -> str:
                """Two-mode synthesis:
                - 'summarize': map-reduce summarization using a summarization pipeline when available
                - 'generate': causal generation fallback (previous behavior)
                """
                global _GEN_PIPELINE, _SUMMARIZER_PIPELINE, _LLAMA_CLIENT

                # trim and limit context size
                max_chars = 2000
                trimmed_chunks = []
                for c in context_chunks:
                    if len(c) > max_chars:
                        trimmed_chunks.append(c[:max_chars])
                    else:
                        trimmed_chunks.append(c)

                if method == "summarize":
                    # Prefer a chunk-level extraction approach: ask a generator to extract facts from each chunk
                    extracted_bits = []

                    for chunk in trimmed_chunks:
                        chunk_prompt = (
                            f"Context chunk:\n{chunk}\n\nQuestion: {question}\n"
                            "Extract any facts or direct answers from the chunk that help answer the question. "
                            "If there is no relevant information in this chunk, reply exactly: NO_RELEVANT_INFO"
                        )

                        # Try Gemini first if enabled
                        answered = None
                        if use_gemini and gemini_api_key and _HAVE_GA:
                            try:
                                genai.configure(api_key=gemini_api_key)
                                resp = genai.generate_text(model=gemini_model or "gemini-1.0", prompt=chunk_prompt, max_output_tokens=150)
                                # extract text
                                text = getattr(resp, "text", None) or (resp.get("candidates", [{}])[0].get("content") if isinstance(resp, dict) else None)
                                if text:
                                    answered = text.strip()
                            except Exception:
                                logger.exception("Gemini chunk extraction failed")

                        # Next, try local llama-cpp if enabled
                        if answered is None and enable_llama and _HAVE_LLAMA_CPP and llama_model_path:
                            try:
                                if _LLAMA_CLIENT is None:
                                    _LLAMA_CLIENT = Llama(model_path=llama_model_path)
                                resp = _LLAMA_CLIENT.create(prompt=chunk_prompt, max_tokens=150, temperature=0.0)
                                try:
                                    answered = resp["choices"][0]["text"].strip()
                                except Exception:
                                    try:
                                        answered = resp.choices[0].text.strip()
                                    except Exception:
                                        answered = str(resp).strip()
                            except Exception:
                                logger.exception("llama chunk extraction failed")

                        # Next, try transformers generator if available
                        if answered is None and _HAVE_TRANSFORMERS:
                            try:
                                if _GEN_PIPELINE is None:
                                    _GEN_PIPELINE = pipeline("text-generation", model="distilgpt2", device=-1)
                                out = _GEN_PIPELINE(chunk_prompt, max_length=150, do_sample=False, num_beams=2)
                                answered = out[0].get("generated_text", "").strip()
                            except Exception:
                                logger.exception("Transformers chunk extraction failed")

                        # Fallback: use summarizer pipeline per-chunk if available
                        if answered is None and _HAVE_TRANSFORMERS:
                            try:
                                if _SUMMARIZER_PIPELINE is None:
                                    _SUMMARIZER_PIPELINE = pipeline("summarization", model=summarizer_model, device=-1)
                                out = _SUMMARIZER_PIPELINE(chunk, max_length=summ_max_length, min_length=summ_min_length)
                                answered = out[0]["summary_text"].strip()
                            except Exception:
                                answered = chunk[:200].strip()

                        if answered and answered != "NO_RELEVANT_INFO":
                            extracted_bits.append(answered)

                    # Combine extracted bits and summarize them to a concise final answer
                    if not extracted_bits:
                        # nothing extracted; fall back to generation later
                        method = "generate"
                    else:
                        combined = "\n\n".join(extracted_bits)
                        if len(combined) > max_chars:
                            combined = combined[:max_chars]
                        # Use summarizer pipeline if available for final polishing
                        if _HAVE_TRANSFORMERS:
                            try:
                                if _SUMMARIZER_PIPELINE is None:
                                    _SUMMARIZER_PIPELINE = pipeline("summarization", model=summarizer_model, device=-1)
                                final = _SUMMARIZER_PIPELINE(combined, max_length=gen_max_length, min_length=20)[0]["summary_text"]
                                return final
                            except Exception:
                                return combined
                        else:
                            return combined

                # Generation fallback
                joined = "\n\n".join(trimmed_chunks)
                prompt = f"Context:\n{joined}\n\nQuestion: {question}\nAnswer:"
                # Prefer cloud Gemini if enabled
                if use_gemini and gemini_api_key:
                    try:
                        # try to call google.generativeai if available
                        if _HAVE_GA:
                            try:
                                genai.configure(api_key=gemini_api_key)
                            except Exception:
                                pass
                            try:
                                # attempt several common argument shapes
                                resp = genai.generate_text(model=gemini_model or "gemini-1.0", prompt=prompt, max_output_tokens=gen_max_length)
                            except TypeError:
                                resp = genai.generate_text(model=gemini_model or "gemini-1.0", text=prompt, max_output_tokens=gen_max_length)

                            # extract text from response
                            text = None
                            try:
                                text = getattr(resp, "text", None)
                            except Exception:
                                text = None
                            if not text:
                                try:
                                    text = resp["candidates"][0]["content"]
                                except Exception:
                                    try:
                                        text = resp["candidates"][0]["text"]
                                    except Exception:
                                        text = str(resp)
                            return text.strip()
                    except Exception:
                        logger.exception("Gemini generation failed")
                # Prefer local llama-cpp backend if the user enabled it and provided a model path
                if enable_llama and _HAVE_LLAMA_CPP and llama_model_path:
                    try:
                        if _LLAMA_CLIENT is None:
                            _LLAMA_CLIENT = Llama(model_path=llama_model_path)
                        resp = _LLAMA_CLIENT.create(prompt=prompt, max_tokens=gen_max_length, temperature=0.0)
                        # llama-cpp-python returns text in choices[0].text or choices[0]['text'] depending on version
                        text = None
                        try:
                            text = resp['choices'][0]['text']
                        except Exception:
                            try:
                                text = resp.choices[0].text
                            except Exception:
                                text = str(resp)
                        return text.strip()
                    except Exception:
                        # fall through to transformers if available
                        logger.exception("Local llama generation failed")

                if _HAVE_TRANSFORMERS and method == "generate":
                    try:
                        if _GEN_PIPELINE is None:
                            _GEN_PIPELINE = pipeline("text-generation", model="distilgpt2", device=-1)
                        out = _GEN_PIPELINE(prompt, max_length=gen_max_length, do_sample=False, num_beams=2)
                        return out[0]["generated_text"][len(prompt):].strip()
                    except Exception:
                        return "(generation failed)\n\n" + joined

                return "(no generator installed)\n\n" + joined

            if st.button("Synthesize Answer from Matches", key="synth_button"):
                top_n = min(synth_top_k, len(items))
                top_chunks = items[:top_n]
                with st.spinner("Generating concise answer from retrieved context..."):
                    synth = synthesize_answer(query, top_chunks, gen_max_length=synth_max_length, method=synth_method)
                    st.subheader("Synthesis")
                    st.write(synth)

        except Exception as e:
            logger.exception("Query failed: %s", e)
            st.error(f"Query failed: {e}")

    st.markdown("---")
    st.header("📊 Data Ingestion Telemetry Logs")
    logs = store.get_all_logs()
    if logs:
        st.dataframe(logs)
    else:
        st.info("No ingestion telemetry logs are available yet.")


if __name__ == "__main__":
    main()
