import logging
import uuid
from typing import Dict, List

import chromadb

from config import DB_PATH, COLLECTION_NAME

logger = logging.getLogger(__name__)


class VectorStore:
    """A thin wrapper around ChromaDB for local persistent vector storage."""

    LOG_COLLECTION_NAME = "injection_logs"

    def __init__(self):
        try:
            self.client = chromadb.PersistentClient(path=DB_PATH)
            self.collection = self.client.get_or_create_collection(name=COLLECTION_NAME)
            self.log_collection = self.client.get_or_create_collection(name=self.LOG_COLLECTION_NAME)
        except Exception as e:
            logger.exception("Failed to initialize ChromaDB client: %s", e)
            raise

    def persist(self) -> None:
        try:
            if hasattr(self.client, "persist"):
                self.client.persist()
        except Exception as e:
            logger.exception("Failed to persist ChromaDB: %s", e)

    def upsert_documents(self, documents: List[str], embeddings: List[List[float]], metadata: List[dict], ids: List[str]) -> None:
        """Insert or update documents with their embeddings and metadata.

        Args:
            documents: The textual chunks.
            embeddings: The corresponding vectors.
            metadata: A list of metadata dicts for each document.
            ids: Unique string identifiers for each document.
        """
        try:
            # Use upsert to allow re-inserting documents with the same ids
            self.collection.upsert(documents=documents, embeddings=embeddings, metadatas=metadata, ids=ids)
            self.persist()
        except Exception as e:
            logger.exception("Failed to upsert documents into vector store: %s", e)
            raise

    def add_log(self, doc_name: str, start_time: str, end_time: str, total_chunks: int, embedding_duration: float) -> None:
        """Save ingestion telemetry to the dedicated log collection."""
        metadata = {
            "document_name": doc_name,
            "start_time": start_time,
            "end_time": end_time,
            "total_chunks": total_chunks,
            "embedding_duration": embedding_duration,
        }
        log_id = uuid.uuid4().hex

        try:
            self.log_collection.add(
                ids=[log_id],
                documents=[doc_name],
                metadatas=[metadata],
            )
            self.persist()
        except Exception as e:
            logger.exception("Failed to add ingestion log: %s", e)
            raise

    def get_all_logs(self) -> List[Dict]:
        """Return all ingestion logs sorted by most recent start time."""
        try:
            result = self.log_collection.get(include=["documents", "metadatas"])
            documents = result.get("documents") or []
            metadatas = result.get("metadatas") or []

            if documents and isinstance(documents[0], (list, tuple)):
                documents = documents[0]
            if metadatas and isinstance(metadatas[0], (list, tuple)):
                metadatas = metadatas[0]

            logs = []
            for i, metadata in enumerate(metadatas):
                logs.append({
                    "Document Name": documents[i] if i < len(documents) else metadata.get("document_name", ""),
                    "Injection Start Time": metadata.get("start_time", ""),
                    "Injection End Time": metadata.get("end_time", ""),
                    "Total Chunks Flattened": metadata.get("total_chunks", 0),
                    "Embedding Processing Time": f"{metadata.get('embedding_duration', 0):.2f} seconds",
                })

            logs.sort(key=lambda item: item.get("Injection Start Time", ""), reverse=True)
            return logs
        except Exception as e:
            logger.exception("Failed to retrieve ingestion logs: %s", e)
            return []

    def query_similar_documents(self, query_embedding: List[float], n_results: int = 3) -> Dict:
        """Query the collection for the most similar documents to `query_embedding`.

        Returns the raw ChromaDB query result dict. If an error occurs an empty
        result structure is returned.
        """
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            logger.exception("Vector store query failed: %s", e)
            return {"documents": [], "metadatas": [], "distances": []}
