import os
from dotenv import load_dotenv

load_dotenv()

# Configuration constants with sensible fallbacks
DB_PATH = os.getenv("DB_PATH", "./my_vector_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_collection")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

__all__ = ["DB_PATH", "COLLECTION_NAME", "EMBEDDING_MODEL_NAME"]
