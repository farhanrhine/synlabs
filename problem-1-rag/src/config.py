from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # LLM Settings
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"

    # Embedding Settings (FastEmbed local ONNX)
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Chunking Defaults
    chunk_size: int = 500
    chunk_overlap: int = 100

    # Retrieval Defaults
    top_k: int = 5
    similarity_threshold: float = 0.55

    # Storage Paths
    qdrant_storage_path: str = "./data/qdrant"
    collection_name: str = "corpus_docs"
    raw_data_path: str = "./data/raw"

    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000

    model_config = SettingsConfigDict(
        env_file=Path(__file__).parent.parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
