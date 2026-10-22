from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.config import settings
from src.embeddings import embedding_provider
from src.store import vector_store
from src.ingestion import ingestion_pipeline
from src.rag import rag_service

app = FastAPI(
    title="Cost-Efficient RAG Service",
    description="Production-grade RAG QA service backed by embedded Qdrant, FastEmbed, and Groq.",
    version="1.0.0"
)


class IngestRequest(BaseModel):
    directory_path: Optional[str] = Field(default=None, description="Path to directory containing documents")
    chunk_size: Optional[int] = Field(default=None, description="Chunk character size")
    chunk_overlap: Optional[int] = Field(default=None, description="Chunk overlap size")


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    chunks_processed: int
    total_points_in_store: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=2, description="User question to answer")
    top_k: Optional[int] = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters (e.g. source_type)")
    similarity_threshold: Optional[float] = Field(default=None, description="Minimum cosine similarity")


class CitationItem(BaseModel):
    citation: str
    point_id: str
    document_id: str
    chunk_index: int
    score: float
    snippet: str


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationItem]
    chunk_count: int
    is_grounded: bool
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    token_usage: TokenUsage


@app.get("/", status_code=status.HTTP_200_OK)
def root() -> Dict[str, Any]:
    """Root landing endpoint with links to documentation and health check."""
    return {
        "service": "SynLabs Cost-Efficient RAG Service",
        "status": "online",
        "documentation": "/docs",
        "health": "/health",
        "endpoints": {
            "query": "POST /query",
            "ingest": "POST /ingest",
            "health": "GET /health"
        }
    }


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, Any]:
    """Report service health, model configuration, and vector store stats."""
    return {
        "status": "healthy",
        "embedding_model": embedding_provider.get_info(),
        "generation_model": settings.groq_model,
        "vector_store": {
            "type": "qdrant-embedded",
            "collection": settings.collection_name,
            "total_points": vector_store.count()
        }
    }


@app.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_200_OK)
def ingest_documents(req: IngestRequest) -> IngestResponse:
    """Ingest documents from directory into Qdrant idempotently."""
    try:
        dir_path = req.directory_path or settings.raw_data_path
        c_size = req.chunk_size or settings.chunk_size
        c_overlap = req.chunk_overlap or settings.chunk_overlap

        result = ingestion_pipeline.ingest_directory(
            directory_path=dir_path,
            chunk_size=c_size,
            chunk_overlap=c_overlap
        )
        return IngestResponse(
            status=result["status"],
            files_processed=result.get("files_processed", 0),
            chunks_processed=result.get("chunks_processed", 0),
            total_points_in_store=result.get("total_points_in_store", 0)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Ingestion failed: {str(e)}"
        )


@app.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
def query_corpus(req: QueryRequest) -> QueryResponse:
    """Answer a user question using grounded retrieval and citations."""
    try:
        k = req.top_k or settings.top_k
        thresh = req.similarity_threshold or settings.similarity_threshold

        result = rag_service.query(
            question=req.question,
            top_k=k,
            filters=req.filters,
            similarity_threshold=thresh
        )
        return QueryResponse(**result)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Query error: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.host, port=settings.port)
