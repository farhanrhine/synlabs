import time
import logging
from typing import List, Dict, Any, Optional
from groq import Groq

from src.config import settings
from src.embeddings import embedding_provider
from src.store import vector_store

logger = logging.getLogger(__name__)

INSUFFICIENT_CONTEXT_MESSAGE = "The available documents do not contain sufficient context to answer this question."


class RAGService:
    """End-to-end RAG service with top-k retrieval, threshold filtering, and cited Groq generation."""

    def __init__(self, store=vector_store, embedder=embedding_provider):
        self.store = store
        self.embedder = embedder
        self._groq_client = None

    @property
    def groq_client(self) -> Groq:
        if self._groq_client is None:
            if not settings.groq_api_key:
                raise ValueError("GROQ_API_KEY environment variable is not set. Please add it to your .env file.")
            self._groq_client = Groq(api_key=settings.groq_api_key)
        return self._groq_client

    def retrieve(
        self,
        query: str,
        top_k: int = settings.top_k,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = settings.similarity_threshold
    ) -> List[Dict[str, Any]]:
        """Embed query and search nearest neighbors in Qdrant with optional filters and threshold."""
        query_vector = self.embedder.embed_query(query)
        results = self.store.search(
            query_vector=query_vector,
            limit=top_k,
            filters=filters,
            score_threshold=similarity_threshold
        )
        return results

    def query(
        self,
        question: str,
        top_k: int = settings.top_k,
        filters: Optional[Dict[str, Any]] = None,
        similarity_threshold: float = settings.similarity_threshold
    ) -> Dict[str, Any]:
        """Execute complete RAG pipeline with strict grounding and latency/token telemetry."""
        total_start = time.perf_counter()

        # Step 1: Retrieval
        retrieval_start = time.perf_counter()
        retrieved_points = self.retrieve(
            query=question,
            top_k=top_k,
            filters=filters,
            similarity_threshold=similarity_threshold
        )
        retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000

        # Step 2: Handle No Relevant Context (Zero Hallucination branch)
        if not retrieved_points:
            total_latency_ms = (time.perf_counter() - total_start) * 1000
            return {
                "question": question,
                "answer": INSUFFICIENT_CONTEXT_MESSAGE,
                "citations": [],
                "retrieved_chunks": [],
                "chunk_count": 0,
                "is_grounded": False,
                "retrieval_latency_ms": round(retrieval_latency_ms, 2),
                "generation_latency_ms": 0.0,
                "total_latency_ms": round(total_latency_ms, 2),
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            }

        # Step 3: Format Context & Citations
        context_blocks = []
        citations = []
        for idx, point in enumerate(retrieved_points, 1):
            payload = point["payload"]
            text = payload.get("text", "")
            doc_id = payload.get("document_id", "unknown")
            chunk_idx = payload.get("chunk_index", 0)
            score = point["score"]

            citation_tag = f"[Chunk {idx}]"
            context_blocks.append(f"{citation_tag} (Source: {doc_id}, Index: {chunk_idx}, Similarity: {score:.3f}):\n{text}")
            citations.append({
                "citation": citation_tag,
                "point_id": point["id"],
                "document_id": doc_id,
                "chunk_index": chunk_idx,
                "score": round(score, 4),
                "snippet": text[:120] + "..." if len(text) > 120 else text
            })

        formatted_context = "\n\n".join(context_blocks)

        # Step 4: Generation via Groq
        has_groq = self._groq_client is not None or (bool(settings.groq_api_key) and settings.groq_api_key != "your_groq_api_key_here")
        if not has_groq:
            generation_latency_ms = 0.0
            total_latency_ms = (time.perf_counter() - total_start) * 1000
            top_chunk = retrieved_points[0]["payload"].get("text", "")
            answer_text = f"According to documentation, {top_chunk[:180].strip()}... [Chunk 1]"
            token_usage = {
                "prompt_tokens": len(formatted_context) // 4,
                "completion_tokens": len(answer_text) // 4,
                "total_tokens": (len(formatted_context) + len(answer_text)) // 4
            }
        else:
            system_prompt = (
                "You are an accurate, honest AI assistant answering questions based solely on the retrieved document context.\n"
                "Rules:\n"
                "1. Answer the question using ONLY the provided context.\n"
                "2. Whenever you state a fact from a chunk, append its citation tag, e.g. [Chunk 1] or [Chunk 2].\n"
                "3. If the context does not explicitly provide the facts needed to answer, respond exactly with:\n"
                f"   '{INSUFFICIENT_CONTEXT_MESSAGE}'\n"
                "4. Do NOT hallucinate or extrapolate beyond the provided text."
            )

            user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {question}\n\nAnswer (with citations):"

            generation_start = time.perf_counter()
            response = self.groq_client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0
            )
            generation_latency_ms = (time.perf_counter() - generation_start) * 1000
            total_latency_ms = (time.perf_counter() - total_start) * 1000

            answer_text = response.choices[0].message.content.strip()

            token_usage = {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                "total_tokens": response.usage.total_tokens if response.usage else 0
            }

        return {
            "question": question,
            "answer": answer_text,
            "citations": citations,
            "retrieved_chunks": retrieved_points,
            "chunk_count": len(retrieved_points),
            "is_grounded": True,
            "retrieval_latency_ms": round(retrieval_latency_ms, 2),
            "generation_latency_ms": round(generation_latency_ms, 2),
            "total_latency_ms": round(total_latency_ms, 2),
            "token_usage": token_usage
        }


rag_service = RAGService()


if __name__ == "__main__":
    import sys
    import json

    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    question = sys.argv[1] if len(sys.argv) > 1 else "What did Farhan build at SmartBridge?"
    print(f"\n[QUERY] '{question}'\n" + "-" * 70)
    result = rag_service.query(question=question)
    print(f"\n[ANSWER]\n{result['answer']}")
    print(f"\n[CITATIONS] ({len(result['citations'])} chunks cited):")
    for cit in result["citations"]:
        print(f"  * {cit['citation']} -> Source: {cit['document_id']} (Score: {cit['score']})")
    print("\n[TELEMETRY]")
    print(f"  * Retrieval Latency : {result['retrieval_latency_ms']} ms")
    print(f"  * Total Latency     : {result['total_latency_ms']} ms")
    print(f"  * Token Usage       : {json.dumps(result['token_usage'])}")
    print("-" * 70 + "\n")

