"""
Query engine RAG com LlamaIndex + Gemini/Ollama.

Embedding: multilingual-e5-large (local, 1024 dims)
Geração:   configurável — Gemini Flash, Gemma2, Qwen2.5, Phi4
Retrieval: MotherDuck VSS com filtros por metadados.
"""

import os
from dataclasses import dataclass
from datetime import date

from litellm import completion

# LiteLLM usa GEMINI_API_KEY; garante compatibilidade com GOOGLE_AI_API_KEY
if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_AI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_AI_API_KEY"]

from indexing.embedder import embed_query as embed_query_local
from indexing.store import get_connection, similarity_search
from rag.prompts import QA_TEMPLATE, SYSTEM_PROMPT


@dataclass
class QueryResult:
    answer: str
    source_chunks: list[dict]
    latency_ms: float
    model: str


def query(
    question: str,
    filters: dict | None = None,
    top_k: int = 6,
    model: str = "gemini/gemini-2.0-flash",
    chunks_table: str = "dou.chunks",
    embed_fn=None,
) -> QueryResult:
    """
    Executa busca semântica + geração.

    model: formato LiteLLM ("gemini/...", "ollama/...", "groq/...", "mistral/...")
    chunks_table: tabela de chunks (dou.chunks, dou.chunks_labse, dou.chunks_bge)
    embed_fn: função query→list[float]; usa embed_query local se None
    filters: secao, orgao, tipo_ato, data_inicio, data_fim, fonte
    """
    import time

    t0 = time.perf_counter()

    # 1. Embedding da query
    _embed = embed_fn if embed_fn is not None else embed_query_local
    q_embedding = _embed(question)

    # 2. Retrieval no MotherDuck
    conn = get_connection()
    chunks = similarity_search(conn, q_embedding, top_k=top_k, filters=filters,
                               chunks_table=chunks_table)
    conn.close()

    if not chunks:
        return QueryResult(
            answer="Não encontrei atos relevantes para essa busca nos filtros selecionados.",
            source_chunks=[],
            latency_ms=(time.perf_counter() - t0) * 1000,
            model=model,
        )

    # 3. Montar contexto
    context_parts = []
    for c in chunks:
        data_str = c["data"].strftime("%d/%m/%Y") if isinstance(c["data"], date) else str(c["data"])
        context_parts.append(
            f"[{c['orgao']} - Seção {c['secao']} - {data_str}]\n"
            f"Título: {c['titulo']}\n"
            f"{c['chunk_texto']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)
    prompt = QA_TEMPLATE.format(context_str=context_str, query_str=question)

    # 4. Geração via LiteLLM (unifica Gemini + Ollama)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    kwargs = {"model": model, "messages": messages}
    if model.startswith("ollama/"):
        kwargs["api_base"] = "http://localhost:11434"

    response = completion(**kwargs)
    answer = response.choices[0].message.content

    latency_ms = (time.perf_counter() - t0) * 1000

    return QueryResult(answer=answer, source_chunks=chunks, latency_ms=latency_ms, model=model)
