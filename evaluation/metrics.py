"""
Avaliação comparativa de LLMs com tríade RAG.

Juiz chama o modelo diretamente via LiteLLM para pontuar cada métrica.

Troca entre dry-run (Ollama) e full (Gemini) alterando DRY_RUN.

Uso:
    python -m evaluation.metrics
"""

from statistics import mean

from dotenv import load_dotenv
from litellm import completion
from rich.console import Console
from rich.table import Table

load_dotenv()
console = Console()

# --- Configuração ---
DRY_RUN = True  # False = usa Gemini como juiz (cota real)

JUDGE_MODEL = "ollama/phi4:14b" if DRY_RUN else "gemini/gemini-2.5-flash-lite"
JUDGE_API_BASE = "http://localhost:11434" if DRY_RUN else None

MODELS = [
    ("gemma2:9b",  "ollama/gemma2:9b"),
    ("qwen2.5:7b", "ollama/qwen2.5:7b"),
    ("phi4:14b",   "ollama/phi4:14b"),
] if DRY_RUN else [
    ("gemini-2.5-flash-lite", "gemini/gemini-2.5-flash-lite"),
    ("gemma2:9b",             "ollama/gemma2:9b"),
    ("qwen2.5:7b",            "ollama/qwen2.5:7b"),
    ("phi4:14b",              "ollama/phi4:14b"),
]

EVAL_QUERIES = [
    "Decisões do Supremo Tribunal Federal sobre inconstitucionalidade de lei estadual",
    "Atos publicados pela Agência Nacional de Energia Elétrica",
    "Portarias do Ministério das Comunicações",
    "Portarias do Ministério da Educação sobre regulação do ensino superior",
    "Despachos do Ministério da Fazenda",
]

ACTIVE_QUERIES = EVAL_QUERIES[:2] if DRY_RUN else EVAL_QUERIES


# --- Juiz ---

def _judge_call(prompt: str) -> float:
    """Chama o juiz e extrai score 0-10 da resposta."""
    kwargs = {"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}]}
    if JUDGE_API_BASE:
        kwargs["api_base"] = JUDGE_API_BASE
    response = completion(**kwargs)
    text = response.choices[0].message.content.strip()
    # extrai primeiro número da resposta
    import re
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    for n in nums:
        val = float(n)
        if 0 <= val <= 10:
            return val / 10.0
    return 0.5


def score_answer_relevance(question: str, answer: str) -> float:
    prompt = f"""Você é um avaliador de sistemas RAG jurídicos.

Pergunta: {question}
Resposta: {answer}

A resposta responde diretamente à pergunta? Pontue de 0 a 10 (apenas o número)."""
    return _judge_call(prompt)


def score_context_relevance(question: str, chunks: list[str]) -> float:
    context = "\n---\n".join(chunks[:3])
    prompt = f"""Você é um avaliador de sistemas RAG jurídicos.

Pergunta: {question}
Trechos recuperados:
{context}

Os trechos são relevantes para responder à pergunta? Pontue de 0 a 10 (apenas o número)."""
    return _judge_call(prompt)


def score_groundedness(answer: str, chunks: list[str]) -> float:
    context = "\n---\n".join(chunks[:3])
    prompt = f"""Você é um avaliador de sistemas RAG jurídicos.

Contexto (DOU):
{context}

Resposta gerada:
{answer}

A resposta está fundamentada apenas no contexto fornecido, sem alucinações? Pontue de 0 a 10 (apenas o número)."""
    return _judge_call(prompt)


# --- Avaliação ---

def run_evaluation() -> None:
    from rag.engine import query as rag_query

    console.print(f"\n[bold]Modo:[/bold] {'DRY-RUN (Ollama)' if DRY_RUN else 'FULL (Gemini judge)'}")
    console.print(f"[bold]Juiz:[/bold] {JUDGE_MODEL}")
    console.print(f"[bold]Modelos:[/bold] {[n for n, _ in MODELS]}")
    console.print(f"[bold]Queries:[/bold] {len(ACTIVE_QUERIES)}\n")

    all_results = {}

    for display_name, model_id in MODELS:
        console.print(f"[cyan]Avaliando: {display_name}[/cyan]")
        scores = {"answer": [], "context": [], "ground": [], "latency": []}

        for question in ACTIVE_QUERIES:
            console.print(f"  [dim]{question[:65]}...[/dim]")
            result = rag_query(question, model=model_id)
            chunks = [c["chunk_texto"] for c in result.source_chunks]

            scores["answer"].append(score_answer_relevance(question, result.answer))
            scores["context"].append(score_context_relevance(question, chunks))
            scores["ground"].append(score_groundedness(result.answer, chunks))
            scores["latency"].append(result.latency_ms)

            console.print(f"    latência: {result.latency_ms:.0f}ms")

        all_results[display_name] = {k: mean(v) for k, v in scores.items()}

    _print_results(all_results)


def _print_results(results: dict) -> None:
    console.print("\n")
    table = Table(title=f"Comparação de Modelos — Juiz: {JUDGE_MODEL}")
    table.add_column("Modelo", style="cyan")
    table.add_column("Answer Rel.", justify="right")
    table.add_column("Context Rel.", justify="right")
    table.add_column("Groundedness", justify="right")
    table.add_column("Latência média", justify="right")

    for name, s in results.items():
        table.add_row(
            name,
            f"{s['answer']:.2f}",
            f"{s['context']:.2f}",
            f"{s['ground']:.2f}",
            f"{s['latency']:.0f}ms",
        )

    console.print(table)


if __name__ == "__main__":
    run_evaluation()
