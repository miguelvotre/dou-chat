"""
Avaliação TruLens: LLMs com embedding e5 fixo — RAG Triad.

Métricas (via juiz LLM):
  - Answer Relevance   : resposta responde à pergunta?
  - Context Relevance  : chunks recuperados são relevantes?
  - Groundedness       : resposta está fundamentada no contexto?

Modos:
  --mode local   phi4:14b (Ollama) como candidato e juiz — dry run
  --mode api     Gemini 2.5 Flash, Llama 3.3 70B, Magistral Medium
                 Juiz: Qwen3 32B com thinking (Groq)

Uso:
    python run_trulens_eval.py --mode local
    python run_trulens_eval.py --mode local --queries 2
    python run_trulens_eval.py --mode api
    python run_trulens_eval.py --mode api --queries 5

Retomada: se o limite de API acabar durante a Fase 2, basta rodar novamente.
O script detecta o checkpoint e retoma de onde parou.
"""
import os
import re
import json
import warnings
import logging

# Desliga OTEL antes de qualquer import do TruLens
os.environ["TRULENS_OTEL_TRACING"] = "0"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message="Failed to compute cost")
logging.getLogger("trulens").setLevel(logging.ERROR)
logging.getLogger("trulens_eval").setLevel(logging.ERROR)

from dotenv import load_dotenv
load_dotenv()

import argparse
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from datetime import datetime

import litellm
import time
import threading

_locks: dict[str, threading.Lock] = {
    "mistral": threading.Lock(),
    "groq":    threading.Lock(),
    "gemini":  threading.Lock(),
}
_last_call: dict[str, float] = {"mistral": 0.0, "groq": 0.0, "gemini": 0.0}
_GROQ_INTERVAL    = 65.0  # ~1 call/min — limitado pelo TPM do Qwen3 32B thinking (6k TPM)
_MISTRAL_INTERVAL = 2.0   # 60 req/min (Mistral free tier — mistral-large-2411)
_GEMINI_INTERVAL  = 8.0   # ~7 RPM seguro (Gemini free tier: 10 RPM)

_original_completion = litellm.completion


def _paced_completion(*args, **kwargs):
    """Serializa e espaça chamadas a providers com rate limit apertado."""
    model = str(kwargs.get("model", args[0] if args else ""))
    provider = None
    interval = _GROQ_INTERVAL
    if "mistral" in model.lower():
        provider, interval = "mistral", _MISTRAL_INTERVAL
    elif "groq" in model.lower():
        provider = "groq"
    elif "gemini" in model.lower():
        provider, interval = "gemini", _GEMINI_INTERVAL

    if provider:
        with _locks[provider]:
            elapsed = time.time() - _last_call[provider]
            wait = interval - elapsed
            if wait > 0:
                time.sleep(wait)
            _last_call[provider] = time.time()

    return _original_completion(*args, **kwargs)


litellm.completion = _paced_completion
from litellm import completion as litellm_completion
from huggingface_hub import InferenceClient
from rich.console import Console

litellm.num_retries = 6
litellm.retry_after = 10

from trulens.core import TruSession
from trulens.apps.app import instrument
from trulens.apps.custom import TruCustomApp

from indexing.store import get_connection, similarity_search
from rag.prompts import QA_TEMPLATE, SYSTEM_PROMPT

if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_AI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_AI_API_KEY"]

console = Console()

# ── Configurações ─────────────────────────────────────────────────────────────

# Embedding fixo: e5 (vencedor do embedding eval)
EMBED_E5 = ("intfloat/multilingual-e5-large", "dou.chunks_e5", "query: ")

LLM_LOCAL = [
    ("phi4_14b", "ollama/phi4:14b"),
]

LLM_API = [
    ("gemini_flash",     "gemini/gemini-2.5-flash"),
    ("llama33_70b",      "groq/llama-3.3-70b-versatile"),
    ("magistral_medium", "mistral/magistral-medium-latest"),
]

JUDGE_LOCAL = ("ollama/phi4:14b", {"api_base": "http://localhost:11434"})
JUDGE_API   = ("groq/qwen/qwen3-32b", {"max_tokens": 1024})

EVAL_QUERIES = [
    "Quais são as portarias recentes sobre contratação de pessoal no setor público?",
    "Atos recentes do Ministério da Fazenda sobre política fiscal",
    "Nomeações e exonerações em cargos de confiança do Poder Executivo",
    "regulamentação de proteção de dados pessoais no serviço público",
    "licitação dispensada por valor abaixo do limite legal",
]

TOP_K = 6

# ── HF Embedder ───────────────────────────────────────────────────────────────

_hf: InferenceClient | None = None


def _get_hf() -> InferenceClient:
    global _hf
    if _hf is None:
        _hf = InferenceClient(token=os.environ["HF_TOKEN"])
    return _hf


def embed_query_hf(text: str, model_id: str, prefix: str) -> list[float]:
    result = _get_hf().feature_extraction(f"{prefix}{text}", model=model_id)
    return np.array(result).flatten().tolist()


# ── RAG App (instrumentada pelo TruLens) ──────────────────────────────────────

class RAGApp:
    """
    Combina embedding e5 fixo com um modelo gerador.
    Métodos instrumentados pelo TruLens para rastreamento.
    """

    def __init__(self, llm_model: str, llm_kwargs: dict):
        embed_model_id, chunks_table, embed_prefix = EMBED_E5
        self.embed_model_id = embed_model_id
        self.embed_prefix   = embed_prefix
        self.chunks_table   = chunks_table
        self.llm_model      = llm_model
        self.llm_kwargs     = llm_kwargs
        self._conn          = get_connection()
        self._last_context: list[str] = []
        self._last_answer:  str       = ""

    @instrument
    def retrieve(self, query: str) -> list[str]:
        q_emb  = embed_query_hf(query, self.embed_model_id, self.embed_prefix)
        chunks = similarity_search(self._conn, q_emb, top_k=TOP_K, chunks_table=self.chunks_table)
        self._last_context = [c["chunk_texto"] for c in chunks]
        return self._last_context

    @instrument
    def generate(self, query: str, context: list[str]) -> str:
        context_str = "\n\n---\n\n".join(context)
        prompt      = QA_TEMPLATE.format(context_str=context_str, query_str=query)
        messages    = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ]
        kwargs = {"model": self.llm_model, "messages": messages, "max_tokens": 512,
                  **self.llm_kwargs}
        resp   = litellm_completion(**kwargs)
        answer = (resp.choices[0].message.content or "").strip()
        # Remove thinking tags do Qwen3 (modo thinking habilitado por padrão no Groq)
        answer = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        self._last_answer = answer
        return self._last_answer

    @instrument
    def query(self, question: str) -> str:
        context = self.retrieve(question)
        return self.generate(question, context)

    def close(self):
        self._conn.close()


# ── Juiz (rate-limited) ───────────────────────────────────────────────────────

def _judge(prompt: str, judge_model: str, judge_kwargs: dict) -> float:
    """Chama o juiz e extrai score 0-1. Rate limiting via _paced_completion.
    Em caso de 429 TPM, aguarda 65s e tenta mais uma vez antes de desistir."""
    kwargs = {"model": judge_model, "messages": [{"role": "user", "content": prompt}],
              "max_tokens": 50, "num_retries": 0, **judge_kwargs}
    for attempt in range(2):
        try:
            resp = _paced_completion(**kwargs)
            text = (resp.choices[0].message.content or "").strip()
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
            nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
            for n in nums:
                val = float(n)
                if 0 <= val <= 10:
                    return val / 10.0
            return 0.5
        except litellm.RateLimitError:
            if attempt == 0:
                console.print("  [yellow]429 TPM — aguardando 65s para reset...[/yellow]")
                time.sleep(65)
            else:
                console.print("  [red]429 persistente — score padrão 0.5[/red]")
                return 0.5


def evaluate_record(question: str, answer: str, context: list[str],
                    judge_model: str, judge_kwargs: dict) -> dict[str, float]:
    """Avalia uma resposta nas 3 métricas RAG Triad."""
    ar = _judge(
        f"Avaliador RAG jurídico.\nPergunta: {question}\nResposta: {answer}\n"
        "A resposta responde à pergunta? Responda APENAS com um número de 0 a 10.",
        judge_model, judge_kwargs,
    )
    cr = _judge(
        f"Avaliador RAG jurídico.\nPergunta: {question}\nContexto:\n{'---'.join(context[:3])}\n"
        "O contexto é relevante para a pergunta? Responda APENAS com um número de 0 a 10.",
        judge_model, judge_kwargs,
    )
    gr = _judge(
        f"Avaliador RAG jurídico.\nContexto:\n{'---'.join(context[:3])}\nResposta: {answer}\n"
        "A resposta está fundamentada no contexto? Responda APENAS com um número de 0 a 10.",
        judge_model, judge_kwargs,
    )
    return {"answer": ar, "context": cr, "ground": gr}


# ── Checkpoint ────────────────────────────────────────────────────────────────

def _checkpoint_path(mode: str) -> str:
    os.makedirs("results", exist_ok=True)
    return f"results/checkpoint_{mode}.json"


def _load_checkpoint(mode: str) -> dict | None:
    path = _checkpoint_path(mode)
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        console.print(f"[yellow]Checkpoint carregado: {path}[/yellow]")
        return data
    return None


def _save_checkpoint(mode: str, payload: dict) -> None:
    with open(_checkpoint_path(mode), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


# ── Execução ──────────────────────────────────────────────────────────────────

def run(mode: str, n_queries: int) -> None:
    llms                      = LLM_LOCAL if mode == "local" else LLM_API
    judge_model, judge_kwargs = JUDGE_LOCAL if mode == "local" else JUDGE_API
    queries                   = EVAL_QUERIES[:n_queries]

    console.print(f"\n[bold]Modo:[/bold] {mode.upper()}")
    console.print(f"[bold]Juiz:[/bold] {judge_model}")
    console.print(f"[bold]Queries:[/bold] {len(queries)}")
    console.print(f"[bold]LLMs:[/bold] {len(llms)}\n")

    checkpoint = _load_checkpoint(mode)
    all_scores: dict[str, dict] = {}

    if checkpoint and checkpoint.get("phase1_complete"):
        console.print("[yellow]Fase 1 já concluída — retomando Fase 2[/yellow]\n")
        all_scores = checkpoint["all_scores"]
    else:
        # ── Fase 1: Geração ──────────────────────────────────────────────────
        console.rule("[bold]Fase 1: Geração[/bold]")
        TruSession()

        for n, (llm_name, llm_model) in enumerate(llms, 1):
            app_id     = f"e5__{llm_name}__{mode}"
            llm_kwargs = {"api_base": "http://localhost:11434"} if llm_model.startswith("ollama/") else {}
            console.print(f"[cyan][{n}/{len(llms)}] {app_id}[/cyan]")

            rag     = RAGApp(llm_model=llm_model, llm_kwargs=llm_kwargs)
            tru_app = TruCustomApp(rag, app_name=app_id, app_version="1", feedback_mode="none")

            records      = []
            combo_failed = False
            with tru_app:
                for qi, question in enumerate(queries):
                    console.print(f"  [dim]Q{qi+1}: {question[:70]}[/dim]")
                    try:
                        rag.query(question)
                        records.append({
                            "question": question,
                            "answer":   rag._last_answer,
                            "context":  list(rag._last_context),
                        })
                    except Exception as e:
                        err_str = str(e)
                        if "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower():
                            console.print(f"  [yellow]429/quota — pulando {llm_name}[/yellow]")
                            combo_failed = True
                            break
                        else:
                            console.print(f"  [red]Erro Q{qi+1}: {err_str[:120]}[/red]")
                            records.append({"question": question, "answer": "", "context": []})

            rag.close()
            if not combo_failed:
                all_scores[llm_name] = {"records": records}

        _save_checkpoint(mode, {
            "mode": mode, "judge": judge_model,
            "phase1_complete": True, "all_scores": all_scores,
        })
        console.print("[green]Checkpoint Fase 1 salvo.[/green]\n")

    # ── Fase 2: Avaliação com juiz ───────────────────────────────────────────
    console.rule("[bold]Fase 2: Avaliação com juiz[/bold]")
    results: dict[str, dict] = {}

    for llm_name, data in all_scores.items():
        console.print(f"[cyan]{llm_name}[/cyan]")
        combo_scores: dict[str, list] = {"answer": [], "context": [], "ground": []}

        for rec in data["records"]:
            if "scores" in rec:
                console.print(f"  [dim]Já avaliado: {rec['question'][:60]}[/dim]")
                s = rec["scores"]
            else:
                console.print(f"  [dim]Avaliando: {rec['question'][:60]}[/dim]")
                s = evaluate_record(rec["question"], rec["answer"], rec["context"],
                                    judge_model, judge_kwargs)
                rec["scores"] = s
                _save_checkpoint(mode, {
                    "mode": mode, "judge": judge_model,
                    "phase1_complete": True, "all_scores": all_scores,
                })

            combo_scores["answer"].append(s["answer"])
            combo_scores["context"].append(s["context"])
            combo_scores["ground"].append(s["ground"])
            console.print(f"    AR={s['answer']:.2f} CR={s['context']:.2f} GR={s['ground']:.2f}")

        if combo_scores["answer"]:
            results[llm_name] = {k: sum(v) / len(v) for k, v in combo_scores.items()}

    _print_leaderboard(results, mode, judge_model)
    _save_chart(results, mode, judge_model, llms)

    os.makedirs("results", exist_ok=True)
    ts        = datetime.now().strftime("%Y%m%d_%H%M")
    json_path = f"results/eval_{mode}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"mode": mode, "judge": judge_model, "queries": n_queries, "results": results},
            f, indent=2, ensure_ascii=False,
        )
    console.print(f"[green]JSON: {json_path}[/green]")
    console.print("Dashboard TruLens (traces): [bold]trulens-eval[/bold]")


def _print_leaderboard(results: dict, mode: str, judge: str) -> None:
    from rich.table import Table
    table = Table(title=f"Avaliação RAG — {mode.upper()} — Juiz: {judge}")
    table.add_column("LLM", style="cyan")
    table.add_column("Answer Rel.", justify="right")
    table.add_column("Context Rel.", justify="right")
    table.add_column("Groundedness", justify="right")
    table.add_column("Média", justify="right", style="bold")
    for llm_name, s in sorted(results.items(), key=lambda x: -sum(x[1].values())):
        avg = sum(s.values()) / 3
        table.add_row(llm_name,
                      f"{s['answer']:.2f}", f"{s['context']:.2f}", f"{s['ground']:.2f}",
                      f"{avg:.2f}")
    console.print(table)


def _save_chart(results: dict, mode: str, judge: str, llms: list) -> None:
    os.makedirs("results", exist_ok=True)
    llm_names = [l[0] for l in llms if l[0] in results]
    metrics   = [("answer", "Answer Relevance"), ("context", "Context Relevance"), ("ground", "Groundedness")]
    colors    = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=True)
    fig.suptitle(f"TruLens RAG Evaluation — {mode.upper()}\nJuiz: {judge}",
                 fontsize=13, fontweight="bold", y=1.03)

    x     = np.arange(len(llm_names))
    width = 0.5

    for ax, (metric_key, metric_label) in zip(axes, metrics):
        vals = [results.get(ln, {}).get(metric_key, 0.0) for ln in llm_names]
        bars = ax.bar(x, vals, width, color=colors[:len(llm_names)], alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_title(metric_label, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(llm_names, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.grid(axis="y", alpha=0.3)
        if ax is axes[0]:
            ax.set_ylabel("Score (0–1)")

    plt.tight_layout()
    ts   = datetime.now().strftime("%Y%m%d_%H%M")
    path = f"results/eval_{mode}_{ts}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    console.print(f"[green]Chart salvo: {path}[/green]")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["local", "api"], default="local")
    parser.add_argument("--queries", type=int, default=None,
                        help="Número de queries (default: 2 local, 5 api)")
    args = parser.parse_args()

    n = args.queries or (2 if args.mode == "local" else 5)
    run(args.mode, n)
