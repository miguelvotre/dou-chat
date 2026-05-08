"""
Teste mínimo: valida conectividade com todos os LLMs do eval antes de rodar.

Candidatos: Gemini 2.5 Flash, Llama 3.3 70B, Qwen3 32B, Magistral Medium
Juiz:       Mistral Medium

Uso:
    python tests/test_llm_connectivity.py
"""
import os
import re
from dotenv import load_dotenv
load_dotenv()

import litellm
from rich.console import Console
from rich.table import Table

console = Console()

MODELS = [
    ("Gemini 2.5 Flash  [candidato]", "gemini/gemini-2.5-flash"),
    ("Llama 3.3 70B     [candidato]", "groq/llama-3.3-70b-versatile"),
    ("Magistral Medium  [candidato]", "mistral/magistral-medium-latest"),
    ("Qwen3 32B         [juiz]     ", "groq/qwen/qwen3-32b"),
]

if not os.getenv("GEMINI_API_KEY") and os.getenv("GOOGLE_AI_API_KEY"):
    os.environ["GEMINI_API_KEY"] = os.environ["GOOGLE_AI_API_KEY"]

PROMPT = "Responda em uma palavra: qual é a capital do Brasil?"

table = Table(title="Teste de conectividade — LLMs do eval")
table.add_column("Modelo", style="cyan")
table.add_column("Status", justify="center")
table.add_column("Resposta")

for label, model in MODELS:
    console.print(f"[dim]Testando {label.strip()}...[/dim]")
    kwargs = {"model": model, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 200}
    if model.startswith("ollama/"):
        kwargs["api_base"] = "http://localhost:11434"
    try:
        resp    = litellm.completion(**kwargs)
        content = (resp.choices[0].message.content or "").strip()
        # Strip thinking tags (Qwen3)
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        table.add_row(label, "[green]OK[/green]", content[:80])
    except Exception as e:
        table.add_row(label, "[red]ERRO[/red]", str(e)[:80])

console.print(table)
