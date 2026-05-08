"""
TruLens evaluation — embedding model comparison.

Compares e5, labse, bge-m3 using cosine and euclidean distance metrics
between the query and retrieved chunks. Results are stored in a local
SQLite database and can be inspected in the TruLens dashboard.

Usage:
    # Run evaluation
    python -m evaluation.trulens_eval

    # Run evaluation + open dashboard
    python -m evaluation.trulens_eval --dashboard
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from dotenv import load_dotenv

load_dotenv()

from trulens.core import Metric, TruSession
from trulens.apps.basic import TruBasicApp

from indexing.store import CHUNK_TABLES, get_connection, similarity_search
from evaluation.test_queries import EVAL_QUERIES

# ── Embedding models ──────────────────────────────────────────────────────────
EMBED_MODELS = {
    "e5":    "intfloat/multilingual-e5-large",
    "labse": "sentence-transformers/LaBSE",
    "bge":   "BAAI/bge-m3",
}

EMBED_PREFIX = {
    "e5":    ("query: ", "passage: "),
    "labse": ("", ""),
    "bge":   ("", ""),
}

TOP_K = 6
DB_PATH = "default.sqlite"  # TruLens singleton always writes here

# ── Lazy model cache ──────────────────────────────────────────────────────────
_models: dict = {}

def _get_model(embed_key: str):
    if embed_key not in _models:
        from sentence_transformers import SentenceTransformer
        _models[embed_key] = SentenceTransformer(EMBED_MODELS[embed_key])
    return _models[embed_key]


def _embed(text: str, embed_key: str, is_query: bool = True) -> np.ndarray:
    model = _get_model(embed_key)
    q_prefix, p_prefix = EMBED_PREFIX[embed_key]
    prefix = q_prefix if is_query else p_prefix
    return model.encode(f"{prefix}{text}", normalize_embeddings=True)


# ── Retrieval function factory ────────────────────────────────────────────────
def make_retrieval_fn(embed_key: str):
    """Returns a function: query str -> concatenated context str."""
    table, _ = CHUNK_TABLES[embed_key]

    def retrieve(query: str) -> str:
        q_emb = _embed(query, embed_key, is_query=True).tolist()
        conn = get_connection()
        chunks = similarity_search(conn, q_emb, top_k=TOP_K, chunks_table=table)
        conn.close()
        if not chunks:
            return ""
        parts = []
        for c in chunks:
            parts.append(
                f"[{c['orgao']} — Seção {c['secao']}]\n"
                f"Título: {c['titulo']}\n"
                f"{c['chunk_texto']}"
            )
        return "\n\n---\n\n".join(parts)

    return retrieve


# ── Feedback function factory ─────────────────────────────────────────────────
def make_feedback_fns(embed_key: str) -> list:
    """Returns cosine distance and euclidean distance feedback functions."""

    def cosine_distance(query: str, context: str) -> float:
        if not context.strip():
            return 1.0  # max distance = no retrieval
        q = _embed(query, embed_key, is_query=True)
        c = _embed(context[:512], embed_key, is_query=False)  # truncate for speed
        similarity = float(np.dot(q, c) / (np.linalg.norm(q) * np.linalg.norm(c) + 1e-9))
        return round(1.0 - similarity, 4)  # distance: lower = better

    def euclidean_distance(query: str, context: str) -> float:
        if not context.strip():
            return 2.0  # max distance for normalized vectors
        q = _embed(query, embed_key, is_query=True)
        c = _embed(context[:512], embed_key, is_query=False)
        return round(float(np.linalg.norm(q - c)), 4)

    f_cosine = Metric(implementation=cosine_distance, name="Cosine Distance").on_input_output()
    f_euclidean = Metric(implementation=euclidean_distance, name="Euclidean Distance").on_input_output()
    return [f_cosine, f_euclidean]


# ── Evaluation runner ─────────────────────────────────────────────────────────
def run_evaluation(embed_keys: list[str] | None = None) -> None:
    keys = embed_keys or list(EMBED_MODELS.keys())

    session = TruSession()
    session.reset_database()  # fresh run each time

    print(f"\nDOU Chat — Embedding Evaluation")
    print(f"Models  : {keys}")
    print(f"Queries : {len(EVAL_QUERIES)}")
    print(f"Top-k   : {TOP_K}")
    print(f"DB      : {DB_PATH}\n")

    for embed_key in keys:
        print(f"[{embed_key}] Loading model: {EMBED_MODELS[embed_key]}")
        retrieve = make_retrieval_fn(embed_key)
        feedbacks = make_feedback_fns(embed_key)

        tru_app = TruBasicApp(
            retrieve,
            app_name="DOU Retrieval",
            app_version=embed_key,
            feedbacks=feedbacks,
        )

        print(f"[{embed_key}] Running {len(EVAL_QUERIES)} queries...")
        for i, (query, _filters) in enumerate(EVAL_QUERIES, 1):
            with tru_app as recording:
                _ = tru_app.app(query)
            print(f"  [{i:02d}/{len(EVAL_QUERIES)}] {query[:70]}")

        print(f"[{embed_key}] Done.\n")

    print("─" * 60)
    leaderboard = session.get_leaderboard()
    print(leaderboard.to_string())
    print("─" * 60)
    print(f"\nResults saved to: {DB_PATH}")
    print("Run with --dashboard to open the TruLens UI.\n")


def launch_dashboard() -> None:
    from trulens.dashboard import run_dashboard
    session = TruSession()
    run_dashboard(session)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TruLens embedding evaluation for DOU Chat")
    parser.add_argument("--dashboard", action="store_true", help="Open TruLens dashboard after evaluation")
    parser.add_argument("--only-dashboard", action="store_true", help="Only open dashboard (skip evaluation)")
    parser.add_argument("--models", nargs="+", choices=list(EMBED_MODELS.keys()), help="Embedding models to evaluate")
    args = parser.parse_args()

    if args.only_dashboard:
        launch_dashboard()
    else:
        run_evaluation(embed_keys=args.models)
        if args.dashboard:
            launch_dashboard()
