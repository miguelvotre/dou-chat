"""
Embeddings com intfloat/multilingual-e5-large.

- Local (ingestão): sentence-transformers, sem API
- Produção (Streamlit): HF Inference API via HF_TOKEN

Controlado pela variável de ambiente EMBED_MODE:
  local (default) — sentence-transformers
  api             — HuggingFace Serverless Inference API
"""

import os

MODEL_NAME = "intfloat/multilingual-e5-large"
EMBED_MODE = os.getenv("EMBED_MODE", "local")

# --- Local ---
_st_model = None

def _get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer(MODEL_NAME)
    return _st_model


# --- HF API ---
_hf_client = None

def _get_hf_client():
    global _hf_client
    if _hf_client is None:
        from huggingface_hub import InferenceClient
        _hf_client = InferenceClient(token=os.environ["HF_TOKEN"])
    return _hf_client


# --- Interface pública ---

def embed_documents(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embeda chunks para indexação (sempre local)."""
    model = _get_st_model()
    prefixed = [f"passage: {t}" for t in texts]
    embeddings = model.encode(prefixed, batch_size=batch_size, normalize_embeddings=True)
    return embeddings.tolist()


def embed_query(text: str) -> list[float]:
    """Embeda query — local ou HF API dependendo de EMBED_MODE."""
    if EMBED_MODE == "api":
        client = _get_hf_client()
        result = client.feature_extraction(f"query: {text}", model=MODEL_NAME)
        return result.flatten().tolist()
    else:
        model = _get_st_model()
        embedding = model.encode(f"query: {text}", normalize_embeddings=True)
        return embedding.tolist()
