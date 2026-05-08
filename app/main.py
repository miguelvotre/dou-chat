"""
Main Streamlit app.

Flow:
  1. Sidebar filters
  2. Semantic search input
  3. Relevant acts list
  4. Follow-up chat over results
"""

import os
import sys
from pathlib import Path

# Ensure project root is on the path when running from a subdirectory (e.g. Streamlit Cloud)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import date, timedelta

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="DOU Chat",
    page_icon="📋",
    layout="wide",
)


def main():
    st.title("📋 DOU Chat")
    st.caption("Semantic search over Brazil's Diário Oficial da União")

    # --- Filters ---
    with st.sidebar:
        st.header("Filters")

        secao = st.selectbox(
            "Section",
            options=["All", "1", "2", "3"],
            help="1=Normative acts, 2=Personnel, 3=Contracts/Notices",
        )

        tipo_ato = st.selectbox(
            "Act type",
            options=["All", "portaria", "resolucao", "instrucao_normativa",
                     "decreto", "edital", "contrato", "extrato", "despacho",
                     "nomeacao", "exoneracao", "aviso", "outros"],
        )

        orgao = st.text_input("Issuing body (partial match)")

        col1, col2 = st.columns(2)
        data_inicio = col1.date_input(
            "From",
            value=date.today() - timedelta(days=90),
            max_value=date.today(),
        )
        data_fim = col2.date_input(
            "To",
            value=date.today(),
            max_value=date.today(),
        )

        top_k = st.slider("Number of acts", min_value=3, max_value=20, value=6)

    # Build filters
    filters: dict = {
        "data_inicio": data_inicio,
        "data_fim": data_fim,
    }
    if secao != "All":
        filters["secao"] = secao
    if tipo_ato != "All":
        filters["tipo_ato"] = tipo_ato
    if orgao.strip():
        filters["orgao"] = orgao.strip()

    # --- Search ---
    question = st.text_input(
        "What are you looking for?",
        placeholder="e.g. ordinances on temporary public sector hiring",
    )

    if not question:
        st.info("Enter a question or search term to get started.")
        return

    with st.spinner("Searching..."):
        from rag.engine import query as rag_query
        result = rag_query(question, filters=filters, top_k=top_k)

    # --- Answer ---
    st.markdown("### Answer")
    st.markdown(result.answer)
    st.caption(f"Latency: {result.latency_ms:.0f}ms")

    # --- Retrieved acts ---
    if result.source_chunks:
        st.markdown("---")
        st.markdown(f"### Relevant acts ({len(result.source_chunks)})")

        seen_atos = {}
        for chunk in result.source_chunks:
            ato_id = chunk["ato_id"]
            if ato_id in seen_atos:
                continue
            seen_atos[ato_id] = chunk

        for chunk in seen_atos.values():
            data_str = (
                chunk["data"].strftime("%d/%m/%Y")
                if isinstance(chunk["data"], date)
                else str(chunk["data"])
            )
            score_pct = int(chunk["score"] * 100)

            with st.expander(
                f"[Section {chunk['secao']}] {chunk['titulo'] or chunk['orgao']} — {data_str} ({score_pct}% relevant)"
            ):
                st.write(f"**Issuing body:** {chunk['orgao']}")
                st.write(f"**Type:** {chunk['tipo_ato']}")
                if chunk.get("subtitulo"):
                    st.write(f"**Subtitle:** {chunk['subtitulo']}")
                st.markdown("**Excerpt:**")
                st.write(chunk["chunk_texto"])
                if chunk.get("url_original"):
                    st.markdown(f"[View in DOU]({chunk['url_original']})")

    # --- Follow-up chat ---
    if result.source_chunks:
        st.markdown("---")
        st.markdown("### Follow-up questions")

        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        for msg in st.session_state["chat_history"]:
            role = "user" if msg["role"] == "user" else "assistant"
            st.chat_message(role).write(msg["content"])

        follow_up = st.chat_input("Ask a question about the retrieved acts...")
        if follow_up:
            st.session_state["chat_history"].append({"role": "user", "content": follow_up})
            st.chat_message("user").write(follow_up)

            with st.spinner("Generating..."):
                follow_result = rag_query(follow_up, filters=filters, top_k=top_k)

            st.chat_message("assistant").write(follow_result.answer)
            st.session_state["chat_history"].append(
                {"role": "assistant", "content": follow_result.answer}
            )


if __name__ == "__main__":
    main()
