"""
DOU Chat — Streamlit

Usage:
    streamlit run app.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from datetime import date

st.set_page_config(page_title="DOU Chat", page_icon="📋", layout="wide", initial_sidebar_state="expanded")

from indexing.store import CHUNK_TABLES, get_connection, similarity_search
from rag.prompts import QA_TEMPLATE, SYSTEM_PROMPT
from litellm import completion

# ── Fixed config ──────────────────────────────────────────────────────────────
LLM_MODEL    = "mistral/magistral-medium-latest"
LLM_API_BASE = "http://localhost:11434"  # used only for ollama models

# Fallback chain ordered by TruLens eval results (2026-05-08)
# Magistral > Llama 3.3 > Gemini Flash | Qwen3 not evaluated (used as judge)
API_FALLBACK_CHAIN: list[tuple[str, str]] = [
    ("groq/llama-3.3-70b-versatile",   "Llama 3.3 70B"),
    ("gemini/gemini-2.5-flash",        "Gemini 2.5 Flash"),
    ("groq/qwen/qwen3-32b",            "Qwen3 32B"),
]

EMBED_MODEL_ID = "intfloat/multilingual-e5-large"
EMBED_PREFIX   = "query: "
CHUNKS_TABLE   = CHUNK_TABLES["e5"][0]

SECTIONS    = ["(all)",   "DO1", "DO2", "DO3", "DO1E", "DO2E", "DO3E"]
SECTIONS_PT = ["(todas)", "DO1", "DO2", "DO3", "DO1E", "DO2E", "DO3E"]

# ── Portfolio ─────────────────────────────────────────────────────────────────
PORTFOLIO = [
    {
        "title": "DOU Chat",
        "title_pt": "DOU Chat",
        "desc": "AI-powered semantic search and Q&A over Brazil's Official Gazette using RAG.",
        "desc_pt": "Busca semântica e Q&A sobre o Diário Oficial da União com RAG.",
        "tags": ["RAG", "LLM", "DuckDB", "Streamlit", "MotherDuck", "TruLens", "LLM Observability"],
        "url_live": "https://dou-chat-llm.streamlit.app",
        "url_github": "https://github.com/miguelvotre/dou-chat",
        "status": "live",
        "current": True,
    },
    {
        "title": "Olist Analytics",
        "title_pt": "Olist Analytics",
        "desc": "E-commerce insights dashboard for the Brazilian Olist dataset (2017–2018). dbt + DuckDB + Streamlit.",
        "desc_pt": "Dashboard de e-commerce sobre o dataset Olist (2017–2018). dbt + DuckDB + Streamlit.",
        "tags": ["DuckDB", "dbt", "Streamlit", "LLM", "MotherDuck", "Airflow", "Text-to-SQL"],
        "url_live": "https://olist-ecommerce-analytics-dashboard.streamlit.app",
        "url_github": "https://github.com/miguelvotre/olist-ecommerce-analytics",
        "status": "live",
    },
    {
        "title": "Padel IA Analytics",
        "title_pt": "Padel IA Analytics",
        "desc": "AI-powered padel match statistics extracted from video using computer vision.",
        "desc_pt": "Estatísticas de partidas de padel extraídas de vídeo com visão computacional.",
        "tags": ["Computer Vision", "AI", "Sports Analytics", "Python"],
        "url_live": None,
        "url_github": None,
        "status": "soon",
    },
]

# ── i18n ──────────────────────────────────────────────────────────────────────
T = {
    "en": {
        "page_caption":     "Legal assistant for Brazil's Official Gazette (Diário Oficial da União)",
        "settings":         "Settings",
        "retrieved_chunks": "Retrieved excerpts (top-k)",
        "filters":          "Optional filters",
        "section":          "Section",
        "date_from":        "From",
        "date_to":          "To",
        "show_context":     "Show retrieved excerpts",
        "clear_chat":       "Clear conversation",
        "chat_placeholder": "Ask about the Official Gazette...",
        "searching":        "Searching and generating answer...",
        "no_results":       "No relevant acts found for this query with the selected filters.",
        "context_label":    "Retrieved excerpts ({n})",
        "score_label":      "score",
        "suggested":        "Try a question:",
        "suggestions": [
            "Supreme Court decisions on unconstitutionality of state laws",
            "Acts published by the National Electric Energy Agency (ANEEL)",
            "Ministry of Education ordinances on higher education regulation",
        ],
        "error_retrieval":  "Retrieval error: ",
        "error_generation": "Generation error: ",
        "sections_list":    SECTIONS,
        "my_projects":      "My Projects",
        "proj_live":        "Live",
        "proj_soon":        "Soon",
        "proj_view":        "View project →",
        "proj_github":      "GitHub →",
        "portfolio":        "Portfolio Projects",
        "sidebar_hide":     "Hide Sidebar",
        "sidebar_show":     "Show Sidebar",
    },
    "pt": {
        "page_caption":     "Assistente jurídico para o Diário Oficial da União",
        "settings":         "Configurações",
        "retrieved_chunks": "Trechos recuperados (top-k)",
        "filters":          "Filtros opcionais",
        "section":          "Seção",
        "date_from":        "De",
        "date_to":          "Até",
        "show_context":     "Mostrar trechos recuperados",
        "clear_chat":       "Limpar conversa",
        "chat_placeholder": "Pergunte sobre o Diário Oficial...",
        "searching":        "Buscando e gerando resposta...",
        "no_results":       "Não encontrei atos relevantes para essa busca nos filtros selecionados.",
        "context_label":    "Trechos recuperados ({n})",
        "score_label":      "score",
        "suggested":        "Experimente uma pergunta:",
        "suggestions": [
            "Decisões do Supremo Tribunal Federal sobre inconstitucionalidade de lei estadual",
            "Atos publicados pela Agência Nacional de Energia Elétrica",
            "Portarias do Ministério da Educação sobre regulação do ensino superior",
        ],
        "error_retrieval":  "Erro no retrieval: ",
        "error_generation": "Erro na geração: ",
        "sections_list":    SECTIONS_PT,
        "my_projects":      "Meus Projetos",
        "proj_live":        "Ativo",
        "proj_soon":        "Em breve",
        "proj_view":        "Ver projeto →",
        "proj_github":      "GitHub →",
        "portfolio":        "Portfólio",
        "sidebar_hide":     "Ocultar barra",
        "sidebar_show":     "Mostrar barra",
    },
}

def t(key: str) -> str:
    return T[st.session_state.get("lang", "en")][key]

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "lang" not in st.session_state:
    st.session_state.lang = "en"
if "theme" not in st.session_state:
    st.session_state.theme = "light"

# ── Theme palettes ────────────────────────────────────────────────────────────
THEMES: dict[str, dict] = {
    "light": dict(
        bg="#ffffff",      bg2="#f0f2f6",  bg3="#e8eaf0",
        sbg="#f0f2f6",     sbg2="#e4e6ee",
        text="#0e1117",    text2="#555",   text_inv="#ffffff",
        border="#d4d6de",  input_bg="#ffffff",
        btn_bg="#ffffff",  btn_hover="#e8eaef",
        msg_asst="#f4f6fb", msg_user="#e8f0fe",
        badge_live="#1a9e5c", badge_soon="#999",
        shadow="rgba(0,0,0,0.07)",
    ),
    "dark": dict(
        bg="#0e1117",      bg2="#1c1f2b",  bg3="#252836",
        sbg="#161923",     sbg2="#1c1f2b",
        text="#e8eaf0",    text2="#9ea3b0", text_inv="#0e1117",
        border="#2e3244",  input_bg="#1c1f2b",
        btn_bg="#252836",  btn_hover="#2e3244",
        msg_asst="#1c1f2b", msg_user="#1a2540",
        badge_live="#2ecc71", badge_soon="#777",
        shadow="rgba(0,0,0,0.35)",
    ),
}

# ── CSS injection (runs every render) ────────────────────────────────────────
def _inject_css() -> None:
    p = THEMES[st.session_state.theme]
    st.markdown(f"""<style>
[data-testid="stToolbar"],[data-testid="stDecoration"]{{display:none!important}}

/* sidebar: always visible, no collapse controls */
[data-testid="stSidebarCollapsedControl"],[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"]{{display:none!important}}
[data-testid="stSidebar"]{{transform:none!important;margin-left:0!important;visibility:visible!important}}
/* sidebar compact layout */
[data-testid="stSidebarContent"]{{padding-top:1.25rem!important}}
section[data-testid="stSidebar"]>div{{padding-top:1rem!important}}
[data-testid="stSidebar"] [data-testid="stVerticalBlock"]{{gap:0.75rem!important}}
[data-testid="stSidebar"] hr{{margin:0.5rem 0!important}}
[data-testid="stSidebar"] h1{{margin-bottom:0.5rem!important;padding-bottom:0!important}}
[data-testid="stSidebar"] [data-testid="stExpander"]{{border:none!important;box-shadow:none!important;margin-top:0.4rem!important}}
[data-testid="stSidebar"] [data-testid="stExpander"] details summary p{{font-size:0.72rem!important}}

/* backgrounds + base text */
.stApp,[data-testid="stAppViewContainer"],[data-testid="stMain"],[data-testid="stHeader"]{{background-color:{p['bg']}!important;color:{p['text']}!important}}
[data-testid="stSidebar"]>div:first-child{{background-color:{p['sbg']}!important}}

/* text — targeted, not broad divs */
.stApp p,.stApp h1,.stApp h2,.stApp h3,.stApp h4,
.stApp li, .stApp td,
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3,
[data-testid="stMarkdownContainer"] li,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] span[class],
[data-testid="stSidebar"] label,
[data-testid="stWidgetLabel"] p,
[data-testid="stSlider"] [data-testid="stWidgetLabel"] p,
.stSlider span, .stSelectbox label, .stDateInput label,
.stToggle label, .stCheckbox label,
[data-testid="stCaptionContainer"] p
{{color:{p['text']}!important}}

.stCaption p, [data-testid="stCaptionContainer"] p {{color:{p['text2']}!important}}

/* buttons */
.stButton>button{{background-color:{p['btn_bg']}!important;color:{p['text']}!important;border:1px solid {p['border']}!important;font-size:13px!important}}
.stButton>button:hover{{background-color:{p['btn_hover']}!important}}

/* chat messages */
[data-testid="stChatMessage"]{{background-color:{p['msg_asst']}!important;border:1px solid {p['border']}!important}}
[data-testid="stChatMessage"] p{{color:{p['text']}!important}}

/* chat input bar */
[data-testid="stChatInputContainer"],[data-testid="stBottomBlockContainer"]{{background-color:{p['bg']}!important;border-top:1px solid {p['border']}!important}}
[data-testid="stChatInput"]>div{{background-color:{p['input_bg']}!important;border-color:{p['border']}!important}}
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputContainer"] textarea{{
    background-color:{p['input_bg']}!important;
    color:{p['text']}!important;
    -webkit-text-fill-color:{p['text']}!important;
}}
[data-testid="stChatInput"] textarea::placeholder{{color:{p['text2']}!important;-webkit-text-fill-color:{p['text2']}!important;opacity:0.6!important}}

/* inputs (text, date) */
[data-baseweb="input"]>div{{background-color:{p['input_bg']}!important;border-color:{p['border']}!important}}
[data-baseweb="input"] input,[data-baseweb="base-input"] input{{
    color:{p['text']}!important;-webkit-text-fill-color:{p['text']}!important;
    background-color:{p['input_bg']}!important;
}}
[data-baseweb="input"] input::placeholder{{color:{p['text2']}!important;-webkit-text-fill-color:{p['text2']}!important;opacity:0.6!important}}

/* select */
[data-baseweb="select"]>div:first-child{{background-color:{p['input_bg']}!important;border-color:{p['border']}!important;color:{p['text']}!important}}
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p{{color:{p['text']}!important}}

/* dropdown menus */
[data-baseweb="popover"]>[data-baseweb="block"]{{background-color:{p['bg2']}!important;border-color:{p['border']}!important}}
[data-baseweb="menu"] ul{{background-color:{p['bg2']}!important}}
[data-baseweb="option"]{{background-color:{p['bg2']}!important;color:{p['text']}!important}}
[data-baseweb="option"]:hover{{background-color:{p['bg3']}!important}}

/* calendar (date picker) — aggressive dark mode */
[data-baseweb="calendar"]{{background-color:{p['bg2']}!important;border:1px solid {p['border']}!important}}
[data-baseweb="calendar"] [role="grid"],[data-baseweb="calendar"] [role="row"],
[data-baseweb="calendar"] [role="gridcell"],[data-baseweb="calendar"] td,
[data-baseweb="calendar"] th,[data-baseweb="calendar"] tr{{background-color:{p['bg2']}!important}}
[data-baseweb="calendar"] *{{color:{p['text']}!important}}
[data-baseweb="calendar"] button{{background-color:transparent!important}}
[data-baseweb="calendar"] [aria-selected="true"],
[data-baseweb="calendar"] button[aria-selected="true"]{{
    background-color:#4a86c8!important;border-radius:50%!important;
}}
[data-baseweb="calendar"] [aria-selected="true"] *,
[data-baseweb="calendar"] button[aria-selected="true"] *{{
    color:#ffffff!important;background-color:transparent!important;
}}
[data-baseweb="calendar"] button:not([aria-selected="true"]):hover{{background-color:{p['bg3']}!important}}

/* expander */
[data-testid="stExpander"]{{background-color:{p['bg2']}!important;border:1px solid {p['border']}!important}}
[data-testid="stExpander"] summary,[data-testid="stExpander"] summary p{{color:{p['text']}!important}}

/* status */
[data-testid="stStatus"]{{background-color:{p['bg2']}!important;border-color:{p['border']}!important}}
[data-testid="stStatus"] p{{color:{p['text']}!important}}

/* divider */
hr{{border-color:{p['border']}!important;opacity:0.5}}

/* dialog */
[data-testid="stModal"]>div{{background-color:{p['bg']}!important;border:1px solid {p['border']}!important}}
[data-testid="stModal"] h2,[data-testid="stModal"] p{{color:{p['text']}!important}}
[data-testid="stDialog"] [data-testid="stVerticalBlock"] p{{color:{p['text']}!important}}

/* ── Columns: no bg leak ── */
[data-testid="column"]{{background-color:transparent!important}}
</style>""", unsafe_allow_html=True)


# ── Portfolio sidebar expander ────────────────────────────────────────────────
def render_portfolio_sidebar() -> None:
    lang = st.session_state.lang
    p    = THEMES[st.session_state.theme]
    with st.expander(t("portfolio").upper()):
        for proj in PORTFOLIO:
            title   = proj["title_pt"] if lang == "pt" else proj["title"]
            desc    = proj["desc_pt"]  if lang == "pt" else proj["desc"]
            is_soon = proj["status"] == "soon"
            badge   = T[lang]["proj_soon"] if is_soon else T[lang]["proj_live"]
            badge_c = p["badge_soon"] if is_soon else p["badge_live"]

            tags_html = "".join(
                f'<span style="display:inline-block;font-size:0.62rem;background:{p["bg3"]};'
                f'color:{p["text2"]};border:1px solid {p["border"]};border-radius:4px;'
                f'padding:1px 5px;margin:1px 2px 1px 0;">{tag}</span>'
                for tag in proj["tags"]
            )

            links_html = ""
            if proj.get("url_live") and not proj.get("current"):
                links_html += (
                    f'<a href="{proj["url_live"]}" target="_blank" style="font-size:0.7rem;'
                    f'color:#4a86c8;text-decoration:none;">{T[lang]["proj_view"]}</a>'
                )
            if proj.get("url_github"):
                if links_html:
                    links_html += '<span style="color:#aaa;margin:0 4px;">·</span>'
                links_html += (
                    f'<a href="{proj["url_github"]}" target="_blank" style="font-size:0.7rem;'
                    f'color:#4a86c8;text-decoration:none;">{T[lang]["proj_github"]}</a>'
                )

            st.markdown(
                f'<div style="margin-bottom:10px;">'
                f'<div style="margin-bottom:3px;">'
                f'<span style="font-size:0.8rem;font-weight:700;color:{p["text"]};">{title}</span>'
                f'<span style="font-size:0.62rem;color:{badge_c};border:1px solid {badge_c};'
                f'border-radius:8px;padding:1px 6px;margin-left:6px;">{badge}</span>'
                f'</div>'
                f'<p style="font-size:0.68rem;color:{p["text2"]};margin:0 0 4px;'
                f'text-transform:uppercase;line-height:1.35;">{desc}</p>'
                f'<div style="margin-bottom:4px;">{tags_html}</div>'
                f'<div>{links_html}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ── Cache ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_db():
    return get_connection()

def embed_query_local(text: str) -> list[float]:
    from indexing.embedder import embed_query
    return embed_query(text)

# ── Generation ────────────────────────────────────────────────────────────────
def _build_messages(question: str, chunks: list[dict], lang: str = "en") -> list[dict]:
    parts = []
    for c in chunks:
        d = c["data"].strftime("%d/%m/%Y") if isinstance(c["data"], date) else str(c["data"])
        parts.append(
            f"[{c['orgao']} — Seção {c['secao']} — {d}]\n"
            f"Título: {c['titulo']}\n{c['chunk_texto']}"
        )
    lang_note = "Respond in English." if lang == "en" else "Responda em português."
    prompt = QA_TEMPLATE.format(context_str="\n\n---\n\n".join(parts), query_str=question)
    prompt += f"\n\n{lang_note}"
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]

def _is_rate_limit(exc: Exception) -> bool:
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "quota" in s or "too many requests" in s

def generate(question: str, chunks: list[dict], lang: str = "en", thinking=None) -> str:
    messages = _build_messages(question, chunks, lang=lang)

    if LLM_MODEL.startswith("ollama/"):
        resp = completion(model=LLM_MODEL, messages=messages, api_base=LLM_API_BASE)
        return (resp.choices[0].message.content or "").strip()

    def log(msg: str):
        if thinking: thinking.write(msg)

    # Primary model first, then fallback chain on rate limit
    chain = [(LLM_MODEL, LLM_MODEL.split("/")[-1])] + [
        (m, n) for m, n in API_FALLBACK_CHAIN if m != LLM_MODEL
    ]

    last_exc = None
    for model_id, name in chain:
        try:
            resp = completion(model=model_id, messages=messages)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            if _is_rate_limit(exc):
                log("⚠️ Rate limit reached — trying next model...")
                last_exc = exc
            else:
                raise
    raise RuntimeError(f"All models exhausted. Last error: {last_exc}")

# ── Context helper (inline citation cards) ───────────────────────────────────
def render_context(chunks: list[dict]) -> None:
    st.markdown(
        f'<p style="font-size:0.72rem;font-weight:600;color:#4a86c8;letter-spacing:0.07em;'
        f'text-transform:uppercase;margin:0.6rem 0 0.3rem;">— {t("context_label").format(n=len(chunks))} —</p>',
        unsafe_allow_html=True,
    )
    for c in chunks:
        d = c["data"].strftime("%d/%m/%Y") if isinstance(c["data"], date) else str(c["data"])
        titulo = c["titulo"][:90] + ("…" if len(c["titulo"]) > 90 else "")
        trecho = c["chunk_texto"][:240] + ("…" if len(c["chunk_texto"]) > 240 else "")
        st.markdown(
            f'<div style="border-left:3px solid #4a86c8;background:#f4f7fc;'
            f'padding:0.5rem 0.75rem;margin:0.25rem 0;border-radius:0 6px 6px 0;line-height:1.45;">'
            f'<strong style="font-size:0.8rem;color:#1a1a2e;">{titulo}</strong><br>'
            f'<span style="font-size:0.72rem;color:#777;">{c["orgao"]} · Seção {c["secao"]} · {d}</span><br>'
            f'<span style="font-size:0.78rem;color:#444;">{trecho}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# Render
# ═════════════════════════════════════════════════════════════════════════════
_inject_css()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title(t("settings"))
    lang_label = "🇧🇷 Português" if st.session_state.lang == "en" else "🇺🇸 English"
    if st.button(lang_label, use_container_width=True):
        st.session_state.lang = "pt" if st.session_state.lang == "en" else "en"
        st.rerun()
    top_k = st.slider(t("retrieved_chunks"), 2, 15, 6)
    st.divider()
    st.markdown(f"**{t('filters')}**")
    secao_label = st.selectbox(t("section"), t("sections_list"))
    secao = None if secao_label in ["(all)", "(todas)"] else secao_label
    c1, c2 = st.columns(2)
    data_ini = c1.date_input(t("date_from"), value=None)
    data_fim = c2.date_input(t("date_to"),   value=None)
    st.divider()
    show_context = st.toggle(t("show_context"), value=False)
    if st.button(t("clear_chat"), use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    st.divider()
    render_portfolio_sidebar()

# ── Title (below gear row) ────────────────────────────────────────────────────
st.markdown(
    f'<div style="text-align:center;padding:0.5rem 0 1.2rem;">'
    f'<div style="width:68px;height:68px;background:linear-gradient(135deg,#4a86c8,#2d6aad);'
    f'border-radius:50%;display:inline-flex;align-items:center;justify-content:center;'
    f'font-size:30px;box-shadow:0 4px 14px rgba(74,134,200,0.25);margin-bottom:0.6rem;">📋</div>'
    f'<h1 style="font-size:2rem;font-weight:700;margin:0 0 0.2rem 0;">DOU Chat</h1>'
    f'<p style="color:#888;font-size:0.88rem;margin:0;">{t("page_caption")}</p>'
    f'</div>',
    unsafe_allow_html=True,
)

# ── Suggested questions ───────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown(
        f'<p style="font-size:0.72rem;font-weight:600;color:#999;letter-spacing:0.08em;'
        f'text-transform:uppercase;text-align:center;margin-bottom:0.5rem;">{t("suggested")}</p>',
        unsafe_allow_html=True,
    )
    for col, suggestion in zip(st.columns(3), t("suggestions")):
        if col.button(suggestion, use_container_width=True):
            st.session_state["_pending_input"] = suggestion
            st.rerun()

# ── Chat history ──────────────────────────────────────────────────────────────
_ASST_BADGE = '<span style="font-size:0.65rem;font-weight:700;color:#4a86c8;letter-spacing:0.09em;text-transform:uppercase;">DOU ASSISTANT</span>'

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant":
            st.markdown(_ASST_BADGE, unsafe_allow_html=True)
        st.markdown(msg["content"])
        if msg.get("context") and show_context:
            render_context(msg["context"])

# ── Chat input ────────────────────────────────────────────────────────────────
pending    = st.session_state.pop("_pending_input", None)
user_input = st.chat_input(t("chat_placeholder"))
prompt     = pending or user_input

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(t("searching")):
            filters: dict = {}
            if secao:    filters["secao"]       = secao
            if data_ini: filters["data_inicio"] = data_ini
            if data_fim: filters["data_fim"]    = data_fim

            try:
                q_emb  = embed_query_local(prompt)
                chunks = similarity_search(
                    get_db(), q_emb, top_k=top_k,
                    filters=filters or None, chunks_table=CHUNKS_TABLE,
                )
            except Exception as e:
                st.error(f"{t('error_retrieval')}{e}")
                st.stop()

        if not chunks:
            answer, chunks = t("no_results"), []
        else:
            try:
                with st.spinner(t("searching")):
                    answer = generate(prompt, chunks, lang=st.session_state.lang)
            except Exception as e:
                answer = f"{t('error_generation')}{e}"

        st.markdown(_ASST_BADGE, unsafe_allow_html=True)
        st.markdown(answer)
        if chunks and show_context:
            render_context(chunks)

    st.session_state.messages.append({
        "role": "assistant", "content": answer, "context": chunks,
    })
