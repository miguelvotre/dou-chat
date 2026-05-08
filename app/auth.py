"""
Google OAuth for Streamlit.
Logs each access to the 'acessos' table in MotherDuck.
"""

import os
from datetime import datetime

import streamlit as st

DDL_ACESSOS = """
CREATE TABLE IF NOT EXISTS acessos (
    id          VARCHAR DEFAULT gen_random_uuid(),
    email       VARCHAR NOT NULL,
    nome        VARCHAR,
    picture_url VARCHAR,
    ip          VARCHAR,
    criado_em   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


def _init_acessos_table(conn) -> None:
    conn.execute(DDL_ACESSOS)


def _log_acesso(conn, user_info: dict) -> None:
    conn.execute(
        "INSERT INTO acessos (email, nome, picture_url) VALUES (?, ?, ?)",
        [user_info.get("email"), user_info.get("name"), user_info.get("picture")],
    )


def require_auth() -> dict | None:
    """
    Enforces Google OAuth authentication.
    Returns user info dict if authenticated, None otherwise.
    Logs the access to MotherDuck.
    """
    from streamlit_google_auth import Authenticate

    authenticator = Authenticate(
        secret_credentials_path=None,
        cookie_name="diario_oficial_auth",
        cookie_key=os.environ["GOOGLE_CLIENT_SECRET"],
        redirect_uri=os.getenv("OAUTH_REDIRECT_URI", "http://localhost:8501"),
        client_id=os.environ["GOOGLE_CLIENT_ID"],
        client_secret=os.environ["GOOGLE_CLIENT_SECRET"],
    )

    authenticator.check_authentification()

    if not st.session_state.get("connected"):
        st.title("DOU Chat")
        st.write("Sign in to access the system.")
        authenticator.login()
        return None

    user_info = st.session_state.get("user_info", {})

    # Log access only once per session
    if not st.session_state.get("_acesso_logado"):
        try:
            from indexing.store import get_connection
            conn = get_connection()
            _init_acessos_table(conn)
            _log_acesso(conn, user_info)
            conn.close()
        except Exception:
            pass  # do not block access on logging failure
        st.session_state["_acesso_logado"] = True

    return user_info
