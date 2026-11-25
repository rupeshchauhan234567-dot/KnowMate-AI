"""
KnowMate AI — Intelligent Document Workspace
=============================================

A polished Streamlit interface for the existing KnowMate RAG pipeline.

Expected project structure
--------------------------
app.py
core/
    __init__.py
    document_loader.py
    rag_pipeline.py
.env
requirements.txt

The app expects these existing functions:
- core.document_loader.load_uploaded_documents(uploaded_files)
- core.rag_pipeline.build_rag_chain(docs, session_id)
- core.rag_pipeline.ask_question(vector_store, question, api_key, model_name)
- core.rag_pipeline.generate_document_summary(vector_store, api_key, model_name)
- core.rag_pipeline.generate_quiz(vector_store, api_key, model_name, number_of_questions)

This file intentionally keeps the RAG backend separate. It improves the UI,
session handling, validation, accessibility, responsiveness and error handling
without changing your embedding or vector-store implementation.
"""

from __future__ import annotations

import html
import os
import uuid
import json
import sqlite3
from datetime import datetime
from typing import Any, Iterable

import streamlit as st
from dotenv import load_dotenv

from core.document_loader import load_uploaded_documents
from core.rag_pipeline import (
    ask_question,
    build_rag_chain,
    generate_document_summary,
    generate_quiz,
)


# =============================================================================
# 01. APP CONFIGURATION
# =============================================================================

load_dotenv()

APP_NAME = "KnowMate AI"
APP_TAGLINE = "Your intelligent knowledge workspace"
APP_VERSION = "2.0"

st.set_page_config(
    page_title=f"{APP_NAME} | Knowledge Workspace",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://docs.streamlit.io/",
        "Report a bug": "https://github.com/",
        "About": f"{APP_NAME} — Intelligent Document Assistant",
    },
)

PAGES = [
    "Dashboard",
    "AI Chat",
    "My Documents",
    "AI Tools",
    "Settings",
]

MODEL_OPTIONS = [
    "openai/gpt-oss-120b",
    "llama-3.1-8b-instant",
]

SUPPORTED_TYPES = ["pdf", "docx", "txt"]
DB_PATH = os.getenv("KNOWMATE_DB_PATH", "knowmate_chat_history.db")


# =============================================================================
# 02. SESSION STATE
# =============================================================================

def initialize_session_state() -> None:
    """Initialize Streamlit state without overwriting existing user choices."""
    defaults: dict[str, Any] = {
        "session_id": uuid.uuid4().hex[:12],
        "vector_store": None,
        "document_names": [],
        "messages": [],
        "page": "Dashboard",
        "summary": "",
        "quiz": "",
        "theme": "Light",
        "model_name": MODEL_OPTIONS[0],
        "api_key_input": os.getenv("GROQ_API_KEY", ""),
        "pending_question": "",
        "last_processed_at": None,
        "document_count": 0,
        "show_welcome": True,
        "active_conversation_id": None,
        "history_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if st.session_state.page not in PAGES:
        st.session_state.page = "Dashboard"
    if st.session_state.theme not in ("Light", "Dark"):
        st.session_state.theme = "Light"
    if st.session_state.model_name not in MODEL_OPTIONS:
        st.session_state.model_name = MODEL_OPTIONS[0]


def get_db_connection():
    """Return a SQLite connection; the database persists across app restarts."""
    connection = sqlite3.connect(DB_PATH, timeout=15, check_same_thread=False)
    connection.row_factory = sqlite3.Row
    return connection


def initialize_chat_database() -> None:
    with get_db_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                messages TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.commit()


def create_conversation() -> str:
    conversation_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    with get_db_connection() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, messages, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (conversation_id, "New conversation", "[]", now, now),
        )
        connection.commit()
    st.session_state.active_conversation_id = conversation_id
    st.session_state.messages = []
    return conversation_id


def save_current_conversation() -> None:
    conversation_id = st.session_state.get("active_conversation_id")
    if not conversation_id:
        conversation_id = create_conversation()
    messages = st.session_state.get("messages", [])
    first_user = next(
        (m.get("content", "") for m in messages if m.get("role") == "user"),
        "New conversation",
    )
    title = str(first_user).strip().replace("\n", " ")[:52] or "New conversation"
    now = datetime.now().isoformat(timespec="seconds")
    with get_db_connection() as connection:
        connection.execute(
            "UPDATE conversations SET title = ?, messages = ?, updated_at = ? WHERE id = ?",
            (title, json.dumps(messages, ensure_ascii=False), now, conversation_id),
        )
        connection.commit()


def get_conversations():
    with get_db_connection() as connection:
        return connection.execute(
            "SELECT id, title, messages, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()


def load_conversation(conversation_id: str) -> None:
    with get_db_connection() as connection:
        row = connection.execute(
            "SELECT messages FROM conversations WHERE id = ?", (conversation_id,)
        ).fetchone()
    if row:
        try:
            messages = json.loads(row["messages"] or "[]")
        except (TypeError, json.JSONDecodeError):
            messages = []
        st.session_state.active_conversation_id = conversation_id
        st.session_state.messages = messages if isinstance(messages, list) else []
        st.session_state.pending_question = ""


def delete_conversation(conversation_id: str) -> None:
    with get_db_connection() as connection:
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        connection.commit()
    if st.session_state.get("active_conversation_id") == conversation_id:
        create_conversation()


initialize_session_state()
initialize_chat_database()
if not st.session_state.get("active_conversation_id"):
    create_conversation()
if not st.session_state.get("history_loaded"):
    load_conversation(st.session_state.active_conversation_id)
    st.session_state.history_loaded = True




# =============================================================================
# 03. THEME TOKENS
# =============================================================================

def get_theme_tokens(theme_name: str) -> dict[str, str]:
    """Return a complete and readable color palette for the selected theme."""
    if theme_name == "Dark":
        return {
            "bg": "#090E1B",
            "bg_alt": "#0D1425",
            "surface": "#121C30",
            "surface_alt": "#17243B",
            "surface_hover": "#1C2B46",
            "sidebar": "#0B1323",
            "text": "#F4F7FF",
            "text_soft": "#D5DDF0",
            "muted": "#9AA9C4",
            "subtle": "#71819F",
            "border": "rgba(163, 181, 215, 0.17)",
            "border_strong": "rgba(163, 181, 215, 0.28)",
            "input": "#111D32",
            "accent": "#9B7BFF",
            "accent_strong": "#8057F2",
            "accent_soft": "rgba(155, 123, 255, 0.16)",
            "cyan": "#4DD8E8",
            "success": "#39C99A",
            "warning": "#F6BD60",
            "danger": "#FF7185",
            "shadow": "rgba(0, 0, 0, 0.24)",
            "hero_text": "#FFFFFF",
            "hero_muted": "#D9D9F1",
            "code": "#0A1222",
        }

    return {
        "bg": "#F5F7FC",
        "bg_alt": "#EEF2F9",
        "surface": "#FFFFFF",
        "surface_alt": "#F8FAFF",
        "surface_hover": "#F0F3FC",
        "sidebar": "#FFFFFF",
        "text": "#1D2740",
        "text_soft": "#34415D",
        "muted": "#65728B",
        "subtle": "#8792A7",
        "border": "#E3E8F2",
        "border_strong": "#D4DCEC",
        "input": "#FFFFFF",
        "accent": "#7652D6",
        "accent_strong": "#6240C2",
        "accent_soft": "rgba(118, 82, 214, 0.10)",
        "cyan": "#0891A8",
        "success": "#138A68",
        "warning": "#A96A12",
        "danger": "#C43F58",
        "shadow": "rgba(39, 54, 88, 0.08)",
        "hero_text": "#FFFFFF",
        "hero_muted": "#E7E9FF",
        "code": "#F2F4FA",
    }


THEME = get_theme_tokens(st.session_state.theme)


# =============================================================================
# 04. GLOBAL CSS
# =============================================================================

def inject_global_styles(tokens: dict[str, str]) -> None:
    """Install all app styles in one place for consistent theme behavior."""
    css = f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');

    :root {{
        --km-bg: {tokens['bg']};
        --km-bg-alt: {tokens['bg_alt']};
        --km-surface: {tokens['surface']};
        --km-surface-alt: {tokens['surface_alt']};
        --km-surface-hover: {tokens['surface_hover']};
        --km-sidebar: {tokens['sidebar']};
        --km-text: {tokens['text']};
        --km-text-soft: {tokens['text_soft']};
        --km-muted: {tokens['muted']};
        --km-subtle: {tokens['subtle']};
        --km-border: {tokens['border']};
        --km-border-strong: {tokens['border_strong']};
        --km-input: {tokens['input']};
        --km-accent: {tokens['accent']};
        --km-accent-strong: {tokens['accent_strong']};
        --km-accent-soft: {tokens['accent_soft']};
        --km-cyan: {tokens['cyan']};
        --km-success: {tokens['success']};
        --km-warning: {tokens['warning']};
        --km-danger: {tokens['danger']};
        --km-shadow: {tokens['shadow']};
        --km-code: {tokens['code']};
    }}

    html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
        font-family: 'DM Sans', sans-serif;
    }}

    html {{
        scroll-behavior: smooth;
    }}

    .stApp {{
        color: var(--km-text) !important;
        background:
            radial-gradient(ellipse at 4% 4%, rgba(124, 91, 220, 0.075), transparent 31%),
            radial-gradient(ellipse at 96% 10%, rgba(8, 145, 168, 0.055), transparent 28%),
            var(--km-bg) !important;
    }}

    [data-testid="stAppViewContainer"] {{
        background: transparent !important;
    }}

    [data-testid="stMain"] {{
        background: transparent !important;
    }}

    [data-testid="stMainBlockContainer"] {{
        max-width: 1480px;
        padding-top: 2.1rem;
        padding-bottom: 3.5rem;
        padding-left: clamp(1rem, 3vw, 3rem);
        padding-right: clamp(1rem, 3vw, 3rem);
    }}

    .block-container {{
        max-width: 1480px;
        padding-top: 2.1rem;
        padding-bottom: 3.5rem;
    }}

    /* Streamlit header: keep it themed, clean and readable. */
    header[data-testid="stHeader"] {{
        background: {tokens['bg']} !important;
        border-bottom: 1px solid var(--km-border) !important;
        box-shadow: none !important;
    }}

    [data-testid="stToolbar"] {{
        background: transparent !important;
    }}

    [data-testid="stDecoration"] {{
        display: none !important;
    }}

    .stDeployButton {{
        display: none !important;
    }}

    #MainMenu {{
        visibility: hidden !important;
    }}

    footer {{
        visibility: hidden !important;
    }}

    /* Typography */
    h1, h2, h3, h4, h5, h6 {{
        font-family: 'Space Grotesk', sans-serif !important;
        color: var(--km-text) !important;
        letter-spacing: -0.035em;
    }}

    p, li, label, small {{
        color: var(--km-text-soft);
    }}

    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    .stMarkdown span {{
        color: var(--km-text);
    }}

    [data-testid="stCaptionContainer"],
    .stCaption {{
        color: var(--km-muted) !important;
    }}

    a {{
        color: var(--km-accent) !important;
        text-decoration: none;
    }}

    a:hover {{
        color: var(--km-cyan) !important;
    }}

    /* Sidebar shell */
    section[data-testid="stSidebar"] {{
        background: var(--km-sidebar) !important;
        border-right: 1px solid var(--km-border) !important;
        min-width: 285px;
    }}

    section[data-testid="stSidebar"] > div {{
        background: transparent !important;
    }}

    section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {{
        gap: 0.72rem;
    }}

    section[data-testid="stSidebar"] .block-container {{
        padding: 1.25rem 1.05rem 2rem;
    }}

    section[data-testid="stSidebar"] * {{
        color: var(--km-text);
    }}

    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: var(--km-muted) !important;
    }}

    /* Brand */
    .km-brand {{
        display: flex;
        align-items: center;
        gap: 13px;
        padding: 9px 5px 17px;
    }}

    .km-brand-mark {{
        width: 48px;
        height: 48px;
        display: flex;
        align-items: center;
        justify-content: center;
        flex: 0 0 48px;
        border-radius: 16px;
        color: #FFFFFF !important;
        font-size: 25px;
        background: linear-gradient(140deg, #8B5CF6 0%, #5B55D9 55%, #0EA5B7 100%);
        box-shadow: 0 9px 26px rgba(111, 78, 220, 0.30);
    }}

    .km-brand-name {{
        color: var(--km-text) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 22px;
        line-height: 1.1;
        font-weight: 700;
        letter-spacing: -0.8px;
    }}

    .km-brand-caption {{
        color: var(--km-muted) !important;
        font-size: 10px;
        margin-top: 5px;
        letter-spacing: 1.55px;
        text-transform: uppercase;
        font-weight: 700;
    }}

    .km-side-label {{
        color: var(--km-subtle) !important;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.7px;
        text-transform: uppercase;
        margin: 18px 0 7px;
    }}

    .km-side-note {{
        border: 1px solid var(--km-border);
        background: var(--km-surface-alt);
        border-radius: 14px;
        padding: 13px;
        margin: 7px 0 3px;
    }}

    .km-side-note-title {{
        color: var(--km-text) !important;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 4px;
    }}

    .km-side-note-copy {{
        color: var(--km-muted) !important;
        font-size: 11px;
        line-height: 1.55;
    }}

    /* Sidebar button navigation */
    section[data-testid="stSidebar"] .stButton > button {{
        justify-content: flex-start;
        text-align: left;
        padding: 0.65rem 0.85rem;
        font-size: 0.88rem;
    }}

    /* Inputs */
    .stTextInput input,
    .stTextArea textarea,
    .stNumberInput input {{
        background: var(--km-input) !important;
        color: var(--km-text) !important;
        border: 1px solid var(--km-border-strong) !important;
        border-radius: 11px !important;
        min-height: 43px;
    }}

    .stTextInput input:focus,
    .stTextArea textarea:focus,
    .stNumberInput input:focus {{
        border-color: var(--km-accent) !important;
        box-shadow: 0 0 0 3px var(--km-accent-soft) !important;
    }}

    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {{
        color: var(--km-subtle) !important;
        opacity: 1;
    }}

    [data-baseweb="select"] > div {{
        background: var(--km-input) !important;
        border-color: var(--km-border-strong) !important;
        border-radius: 11px !important;
        color: var(--km-text) !important;
    }}

    [data-baseweb="select"] span,
    [data-baseweb="select"] input {{
        color: var(--km-text) !important;
    }}

    [data-baseweb="popover"],
    [data-baseweb="menu"],
    [role="listbox"] {{
        background: var(--km-surface) !important;
        border: 1px solid var(--km-border) !important;
    }}

    [role="option"] {{
        color: var(--km-text) !important;
    }}

    [role="option"]:hover {{
        background: var(--km-accent-soft) !important;
    }}

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button,
    .stLinkButton > a {{
        min-height: 43px;
        border-radius: 11px !important;
        border: 1px solid var(--km-border-strong) !important;
        background: var(--km-surface) !important;
        color: var(--km-text) !important;
        font-weight: 700 !important;
        transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease, background 0.22s ease !important;
    }}

    .stButton > button:hover,
    .stDownloadButton > button:hover,
    .stLinkButton > a:hover {{
        transform: translateY(-2px);
        border-color: var(--km-accent) !important;
        box-shadow: 0 9px 22px var(--km-shadow) !important;
        background: var(--km-surface-hover) !important;
    }}

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button {{
        color: #FFFFFF !important;
        border: 0 !important;
        background: linear-gradient(110deg, #8057E8, #5B56D7 58%, #138FA5) !important;
        box-shadow: 0 7px 18px rgba(105, 78, 211, 0.20);
    }}

    .stButton > button[kind="primary"]:hover,
    .stFormSubmitButton > button:hover {{
        color: #FFFFFF !important;
        background: linear-gradient(110deg, #906BF1, #6C66E5 58%, #1A9DB2) !important;
        box-shadow: 0 11px 26px rgba(105, 78, 211, 0.30) !important;
    }}

    .stButton > button:disabled {{
        opacity: 0.48 !important;
        transform: none !important;
        box-shadow: none !important;
    }}

    /* Upload dropzone */
    [data-testid="stFileUploader"] {{
        border: 1px dashed var(--km-border-strong) !important;
        border-radius: 14px !important;
        padding: 9px !important;
        background: var(--km-surface-alt) !important;
    }}

    [data-testid="stFileUploader"] section {{
        background: transparent !important;
    }}

    [data-testid="stFileUploader"] button {{
        background: var(--km-surface) !important;
        color: var(--km-text) !important;
        border: 1px solid var(--km-border-strong) !important;
        border-radius: 9px !important;
    }}

    [data-testid="stFileUploader"] small {{
        color: var(--km-muted) !important;
    }}

    /* Cards */
    .km-card {{
        position: relative;
        overflow: hidden;
        height: 100%;
        background: var(--km-surface);
        border: 1px solid var(--km-border);
        border-radius: 18px;
        padding: 21px;
        box-shadow: 0 7px 24px var(--km-shadow);
        transition: transform 0.24s ease, border-color 0.24s ease, box-shadow 0.24s ease;
    }}

    .km-card:hover {{
        transform: translateY(-5px);
        border-color: rgba(130, 94, 225, 0.48);
        box-shadow: 0 15px 34px var(--km-shadow);
    }}

    .km-card-title {{
        color: var(--km-text) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 16px;
        font-weight: 700;
        margin: 0 0 7px;
    }}

    .km-card-copy {{
        color: var(--km-muted) !important;
        font-size: 12px;
        line-height: 1.65;
        margin: 0;
    }}

    .km-card-icon {{
        width: 43px;
        height: 43px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 13px;
        margin-bottom: 15px;
        font-size: 21px;
        color: #7652D6 !important;
        background: var(--km-accent-soft);
    }}

    .km-stat {{
        background: var(--km-surface);
        border: 1px solid var(--km-border);
        border-radius: 17px;
        padding: 19px 20px;
        min-height: 132px;
        box-shadow: 0 6px 20px var(--km-shadow);
        transition: transform 0.22s ease, border-color 0.22s ease;
    }}

    .km-stat:hover {{
        transform: translateY(-4px);
        border-color: rgba(130, 94, 225, 0.5);
    }}

    .km-stat-icon {{
        font-size: 21px;
        margin-bottom: 13px;
    }}

    .km-stat-label {{
        color: var(--km-muted) !important;
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 5px;
    }}

    .km-stat-value {{
        color: var(--km-text) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 27px;
        font-weight: 700;
        line-height: 1.1;
    }}

    /* Hero */
    .km-hero {{
        position: relative;
        isolation: isolate;
        overflow: hidden;
        border-radius: 25px;
        padding: clamp(25px, 4vw, 43px);
        margin-bottom: 28px;
        border: 1px solid rgba(172, 151, 255, 0.30);
        background:
            radial-gradient(circle at 84% 15%, rgba(67, 206, 220, 0.24), transparent 25%),
            radial-gradient(circle at 95% 90%, rgba(168, 85, 247, 0.35), transparent 35%),
            linear-gradient(118deg, #29205D 0%, #373078 45%, #164C67 100%);
        box-shadow: 0 18px 45px rgba(53, 42, 112, 0.16);
        animation: km-fade-up 0.55s ease both;
    }}

    .km-hero::before {{
        content: "";
        position: absolute;
        z-index: -1;
        width: 280px;
        height: 280px;
        right: -95px;
        top: -125px;
        border-radius: 50%;
        background: rgba(180, 160, 255, 0.25);
        filter: blur(45px);
        animation: km-float 7s ease-in-out infinite alternate;
    }}

    .km-hero::after {{
        content: "";
        position: absolute;
        z-index: -1;
        width: 180px;
        height: 180px;
        right: 22%;
        bottom: -125px;
        border-radius: 50%;
        background: rgba(31, 210, 219, 0.16);
        filter: blur(35px);
    }}

    .km-hero-kicker {{
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #E6DEFF !important;
        border: 1px solid rgba(255,255,255,0.24);
        background: rgba(255,255,255,0.08);
        padding: 7px 12px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 1.4px;
    }}

    .km-hero-title {{
        color: #FFFFFF !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(31px, 4vw, 48px);
        line-height: 1.12;
        font-weight: 700;
        letter-spacing: -1.9px;
        margin: 18px 0 12px;
        max-width: 730px;
    }}

    .km-hero-gradient {{
        background: linear-gradient(90deg, #D6C8FF, #8CEAF1);
        -webkit-background-clip: text;
        background-clip: text;
        color: transparent !important;
        -webkit-text-fill-color: transparent;
    }}

    .km-hero-copy {{
        color: #E1E4F7 !important;
        max-width: 620px;
        font-size: 14px;
        line-height: 1.75;
        margin-bottom: 20px;
    }}

    .km-hero-foot {{
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 9px;
    }}

    .km-chip {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        color: var(--km-muted) !important;
        background: var(--km-accent-soft);
        border: 1px solid var(--km-border);
        padding: 6px 10px;
        border-radius: 999px;
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.4px;
    }}

    .km-chip-live {{
        color: #B9F7E4 !important;
        background: rgba(31, 201, 151, 0.13);
        border-color: rgba(31, 201, 151, 0.27);
    }}

    /* Page headings */
    .km-page-heading {{
        color: var(--km-text) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: clamp(25px, 3vw, 32px);
        font-weight: 700;
        letter-spacing: -1px;
        margin: 4px 0 4px;
    }}

    .km-page-subtitle {{
        color: var(--km-muted) !important;
        font-size: 13px;
        line-height: 1.6;
        margin-bottom: 22px;
    }}

    .km-section-heading {{
        color: var(--km-text) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 18px;
        font-weight: 700;
        margin: 6px 0 12px;
    }}

    .km-eyebrow {{
        color: var(--km-accent) !important;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 10px;
        font-weight: 800;
        margin-bottom: 6px;
    }}

    /* Document rows */
    .km-document-row {{
        display: flex;
        align-items: center;
        gap: 13px;
        padding: 14px;
        background: var(--km-surface);
        border: 1px solid var(--km-border);
        border-radius: 14px;
        margin-bottom: 9px;
        transition: border-color 0.2s ease, transform 0.2s ease;
    }}

    .km-document-row:hover {{
        border-color: var(--km-accent);
        transform: translateX(3px);
    }}

    .km-document-icon {{
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 12px;
        background: var(--km-accent-soft);
        font-size: 19px;
        flex: 0 0 42px;
    }}

    .km-document-name {{
        color: var(--km-text) !important;
        font-size: 13px;
        font-weight: 700;
        overflow-wrap: anywhere;
    }}

    .km-document-meta {{
        color: var(--km-muted) !important;
        font-size: 11px;
        margin-top: 3px;
    }}

    /* Chat messages */
    [data-testid="stChatMessage"] {{
        background: var(--km-surface) !important;
        border: 1px solid var(--km-border) !important;
        border-radius: 16px !important;
        padding: 15px !important;
        box-shadow: 0 5px 18px var(--km-shadow);
        animation: km-fade-up 0.28s ease both;
    }}

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] h1,
    [data-testid="stChatMessage"] h2,
    [data-testid="stChatMessage"] h3 {{
        color: var(--km-text) !important;
    }}

    [data-testid="stChatInput"] {{
        background: var(--km-surface) !important;
        border: 1px solid var(--km-border-strong) !important;
        border-radius: 16px !important;
        box-shadow: 0 7px 22px var(--km-shadow);
    }}

    [data-testid="stChatInput"] textarea {{
        color: var(--km-text) !important;
        background: transparent !important;
    }}

    [data-testid="stChatInput"] textarea::placeholder {{
        color: var(--km-subtle) !important;
    }}

    [data-testid="stChatMessage"] {{
        max-width: 100%;
        line-height: 1.75;
    }}

    [data-testid="stChatInput"] {{
        margin-top: 1rem;
    }}

    /* Tabs */
    button[data-baseweb="tab"] {{
        color: var(--km-muted) !important;
        font-weight: 700 !important;
    }}

    button[data-baseweb="tab"][aria-selected="true"] {{
        color: var(--km-accent) !important;
        border-bottom-color: var(--km-accent) !important;
    }}

    [data-baseweb="tab-highlight"] {{
        background: var(--km-accent) !important;
    }}

    /* Expander */
    [data-testid="stExpander"] {{
        background: var(--km-surface) !important;
        border: 1px solid var(--km-border) !important;
        border-radius: 13px !important;
    }}

    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span {{
        color: var(--km-text) !important;
    }}

    /* Alerts, tables and metrics */
    [data-testid="stAlert"] {{
        border-radius: 13px !important;
        border: 1px solid var(--km-border) !important;
    }}

    [data-testid="stMetricLabel"],
    [data-testid="stMetricValue"] {{
        color: var(--km-text) !important;
    }}

    [data-testid="stTable"],
    [data-testid="stDataFrame"] {{
        background: var(--km-surface) !important;
        color: var(--km-text) !important;
    }}

    hr {{
        border-color: var(--km-border) !important;
        margin: 1.1rem 0 !important;
    }}

    /* Progress */
    [data-testid="stProgressBar"] > div > div {{
        background: linear-gradient(90deg, #8057E8, #1596A9) !important;
    }}

    /* Footer */
    .km-footer {{
        text-align: center;
        color: var(--km-subtle) !important;
        font-size: 11px;
        padding: 30px 0 8px;
        margin-top: 35px;
        border-top: 1px solid var(--km-border);
    }}

    .km-footer strong {{
        color: var(--km-accent) !important;
    }}

    /* Empty states */
    .km-empty {{
        text-align: center;
        padding: 42px 22px;
        background: var(--km-surface);
        border: 1px dashed var(--km-border-strong);
        border-radius: 19px;
    }}

    .km-empty-icon {{
        font-size: 38px;
        margin-bottom: 10px;
    }}

    .km-empty-title {{
        color: var(--km-text) !important;
        font-family: 'Space Grotesk', sans-serif;
        font-size: 18px;
        font-weight: 700;
        margin-bottom: 7px;
    }}

    .km-empty-copy {{
        color: var(--km-muted) !important;
        font-size: 12px;
        line-height: 1.7;
        max-width: 480px;
        margin: 0 auto;
    }}

    /* Animations */
    @keyframes km-fade-up {{
        from {{ opacity: 0; transform: translateY(12px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes km-float {{
        from {{ transform: translate3d(0, 0, 0); }}
        to {{ transform: translate3d(-22px, 24px, 0); }}
    }}

    @keyframes km-pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.65; }}
    }}

    /* Responsive layout */
    @media (max-width: 900px) {{
        [data-testid="stMainBlockContainer"] {{
            padding-left: 1rem;
            padding-right: 1rem;
        }}

        .km-hero {{
            padding: 25px;
            border-radius: 20px;
        }}

        .km-hero-title {{
            font-size: 34px;
        }}
    }}

    @media (max-width: 600px) {{
        .km-hero-title {{
            font-size: 29px;
            letter-spacing: -1px;
        }}

        .km-page-heading {{
            font-size: 25px;
        }}

        .km-stat {{
            min-height: 115px;
            padding: 15px;
        }}

        .km-stat-value {{
            font-size: 23px;
        }}
    }}

    @media (prefers-reduced-motion: reduce) {{
        *,
        *::before,
        *::after {{
            animation-duration: 0.01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: 0.01ms !important;
            scroll-behavior: auto !important;
        }}
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


inject_global_styles(THEME)


# =============================================================================
# 05. HTML AND UI HELPERS
# =============================================================================

def safe_text(value: Any) -> str:
    """Escape user-controlled text before placing it in custom HTML."""
    return html.escape(str(value), quote=True)


def render_page_heading(title: str, subtitle: str, eyebrow: str = "KNOWMATE WORKSPACE") -> None:
    st.markdown(
        f"""
        <div class="km-eyebrow">{safe_text(eyebrow)}</div>
        <div class="km-page-heading">{safe_text(title)}</div>
        <div class="km-page-subtitle">{safe_text(subtitle)}</div>
        """,
        unsafe_allow_html=True,
    )


def render_stat_card(icon: str, label: str, value: Any) -> None:
    st.markdown(
        f"""
        <div class="km-stat">
            <div class="km-stat-icon">{safe_text(icon)}</div>
            <div class="km-stat-label">{safe_text(label)}</div>
            <div class="km-stat-value">{safe_text(value)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_feature_card(icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="km-card">
            <div class="km-card-icon">{safe_text(icon)}</div>
            <div class="km-card-title">{safe_text(title)}</div>
            <p class="km-card-copy">{safe_text(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(icon: str, title: str, description: str) -> None:
    st.markdown(
        f"""
        <div class="km-empty">
            <div class="km-empty-icon">{safe_text(icon)}</div>
            <div class="km-empty-title">{safe_text(title)}</div>
            <p class="km-empty-copy">{safe_text(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_document_row(name: str, index: int | None = None) -> None:
    clean_name = safe_text(name)
    suffix = f"Document {index}" if index is not None else "Indexed successfully"
    extension = name.rsplit(".", 1)[-1].upper() if "." in name else "FILE"
    icon = "📕" if extension == "PDF" else "📘" if extension == "DOCX" else "📄"

    st.markdown(
        f"""
        <div class="km-document-row">
            <div class="km-document-icon">{icon}</div>
            <div style="flex:1; min-width:0;">
                <div class="km-document-name">{clean_name}</div>
                <div class="km-document-meta">{safe_text(suffix)} · {safe_text(extension)} file</div>
            </div>
            <div class="km-chip">READY</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_footer() -> None:
    st.markdown(
        """
        <div class="km-footer">
            <strong>✦ KnowMate AI</strong>
            &nbsp; · &nbsp; Intelligent Document Assistant
            &nbsp; · &nbsp; Powered by Groq
        </div>
        """,
        unsafe_allow_html=True,
    )


def navigate(page_name: str) -> None:
    """Navigate using Streamlit buttons; avoids stale keyed-radio page state."""
    if page_name in PAGES:
        st.session_state.page = page_name
        st.rerun()




def has_knowledge_base() -> bool:
    return st.session_state.vector_store is not None


def current_api_key() -> str:
    return str(st.session_state.get("api_key_input", "")).strip()


def validate_api_key() -> bool:
    if not current_api_key():
        st.error("Groq API key enter karo. Sidebar → AI Settings mein key add karo.")
        return False
    return True


def show_sources(sources: Iterable[Any]) -> None:
    source_list = list(sources or [])
    if not source_list:
        return

    with st.expander(f"📎 View source references ({len(source_list)})"):
        for source in source_list:
            st.markdown(f"- {safe_text(source)}")


def clear_workspace() -> None:
    """Clear only the active workspace; SQLite conversation history is retained."""
    st.session_state.vector_store = None
    st.session_state.document_names = []
    st.session_state.summary = ""
    st.session_state.quiz = ""
    st.session_state.session_id = uuid.uuid4().hex[:12]
    st.session_state.last_processed_at = None
    st.session_state.pending_question = ""
    create_conversation()
    st.rerun()




# =============================================================================
# 06. SIDEBAR
# =============================================================================

def _sync_sidebar_theme():
    st.session_state.theme = st.session_state.sidebar_theme
    st.session_state.settings_theme_choice = st.session_state.sidebar_theme


def _sync_sidebar_model():
    st.session_state.model_name = st.session_state.sidebar_model
    st.session_state.settings_model_choice = st.session_state.sidebar_model


def _sync_settings_theme():
    st.session_state.theme = st.session_state.settings_theme_choice
    st.session_state.sidebar_theme = st.session_state.settings_theme_choice


def _sync_settings_model():
    st.session_state.model_name = st.session_state.settings_model_choice
    st.session_state.sidebar_model = st.session_state.settings_model_choice


def render_sidebar() -> tuple[str, str]:
    """Render button navigation and globally synchronized AI preferences."""
    with st.sidebar:
        st.markdown(
            """
            <div class="km-brand">
                <div class="km-brand-mark">🧠</div>
                <div>
                    <div class="km-brand-name">KnowMate AI</div>
                    <div class="km-brand-caption">Intelligent workspace</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="km-side-label">WORKSPACE</div>', unsafe_allow_html=True)

        nav_labels = {
            "Dashboard": "◈  Dashboard",
            "AI Chat": "✦  AI Document Chat",
            "My Documents": "▤  My Documents",
            "AI Tools": "✧  AI Productivity Tools",
            "Settings": "⚙  Settings",
        }
        for page_name in PAGES:
            if st.button(
                nav_labels[page_name],
                key=f"nav_{page_name.lower().replace(' ', '_')}",
                use_container_width=True,
                type="primary" if st.session_state.page == page_name else "secondary",
            ):
                navigate(page_name)

        st.markdown("---")
        st.markdown('<div class="km-side-label">AI CONFIGURATION</div>', unsafe_allow_html=True)
        st.text_input(
            "Groq API Key",
            type="password",
            key="api_key_input",
            placeholder="gsk_••••••••••••",
            help="Read from .env by default. You can override it here.",
        )

        if "sidebar_model" not in st.session_state:
            st.session_state.sidebar_model = st.session_state.model_name
        st.selectbox(
            "AI Model",
            MODEL_OPTIONS,
            key="sidebar_model",
            on_change=_sync_sidebar_model,
        )

        st.markdown("---")
        st.markdown('<div class="km-side-label">KNOWLEDGE INGESTION</div>', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Add documents to your workspace",
            type=SUPPORTED_TYPES,
            accept_multiple_files=True,
            help="Supported formats: PDF, DOCX and TXT.",
            key="document_uploader",
        )
        if st.button(
            "⚡  Process documents",
            type="primary",
            use_container_width=True,
            disabled=not uploaded_files,
            key="process_documents_button",
        ):
            process_uploaded_documents(uploaded_files)

        st.markdown("---")
        st.markdown('<div class="km-side-label">APPEARANCE</div>', unsafe_allow_html=True)
        if "sidebar_theme" not in st.session_state:
            st.session_state.sidebar_theme = st.session_state.theme
        st.selectbox(
            "Color theme",
            ["Light", "Dark"],
            key="sidebar_theme",
            on_change=_sync_sidebar_theme,
        )

        st.markdown("---")
        st.markdown('<div class="km-side-label">WORKSPACE STATUS</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="km-side-note">
                <div class="km-side-note-title">● {"Connected" if has_knowledge_base() else "Not connected"}</div>
                <div class="km-side-note-copy">
                    📚 {len(st.session_state.document_names)} document(s)<br>
                    💬 {len(st.session_state.messages)} active chat message(s)
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("↻  Clear workspace", use_container_width=True, key="clear_workspace_button"):
            clear_workspace()
        st.caption(f"Version {APP_VERSION} · SQLite chat history enabled")

    return current_api_key(), str(st.session_state.model_name)




# =============================================================================
# 07. DOCUMENT PROCESSING
# =============================================================================

def process_uploaded_documents(uploaded_files: list[Any] | None) -> None:
    """Load files, build the vector store and update workspace state."""
    if not uploaded_files:
        st.warning("Please choose at least one document.")
        return

    if not validate_api_key():
        return

    try:
        with st.spinner("Reading documents and preparing your knowledge base..."):
            docs, names = load_uploaded_documents(uploaded_files)

            if not docs:
                st.warning("No readable text was found in the selected documents.")
                return

            store = build_rag_chain(
                docs=docs,
                session_id=st.session_state.session_id,
            )

            if store is None:
                st.error("The vector store could not be created. Check core/rag_pipeline.py.")
                return

            st.session_state.vector_store = store
            st.session_state.document_names = list(dict.fromkeys(names))
            create_conversation()
            st.session_state.summary = ""
            st.session_state.quiz = ""
            st.session_state.last_processed_at = datetime.now().strftime("%d %b %Y, %I:%M %p")
            st.session_state.pending_question = ""

        st.success(
            f"Knowledge base ready! {len(st.session_state.document_names)} document(s) indexed."
        )
        st.rerun()

    except Exception as exc:
        st.error(f"Document processing failed: {exc}")
        st.exception(exc)


# =============================================================================
# 08. DASHBOARD PAGE
# =============================================================================

def render_dashboard() -> None:
    document_count = len(st.session_state.document_names)
    message_count = len(st.session_state.messages)
    ready_label = "Ready" if has_knowledge_base() else "Not ready"

    st.markdown(
        """
        <section class="km-hero">
            <div class="km-hero-kicker">✦ YOUR PERSONAL AI KNOWLEDGE COMPANION</div>
            <div class="km-hero-title">
                Think smarter.<br>
                <span class="km-hero-gradient">Discover more.</span>
            </div>
            <div class="km-hero-copy">
                Bring your documents together, ask meaningful questions,
                and turn scattered information into clear, useful knowledge.
            </div>
            <div class="km-hero-foot">
                <span class="km-chip km-chip-live">● WORKSPACE ACTIVE</span>
                <span class="km-chip">PDF · DOCX · TXT</span>
                <span class="km-chip">GROQ POWERED</span>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_page_heading(
        "Your workspace at a glance",
        "A simple overview of your documents, conversations and AI workspace.",
        "DASHBOARD",
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        render_stat_card("📚", "Documents indexed", document_count)
    with c2:
        render_stat_card("💬", "Chat messages", message_count)
    with c3:
        render_stat_card("⚡", "Knowledge base", ready_label)

    st.write("")
    left, right = st.columns([1.35, 1], gap="large")

    with left:
        st.markdown('<div class="km-section-heading">Explore your workspace</div>', unsafe_allow_html=True)
        feature_cols = st.columns(2, gap="medium")

        with feature_cols[0]:
            render_feature_card(
                "✦",
                "AI Document Chat",
                "Ask questions and get answers grounded in your indexed documents.",
            )
            if st.button("Open AI Chat →", use_container_width=True, key="dashboard_open_chat"):
                navigate("AI Chat")

        with feature_cols[1]:
            render_feature_card(
                "▤",
                "Document Library",
                "See which files are available in your searchable knowledge base.",
            )
            if st.button("View documents →", use_container_width=True, key="dashboard_open_docs"):
                navigate("My Documents")

        st.markdown("")
        tools_cols = st.columns(2, gap="medium")
        with tools_cols[0]:
            render_feature_card(
                "✧",
                "Smart Summary",
                "Create a concise overview of the indexed document context.",
            )
        with tools_cols[1]:
            render_feature_card(
                "⌁",
                "Knowledge Quiz",
                "Generate practice questions from your document context.",
            )

        if st.button("Explore AI productivity tools →", use_container_width=True):
            navigate("AI Tools")

    with right:
        st.markdown('<div class="km-section-heading">Recent documents</div>', unsafe_allow_html=True)
        recent = st.session_state.document_names[-5:]

        if recent:
            for index, name in enumerate(reversed(recent), start=1):
                render_document_row(name, index)
        else:
            render_empty_state(
                "📂",
                "Your library is empty",
                "Upload a PDF, DOCX or TXT file from the sidebar to start building your knowledge base.",
            )

        st.markdown('<div class="km-section-heading" style="margin-top:22px;">Workspace activity</div>', unsafe_allow_html=True)
        if st.session_state.last_processed_at:
            st.info(f"Last document processing: {st.session_state.last_processed_at}")
        else:
            st.caption("Your recent processing activity will appear here.")

        if not has_knowledge_base():
            st.warning("Add and process documents to unlock AI chat, summaries and quizzes.")


# =============================================================================
# 09. AI CHAT PAGE
# =============================================================================

def render_chat_welcome() -> None:
    st.markdown(
        """
        <div style="text-align:center; padding:36px 12px 22px;">
            <div style="font-size:42px; margin-bottom:8px;">✦</div>
            <div class="km-page-heading" style="font-size:25px;">What would you like to know?</div>
            <div class="km-page-subtitle" style="margin:8px auto 0; max-width:550px;">
                Ask a question about your uploaded documents. KnowMate will search
                your knowledge base and show the source references when available.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def answer_user_question(question: str, api_key: str, model_name: str) -> None:
    """Run a single RAG query and persist both sides of the conversation."""
    clean_question = question.strip()
    if not clean_question:
        return

    st.session_state.messages.append(
        {"role": "user", "content": clean_question}
    )

    with st.chat_message("user"):
        st.markdown(clean_question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents..."):
                answer, sources = ask_question(
                    vector_store=st.session_state.vector_store,
                    question=clean_question,
                    api_key=api_key,
                    model_name=model_name,
                )

            answer = str(answer)
            st.markdown(answer)
            show_sources(sources)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": list(sources or []),
                }
            )
            save_current_conversation()

        except Exception as exc:
            error_message = f"Something went wrong while generating the answer: {exc}"
            st.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )
            save_current_conversation()


def render_chat_page(api_key: str, model_name: str) -> None:
    render_page_heading(
        "AI Document Assistant",
        "A focused space to ask questions and revisit your previous conversations.",
        "CONVERSATIONAL SEARCH",
    )

    history_col, chat_col = st.columns([1.05, 3.2], gap="large")

    with history_col:
        st.markdown('<div class="km-section-heading">Your conversations</div>', unsafe_allow_html=True)
        if st.button("＋ New conversation", type="primary", use_container_width=True, key="new_chat_button"):
            create_conversation()
            st.rerun()

        conversations = get_conversations()
        if not conversations:
            st.caption("Your saved conversations will appear here.")
        for row in conversations:
            active = row["id"] == st.session_state.active_conversation_id
            label = ("● " if active else "　") + (row["title"] or "New conversation")
            preview = ""
            try:
                history_messages = json.loads(row["messages"] or "[]")
                preview = next(
                    (m.get("content", "") for m in reversed(history_messages) if m.get("role") == "user"),
                    "",
                )
            except (TypeError, json.JSONDecodeError):
                pass
            with st.container(border=True):
                if st.button(
                    label[:58],
                    key=f"open_conversation_{row['id']}",
                    use_container_width=True,
                    type="primary" if active else "secondary",
                    help=preview[:180] or "Open this conversation",
                ):
                    load_conversation(row["id"])
                    st.rerun()
                st.caption(datetime.fromisoformat(row["updated_at"]).strftime("%d %b · %I:%M %p"))
                if st.button("Delete", key=f"delete_conversation_{row['id']}", use_container_width=True):
                    delete_conversation(row["id"])
                    st.rerun()

    with chat_col:
        chat_header_left, chat_header_right = st.columns([3, 1])
        with chat_header_left:
            st.markdown(
                '<span class="km-chip km-chip-live">● KNOWLEDGE BASE CONNECTED</span>'
                if has_knowledge_base()
                else '<span class="km-chip">○ NO ACTIVE KNOWLEDGE BASE</span>',
                unsafe_allow_html=True,
            )
        with chat_header_right:
            st.caption(f"Model · {model_name}")

        if not st.session_state.messages:
            render_chat_welcome()
            suggestions = [
                "Summarize the main points",
                "What are the important topics?",
                "Explain the key concepts",
            ]
            suggestion_columns = st.columns(3, gap="small")
            for index, suggestion in enumerate(suggestions):
                with suggestion_columns[index]:
                    if st.button(suggestion, use_container_width=True, key=f"suggestion_{index}"):
                        st.session_state.pending_question = suggestion
                        st.rerun()

        for message in st.session_state.messages:
            with st.chat_message(message.get("role", "assistant")):
                st.markdown(message.get("content", ""))
                if message.get("sources"):
                    show_sources(message["sources"])

        if not has_knowledge_base():
            st.info("Upload and process PDF, DOCX or TXT files from the sidebar to ask new document-based questions. Your saved chat history remains available above.")
        question = st.chat_input(
            "Message KnowMate… Ask something about your documents",
            disabled=not has_knowledge_base(),
        )
        pending = st.session_state.pop("pending_question", "")
        if pending:
            question = pending

        if question:
            if not validate_api_key():
                return
            answer_user_question(question, api_key, model_name)
            st.rerun()




# =============================================================================
# 10. DOCUMENT LIBRARY PAGE
# =============================================================================

def render_documents_page() -> None:
    render_page_heading(
        "My Documents",
        "Your collection of searchable files, organized in one place.",
        "DOCUMENT LIBRARY",
    )

    names = st.session_state.document_names
    count = len(names)

    metric_col, info_col = st.columns([1, 2], gap="medium")
    with metric_col:
        render_stat_card("▤", "Indexed documents", count)
    with info_col:
        if has_knowledge_base():
            st.markdown(
                """
                <div class="km-card">
                    <div class="km-card-title">Knowledge base is ready</div>
                    <p class="km-card-copy">
                        Your documents have been processed and are available for semantic search.
                        Ask questions in AI Document Chat.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="km-card">
                    <div class="km-card-title">No active knowledge base</div>
                    <p class="km-card-copy">
                        Upload files in the sidebar and process them to make them searchable.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.markdown('<div class="km-section-heading">All documents</div>', unsafe_allow_html=True)

    if not names:
        render_empty_state(
            "🗂️",
            "No documents added",
            "Your document library is waiting. Upload your first file from the sidebar.",
        )
        return

    search_term = st.text_input(
        "Search document names",
        placeholder="Type a file name...",
        key="document_search",
    ).strip().lower()

    filtered_names = [
        name for name in names
        if search_term in name.lower()
    ]

    st.caption(f"Showing {len(filtered_names)} of {count} document(s)")

    if not filtered_names:
        st.info("No document names match your search.")
        return

    for index, name in enumerate(filtered_names, start=1):
        render_document_row(name, index)


# =============================================================================
# 11. AI PRODUCTIVITY TOOLS
# =============================================================================

def run_summary(api_key: str, model_name: str) -> None:
    if not validate_api_key():
        return

    try:
        with st.spinner("Creating a summary from your document context..."):
            result = generate_document_summary(
                vector_store=st.session_state.vector_store,
                api_key=api_key,
                model_name=model_name,
            )
        st.session_state.summary = str(result)
        st.success("Summary generated.")
    except Exception as exc:
        st.error(f"Summary generation failed: {exc}")
        st.exception(exc)


def run_quiz(api_key: str, model_name: str, number_of_questions: int) -> None:
    if not validate_api_key():
        return

    try:
        with st.spinner("Preparing your knowledge quiz..."):
            result = generate_quiz(
                vector_store=st.session_state.vector_store,
                api_key=api_key,
                model_name=model_name,
                number_of_questions=number_of_questions,
            )
        st.session_state.quiz = str(result)
        st.success("Quiz generated.")
    except Exception as exc:
        st.error(f"Quiz generation failed: {exc}")
        st.exception(exc)


def render_ai_tools_page(api_key: str, model_name: str) -> None:
    render_page_heading(
        "AI Productivity Tools",
        "Turn document knowledge into concise summaries and practice questions.",
        "LEARN FASTER",
    )

    if not has_knowledge_base():
        render_empty_state(
            "✧",
            "AI tools are waiting for documents",
            "Process at least one document from the sidebar before generating a summary or quiz.",
        )
        return

    summary_tab, quiz_tab = st.tabs(["✧  Smart Summary", "✎  Quiz Generator"])

    with summary_tab:
        render_feature_card(
            "✧",
            "AI Document Summary",
            "Generate a readable overview of key information from the retrieved document context.",
        )
        st.write("")
        if st.button("Generate summary", type="primary", key="generate_summary_button"):
            run_summary(api_key, model_name)

        if st.session_state.summary:
            st.markdown("---")
            st.markdown(st.session_state.summary)
            st.download_button(
                "⬇ Download summary",
                data=st.session_state.summary,
                file_name="knowmate_summary.txt",
                mime="text/plain",
                use_container_width=False,
                key="download_summary_button",
            )

    with quiz_tab:
        render_feature_card(
            "✎",
            "Knowledge Quiz",
            "Create multiple-choice questions to check your understanding of the document context.",
        )
        st.write("")
        number_of_questions = st.slider(
            "Number of questions",
            min_value=3,
            max_value=15,
            value=5,
            step=1,
            key="quiz_question_count",
        )

        if st.button("Generate quiz", type="primary", key="generate_quiz_button"):
            run_quiz(api_key, model_name, number_of_questions)

        if st.session_state.quiz:
            st.markdown("---")
            st.markdown(st.session_state.quiz)
            st.download_button(
                "⬇ Download quiz",
                data=st.session_state.quiz,
                file_name="knowmate_quiz.txt",
                mime="text/plain",
                use_container_width=False,
                key="download_quiz_button",
            )

    st.caption(
        "Note: summary and quiz coverage depends on the context retrieved by the RAG pipeline. "
        "For exhaustive full-document analysis, the backend must process all document chunks."
    )


# =============================================================================
# 12. SETTINGS PAGE
# =============================================================================

def render_settings_page() -> None:
    render_page_heading(
        "Workspace Settings",
        "Manage appearance, model preferences and your local conversation archive.",
        "PREFERENCES",
    )

    st.markdown('<div class="km-section-heading">Appearance</div>', unsafe_allow_html=True)
    if "settings_theme_choice" not in st.session_state:
        st.session_state.settings_theme_choice = st.session_state.theme
    st.radio(
        "Choose a theme",
        ["Light", "Dark"],
        horizontal=True,
        key="settings_theme_choice",
        on_change=_sync_settings_theme,
    )

    st.markdown("---")
    st.markdown('<div class="km-section-heading">AI Model Settings</div>', unsafe_allow_html=True)
    if "settings_model_choice" not in st.session_state:
        st.session_state.settings_model_choice = st.session_state.model_name
    st.selectbox(
        "Default Groq model",
        MODEL_OPTIONS,
        key="settings_model_choice",
        on_change=_sync_settings_model,
    )
    st.caption("The selected model is shared with the sidebar and used for new AI responses.")

    st.markdown("---")
    st.markdown('<div class="km-section-heading">Session details</div>', unsafe_allow_html=True)
    details_left, details_right, details_history = st.columns(3)
    with details_left:
        st.text_input("Session ID", value=st.session_state.session_id, disabled=True)
    with details_right:
        st.text_input("Indexed documents", value=str(len(st.session_state.document_names)), disabled=True)
    with details_history:
        st.text_input("Saved conversations", value=str(len(get_conversations())), disabled=True)
    if st.session_state.last_processed_at:
        st.caption(f"Last processed: {st.session_state.last_processed_at}")

    st.markdown("---")
    st.markdown('<div class="km-section-heading">Workspace reset</div>', unsafe_allow_html=True)
    st.info("Clear workspace removes the currently loaded document index and starts a fresh conversation. Previously saved conversations in SQLite are not deleted.")
    confirm_clear = st.checkbox(
        "I understand that this resets the active document workspace.",
        key="confirm_clear_workspace",
    )
    if st.button(
        "Clear current workspace",
        type="primary",
        disabled=not confirm_clear,
        key="settings_clear_workspace",
    ):
        clear_workspace()




# =============================================================================
# 13. APP ROUTER
# =============================================================================

api_key, model_name = render_sidebar()

current_page = st.session_state.page

if current_page == "Dashboard":
    render_dashboard()

elif current_page == "AI Chat":
    render_chat_page(api_key, model_name)

elif current_page == "My Documents":
    render_documents_page()

elif current_page == "AI Tools":
    render_ai_tools_page(api_key, model_name)

elif current_page == "Settings":
    render_settings_page()

else:
    st.session_state.page = "Dashboard"
    st.rerun()

render_footer()
