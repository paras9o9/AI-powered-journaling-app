import streamlit as st
from journal_model import load_model, predict_entry
from decision_logic import decide_next_step, score_phq9, score_gad7

st.set_page_config(
    page_title="MindJournal · AI Support",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

MODEL_SOURCE = "paras9o9/journal-distilbert-model"


@st.cache_resource
def get_model():
    tokenizer, model = load_model(
        model_path=MODEL_SOURCE,
        tokenizer_name=MODEL_SOURCE,
    )
    return tokenizer, model


tokenizer, model = get_model()

# ── Session state ──────────────────────────────────────────────────────────────
DEFAULTS = {
    "analysis_done": False,
    "pred_label": None,
    "prob_dict": None,
    "decision": None,
    "user_text": "",
    "q_submitted": False,
    "phq_total": None,
    "phq_severity": None,
    "gad_total": None,
    "gad_severity": None,
    "final_decision": None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def reset_app():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v
    st.rerun()


# ── Helpers ────────────────────────────────────────────────────────────────────
BADGE_MAP = {
    "NEU": ("Neutral", "#16a34a", "#f0fdf4", "#166534", "🟢"),
    "HUMOR": ("Humor / Positive", "#2563eb", "#eff6ff", "#1e40af", "😄"),
    "MH": ("Mental health concern", "#d97706", "#fef3c7", "#92400e", "🟡"),
    "SI": ("Suicidal ideation signal", "#dc2626", "#fef2f2", "#991b1b", "🔴"),
}

LABEL_ORDER = [
    ("NEU", "Neutral", "#16a34a"),
    ("HUMOR", "Humor / Positive", "#2563eb"),
    ("MH", "Mental health concern", "#d97706"),
    ("SI", "Suicidal ideation signal", "#dc2626"),
]


def get_badge(label):
    return BADGE_MAP.get(label, (label, "#6366f1", "#eef2ff", "#3730a3", "•"))


def get_support_message(decision):
    rec = decision.get("recommendation", "general_positive_feedback")
    if decision.get("showcrisiscard") or decision.get("show_crisis_card"):
        return (
            "crisis",
            "This entry may reflect significant distress. If you are in immediate "
            "danger, contact emergency services or iCall India: 9152987821",
        )
    if rec == "seek_professional_help":
        return ("warn", "Speaking with a mental health professional would be a helpful next step.")
    if rec == "consider_professional_help":
        return ("info", "Talking to a counsellor, therapist, or trusted person may be beneficial.")
    if rec in ("monitorandselfcare", "monitor_and_self_care"):
        return (
            "soft",
            "You may be going through a difficult period. Rest, self-care, and checking in "
            "with someone you trust can help.",
        )
    return ("ok", "No high-concern signal detected. Your feelings still matter — keep journalling.")


# ── Styles ────────────────────────────────────────────────────────────────────
st.html("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
html, body, [class*="css"], .stApp {
    font-family: 'Inter', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: #f8fafc; min-height: 100vh; }
.block-container {
    max-width: 700px !important;
    padding: 1.75rem 1.5rem 4rem !important;
    margin: 0 auto !important;
}
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }

/* ── Top bar ── */
.mj-topbar {
    display: flex; align-items: center;
    justify-content: space-between;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 1.5rem;
}
.mj-logo { display: flex; align-items: center; gap: 10px; }
.mj-logo-icon {
    width: 34px; height: 34px; background: #4f46e5;
    border-radius: 9px; display: flex; align-items: center;
    justify-content: center; font-size: 15px;
}
.mj-logo-name { font-size: 15px; font-weight: 700; color: #0f172a; letter-spacing: -0.01em; }
.mj-logo-name em { color: #4f46e5; font-style: normal; }
.mj-emergency-chip {
    font-size: 11px; font-weight: 600;
    padding: 4px 10px; border-radius: 999px;
    background: #fff7ed; color: #9a3412;
    border: 1px solid #fed7aa; letter-spacing: 0.01em;
}

/* ── Step tabs ── */
.mj-tabs {
    display: flex; gap: 4px; background: #f1f5f9;
    border-radius: 11px; padding: 3px; margin-bottom: 1.5rem;
}
.mj-tab {
    flex: 1; text-align: center; padding: 7px 4px;
    border-radius: 8px; font-size: 12px; font-weight: 600;
    color: #64748b;
}
.mj-tab.active {
    background: #fff; color: #4f46e5;
    box-shadow: 0 1px 3px rgba(15,23,42,0.08), 0 0 0 0.5px #e2e8f0;
}

/* ── Cards ── */
.mj-card {
    background: #fff;
    border: 1px solid #e8edf3;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    margin-bottom: 1rem;
}
.mj-card-label {
    font-size: 11px; font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.09em;
    color: #6366f1; margin-bottom: 6px;
}
.mj-card-title {
    font-size: 17px; font-weight: 700;
    color: #0f172a; margin-bottom: 5px; letter-spacing: -0.01em;
}
.mj-card-body {
    font-size: 14px; color: #64748b; line-height: 1.6; margin: 0;
}

/* ── Word count ── */
.mj-wc {
    display: flex; align-items: center; gap: 7px;
    font-size: 12px; color: #94a3b8; margin: 8px 0 14px; font-weight: 500;
}

/* ── Textarea ── */
div[data-testid="stTextArea"] { margin-top: 10px; }
div[data-testid="stTextArea"] textarea {
    border-radius: 11px !important;
    border: 1.5px solid #cbd5e1 !important;
    background: #f8fafc !important;
    color: #0f172a !important;
    caret-color: #4f46e5 !important;
    padding: 13px 14px !important;
    font-size: 14px !important;
    line-height: 1.65 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    resize: vertical !important;
    transition: border-color 0.15s, box-shadow 0.15s !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    background: #fff !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.1) !important;
    outline: none !important;
}
div[data-testid="stTextArea"] textarea::placeholder {
    color: #94a3b8 !important; font-style: italic !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button,
div[data-testid="stFormSubmitButton"] > button {
    border-radius: 10px !important;
    min-height: 42px !important;
    font-size: 13.5px !important;
    font-weight: 700 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    letter-spacing: 0.01em !important;
    transition: all 0.14s ease !important;
    cursor: pointer !important;
}
div[data-testid="stBaseButton-primary"] > button,
div[data-testid="stFormSubmitButton"] > button {
    background: #4f46e5 !important;
    color: #fff !important;
    border: none !important;
    box-shadow: 0 1px 2px rgba(79,70,229,0.25) !important;
}
div[data-testid="stBaseButton-primary"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #4338ca !important;
    box-shadow: 0 4px 12px rgba(79,70,229,0.3) !important;
    transform: translateY(-1px) !important;
}
div[data-testid="stBaseButton-secondary"] > button {
    background: #fff !important;
    color: #475569 !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.05) !important;
}
div[data-testid="stBaseButton-secondary"] > button:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
}

/* ── Result badge ── */
.mj-badge {
    display: inline-flex; align-items: center; gap: 8px;
    padding: 6px 15px; border-radius: 999px;
    font-size: 13px; font-weight: 700; margin-bottom: 14px;
}

/* ── Metric pills ── */
.mj-pills { display: flex; flex-wrap: wrap; gap: 7px; margin-bottom: 18px; }
.mj-pill {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 8px; padding: 6px 12px;
    font-size: 12px; font-weight: 600; color: #334155;
}
.mj-pill-key { color: #94a3b8; font-weight: 500; margin-right: 4px; }

/* ── Confidence bars ── */
.mj-conf-head {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #94a3b8; margin-bottom: 11px;
}
.mj-conf-row { display: flex; align-items: center; gap: 10px; margin-bottom: 9px; }
.mj-conf-name {
    font-size: 12.5px; font-weight: 600; color: #334155;
    width: 175px; flex-shrink: 0;
}
.mj-conf-track {
    flex: 1; height: 7px; background: #f1f5f9;
    border-radius: 99px; overflow: hidden;
}
.mj-conf-fill { height: 100%; border-radius: 99px; }
.mj-conf-pct { font-size: 12px; font-weight: 700; color: #64748b; width: 34px; text-align: right; }

/* ── Message boxes ── */
.mj-msg {
    border-radius: 11px; padding: 13px 15px;
    font-size: 13.5px; line-height: 1.6; font-weight: 500;
    display: flex; gap: 11px; align-items: flex-start; margin-top: 2px;
}
.mj-msg-icon { font-size: 16px; margin-top: 1px; flex-shrink: 0; }
.mj-crisis { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }
.mj-warn   { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
.mj-info   { background: #eff6ff; color: #1d4ed8; border: 1px solid #93c5fd; }
.mj-soft   { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.mj-ok     { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }

/* ── Questionnaire results ── */
.mj-q-sec-label {
    font-size: 11px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.08em; color: #6366f1;
    padding-bottom: 8px; border-bottom: 1px solid #f1f5f9; margin-bottom: 12px;
}
.mj-q-pills { display: flex; flex-wrap: wrap; gap: 8px; margin: 10px 0 6px; }
.mj-q-pill { border-radius: 9px; padding: 7px 13px; font-size: 13px; font-weight: 700; }
.mj-q-min  { background: #f0fdf4; color: #166534; border: 1px solid #bbf7d0; }
.mj-q-mild { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
.mj-q-mod  { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.mj-q-sev  { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }

/* ── Expander ── */
div[data-testid="stExpander"] details {
    background: #fff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 700 !important; color: #1e293b !important;
    font-size: 14px !important; padding: 14px 16px !important;
    transition: background 0.12s !important;
}
div[data-testid="stExpander"] summary:hover { background: #fafafa !important; }
div[data-testid="stExpander"] details > div { padding: 0 16px 16px !important; }

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 9px !important;
    border-color: #e2e8f0 !important;
    background: #f8fafc !important;
    font-size: 13.5px !important;
}

/* ── Footer ── */
.mj-footer {
    text-align: center; font-size: 12px; color: #94a3b8;
    padding-top: 1.25rem; border-top: 1px solid #e2e8f0;
    margin-top: 2rem; line-height: 1.7;
}
.mj-footer strong { color: #64748b; }
</style>
""")

# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="mj-topbar">
  <div class="mj-logo">
    <div class="mj-logo-icon">🧠</div>
    <div class="mj-logo-name">Mind<em>Journal</em></div>
  </div>
  <div class="mj-emergency-chip">Not for emergency use</div>
</div>
""",
    unsafe_allow_html=True,

