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


# ── Label / badge helpers ──────────────────────────────────────────────────────
BADGE_MAP = {
    "NEU":   ("Neutral",                  "#16a34a", "#dcfce7", "🟢"),
    "HUMOR": ("Humor / Positive",         "#2563eb", "#dbeafe", "😄"),
    "MH":    ("Mental health concern",    "#d97706", "#fef3c7", "🟡"),
    "SI":    ("Suicidal ideation signal", "#dc2626", "#fee2e2", "🔴"),
}


def get_badge(label):
    return BADGE_MAP.get(label, (label, "#6366f1", "#ede9fe", "•"))


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


# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<link href="https://api.fontshare.com/v2/css?f[]=satoshi@400,500,700&display=swap" rel="stylesheet">
<style>
/* ── Reset / base ── */
html, body, [class*="css"], .stApp {
    font-family: 'Satoshi', 'Inter', system-ui, sans-serif !important;
    -webkit-font-smoothing: antialiased;
}
.stApp { background: #f1f5f9; min-height: 100vh; }
.block-container {
    max-width: 740px !important;
    padding: 2rem 1.5rem 4rem !important;
    margin: 0 auto !important;
}
#MainMenu, footer, header { visibility: hidden; }
div[data-testid="stToolbar"] { display: none; }

/* ── Header ── */
.mj-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #e2e8f0;
    margin-bottom: 2rem;
}
.mj-logo { display: flex; align-items: center; gap: 0.6rem; }
.mj-logo-icon {
    width: 36px; height: 36px;
    background: #4f46e5; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.mj-logo-name { font-size: 1.05rem; font-weight: 700; color: #0f172a; }
.mj-logo-name span { color: #4f46e5; }
.mj-chip {
    font-size: 0.72rem; font-weight: 600;
    padding: 0.3rem 0.75rem; border-radius: 999px;
    background: #fff7ed; color: #9a3412; border: 1px solid #fed7aa;
}

/* ── Cards ── */
.mj-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 3px rgba(15,23,42,.04), 0 4px 12px rgba(15,23,42,.04);
}
.mj-step-label {
    font-size: 0.7rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .1em;
    color: #6366f1; margin-bottom: 0.4rem;
}
.mj-card-title {
    font-size: 1.05rem; font-weight: 700;
    color: #0f172a; margin-bottom: 0.35rem; line-height: 1.3;
}
.mj-card-body {
    font-size: 0.92rem; color: #475569;
    line-height: 1.65; margin-bottom: 0;
}

/* ── Word count ── */
.mj-wc {
    font-size: 0.78rem; color: #94a3b8;
    font-weight: 500; margin: 0.4rem 0 0.75rem;
}

/* ── Textarea overrides ── */
div[data-testid="stTextArea"] textarea {
    border-radius: 12px !important;
    border: 1.5px solid #cbd5e1 !important;
    background: #f8fafc !important;
    color: #0f172a !important;
    caret-color: #4f46e5 !important;
    padding: 0.9rem 1rem !important;
    font-size: 0.95rem !important;
    line-height: 1.7 !important;
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    resize: vertical !important;
    transition: border-color .18s, box-shadow .18s !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    background: #ffffff !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,.12) !important;
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
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    font-family: 'Satoshi', 'Inter', sans-serif !important;
    letter-spacing: .01em !important;
    transition: all .16s ease !important;
    cursor: pointer !important;
}
/* Primary */
div[data-testid="stBaseButton-primary"] > button,
div[data-testid="stFormSubmitButton"] > button {
    background: #4f46e5 !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: 0 1px 3px rgba(79,70,229,.3), 0 4px 12px rgba(79,70,229,.2) !important;
}
div[data-testid="stBaseButton-primary"] > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #4338ca !important;
    box-shadow: 0 4px 16px rgba(79,70,229,.35) !important;
    transform: translateY(-1px) !important;
}
/* Secondary */
div[data-testid="stBaseButton-secondary"] > button {
    background: #ffffff !important;
    color: #475569 !important;
    border: 1.5px solid #e2e8f0 !important;
    box-shadow: 0 1px 2px rgba(15,23,42,.06) !important;
}
div[data-testid="stBaseButton-secondary"] > button:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
    color: #0f172a !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] details {
    background: #ffffff !important;
    border: 1px solid #e2e8f0 !important;
    border-radius: 14px !important;
    box-shadow: 0 1px 3px rgba(15,23,42,.04) !important;
    overflow: hidden !important;
}
div[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #1e293b !important;
    font-size: 0.9rem !important;
    padding: 1rem 1.1rem !important;
    transition: background .15s !important;
}
div[data-testid="stExpander"] summary:hover { background: #f8fafc !important; }
div[data-testid="stExpander"] details > div { padding: 0.25rem 1.1rem 1.1rem !important; }

/* ── Selectbox ── */
div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border-color: #cbd5e1 !important;
    background: #f8fafc !important;
    font-size: 0.88rem !important;
}

/* ── Result badge ── */
.mj-badge {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.5rem 1.1rem; border-radius: 999px;
    font-size: 0.88rem; font-weight: 700; margin-bottom: 1rem;
}

/* ── Metric pills ── */
.mj-metric-row { display: flex; flex-wrap: wrap; gap: 0.55rem; margin: 0.5rem 0 1rem; }
.mj-pill {
    background: #f8fafc; border: 1px solid #e2e8f0;
    border-radius: 10px; padding: 0.45rem 0.85rem;
    font-size: 0.82rem; font-weight: 600; color: #334155;
}
.mj-pill-key { color: #94a3b8; font-weight: 500; margin-right: 0.3rem; }

/* ── Confidence bars ── */
.mj-conf-title {
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .09em;
    color: #94a3b8; margin-bottom: 0.8rem;
}
.mj-conf-row { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.7rem; }
.mj-conf-label { font-size: 0.82rem; font-weight: 600; color: #334155; width: 190px; flex-shrink: 0; }
.mj-conf-bg { flex: 1; height: 8px; background: #f1f5f9; border-radius: 999px; overflow: hidden; }
.mj-conf-fill { height: 100%; border-radius: 999px; }
.mj-conf-pct { font-size: 0.78rem; font-weight: 700; color: #64748b; width: 40px; text-align: right; }

/* ── Message boxes ── */
.mj-msg {
    border-radius: 12px; padding: 0.9rem 1.1rem;
    font-size: 0.9rem; line-height: 1.65; font-weight: 500;
    display: flex; gap: 0.75rem; align-items: flex-start;
}
.mj-msg-icon { font-size: 1.1rem; margin-top: 0.05rem; flex-shrink: 0; }
.mj-crisis { background: #fef2f2; color: #991b1b; border: 1px solid #fca5a5; }
.mj-warn   { background: #fffbeb; color: #92400e; border: 1px solid #fcd34d; }
.mj-info   { background: #eff6ff; color: #1d4ed8; border: 1px solid #93c5fd; }
.mj-soft   { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }
.mj-ok     { background: #f0fdf4; color: #166534; border: 1px solid #86efac; }

/* ── Questionnaire ── */
.mj-q-header {
    font-size: 0.72rem; font-weight: 700;
    text-transform: uppercase; letter-spacing: .09em;
    color: #6366f1; padding-bottom: 0.5rem;
    border-bottom: 1px solid #e2e8f0; margin-bottom: 0.9rem;
}
.mj-q-result-row { display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.75rem 0; }
.mj-q-pill { border-radius: 10px; padding: 0.55rem 1rem; font-size: 0.85rem; font-weight: 700; }
.mj-q-min  { background: #dcfce7; color: #166534; border: 1px solid #86efac; }
.mj-q-mild { background: #fef9c3; color: #854d0e; border: 1px solid #fde047; }
.mj-q-mod  { background: #fef3c7; color: #92400e; border: 1px solid #fcd34d; }
.mj-q-sev  { background: #fee2e2; color: #991b1b; border: 1px solid #fca5a5; }

/* ── Divider ── */
.mj-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #e2e8f0 30%, #e2e8f0 70%, transparent);
    margin: 1rem 0; border: none;
}

/* ── Footer ── */
.mj-footer {
    text-align: center; font-size: 0.78rem; color: #94a3b8;
    margin-top: 2rem; padding-top: 1.25rem;
    border-top: 1px solid #e2e8f0; line-height: 1.6;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="mj-header">
  <div class="mj-logo">
    <div class="mj-logo-icon">🧠</div>
    <div class="mj-logo-name">Mind<span>Journal</span></div>
  </div>
  <div class="mj-chip">Not for emergency use</div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Step 1 — journal entry ─────────────────────────────────────────────────────
st.markdown(
    """
<div class="mj-card">
  <div class="mj-step-label">Step 1 of 3 · Your entry</div>
  <div class="mj-card-title">What's on your mind today?</div>
  <div class="mj-card-body">
    Write freely about your day, your thoughts, or anything that has been weighing on you.
    Your entry is only processed locally.
  </div>
</div>
""",
    unsafe_allow_html=True,
)

user_text = st.text_area(
    "Journal entry",
    height=200,
    placeholder=(
        "e.g. I've been feeling emotionally drained lately. Some days I manage well, "
        "but other times everything feels heavier than usual…"
    ),
    key="user_text",
    label_visibility="collapsed",
)

word_count = len(user_text.split()) if user_text.strip() else 0

if word_count == 0:
    wc_color, wc_hint = "#94a3b8", "Try writing at least 2–3 sentences for better feedback."
elif word_count < 10:
    wc_color, wc_hint = "#d97706", "A bit short — a few more words helps the model."
elif word_count <= 150:
    wc_color, wc_hint = "#16a34a", "Good length."
else:
    wc_color, wc_hint = "#d97706", "Long entry — the model reads up to ~512 tokens."

st.markdown(
    f'<p class="mj-wc"><span style="color:{wc_color};font-weight:700">{word_count} words</span>'
    f"&nbsp;·&nbsp; {wc_hint}</p>",
    unsafe_allow_html=True,
)

col_analyse, col_reset = st.columns([5, 1])
with col_analyse:
    analyse = st.button("Analyse my entry", type="primary", use_container_width=True)
with col_reset:
    if st.button("Reset", use_container_width=True):
        reset_app()

# ── Run analysis ───────────────────────────────────────────────────────────────
if analyse:
    if not user_text.strip():
        st.warning("Please write something before analysing.")
    elif word_count < 3:
        st.warning("Your entry is very short. Write a few more words for useful feedback.")
    else:
        with st.spinner("Analysing your entry…"):
            pred_label, prob_dict = predict_entry(user_text, tokenizer, model)
            decision = decide_next_step(pred_label, prob_dict)

        st.session_state.update(
            analysis_done=True,
            pred_label=pred_label,
            prob_dict=prob_dict,
            decision=decision,
            q_submitted=False,
            phq_total=None,
            phq_severity=None,
            gad_total=None,
            gad_severity=None,
            final_decision=None,
        )

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.analysis_done:
    pred_label = st.session_state.pred_label
    prob_dict  = st.session_state.prob_dict
    decision   = st.session_state.decision

    label_text, label_color, label_bg, label_icon = get_badge(pred_label)
    msg_type, msg_text = get_support_message(decision)
    risk_tier = decision.get("risk_tier", "low").capitalize()

    LABEL_ORDER = [
        ("NEU",   "Neutral",                  "#16a34a"),
        ("HUMOR", "Humor / Positive",         "#2563eb"),
        ("MH",    "Mental health concern",    "#d97706"),
        ("SI",    "Suicidal ideation signal", "#dc2626"),
    ]

    # Build confidence bars HTML
    conf_bars = ""
    for code, name, bar_color in LABEL_ORDER:
        p = float(prob_dict.get(code, 0.0))
        conf_bars += (
            f'<div class="mj-conf-row">'
            f'<div class="mj-conf-label">{name}</div>'
            f'<div class="mj-conf-bg">'
            f'<div class="mj-conf-fill" style="width:{p*100:.1f}%;background:{bar_color}"></div>'
            f'</div>'
            f'<div class="mj-conf-pct">{p:.0%}</div>'
            f'</div>'
        )

    # Step 2 card — fully self-contained HTML block (no split open/close)
    st.markdown(
        f"""
<div class="mj-card">
  <div class="mj-step-label">Step 2 of 3 · Analysis result</div>
  <div class="mj-card-title">Model interpretation</div>

  <div class="mj-badge"
       style="background:{label_bg};color:{label_color};border:1px solid {label_color}33">
    {label_icon}&nbsp;&nbsp;{label_text}
  </div>

  <div class="mj-metric-row">
    <div class="mj-pill"><span class="mj-pill-key">Risk tier</span>{risk_tier}</div>
    <div class="mj-pill"><span class="mj-pill-key">Top signal</span>{pred_label}</div>
  </div>

  <div class="mj-conf-title">Confidence across classes</div>
  {conf_bars}
</div>
""",
        unsafe_allow_html=True,
    )

    # Step 3 card — guidance
    MSG_ICONS   = {"crisis": "⛑️", "warn": "⚠️", "info": "💬", "soft": "🌿", "ok": "✅"}
    MSG_CLASSES = {"crisis": "mj-crisis", "warn": "mj-warn", "info": "mj-info",
                   "soft": "mj-soft", "ok": "mj-ok"}
    icon = MSG_ICONS.get(msg_type, "•")
    cls  = MSG_CLASSES.get(msg_type, "mj-ok")

    st.markdown(
        f"""
<div class="mj-card">
  <div class="mj-step-label">Step 3 of 3 · Guidance</div>
  <div class="mj-card-title">Suggested next step</div>
  <div class="mj-msg {cls}">
    <span class="mj-msg-icon">{icon}</span>
    <span>{msg_text}</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    # ── Optional questionnaires ────────────────────────────────────────────────
    show_prompt = decision.get("showphqgadprompt") or decision.get("show_phq_gad_prompt")
    if show_prompt:
        with st.expander("📋  Optional: PHQ-9 & GAD-7 Check-in", expanded=False):
            st.markdown(
                '<p style="font-size:0.92rem;color:#475569;margin-bottom:1rem">'
                "These brief validated questionnaires give a more structured check-in "
                "on mood and anxiety.</p>",
                unsafe_allow_html=True,
            )

            RESP = {0: "Not at all", 1: "Several days",
                    2: "More than half the days", 3: "Nearly every day"}

            PHQ_QS = [
                "Little interest or pleasure in doing things",
                "Feeling down, depressed, or hopeless",
                "Trouble falling or staying asleep, or sleeping too much",
                "Feeling tired or having little energy",
                "Poor appetite or overeating",
                "Feeling bad about yourself, or feeling like a failure",
                "Trouble concentrating on things",
                "Moving or speaking unusually slowly — or being unusually fidgety",
                "Thoughts that you would be better off dead, or of hurting yourself",
            ]
            GAD_QS = [
                "Feeling nervous, anxious, or on edge",
                "Not being able to stop or control worrying",
                "Worrying too much about different things",
                "Trouble relaxing",
                "Being so restless that it is hard to sit still",
                "Becoming easily annoyed or irritable",
                "Feeling afraid as if something awful might happen",
            ]

            with st.form("questionnaire_form"):
                st.markdown(
                    '<div class="mj-q-header">PHQ-9 · Depression screen</div>',
                    unsafe_allow_html=True,
                )
                phq_answers = [
                    st.selectbox(
                        f"{i+1}. {q}",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: RESP[x],
                        key=f"phq_{i}",
                    )
                    for i, q in enumerate(PHQ_QS)
                ]

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    '<div class="mj-q-header">GAD-7 · Anxiety screen</div>',
                    unsafe_allow_html=True,
                )
                gad_answers = [
                    st.selectbox(
                        f"{i+1}. {q}",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: RESP[x],
                        key=f"gad_{i}",
                    )
                    for i, q in enumerate(GAD_QS)
                ]

                submitted = st.form_submit_button(
                    "Submit questionnaires", use_container_width=True
                )

            # Handle submit outside the form block to avoid double-render
            if submitted:
                phq_total, phq_severity = score_phq9(phq_answers)
                gad_total, gad_severity = score_gad7(gad_answers)
                final_decision = decide_next_step(
                    pred_label, prob_dict,
                    phq9_items=phq_answers,
                    gad7_items=gad_answers,
                )
                st.session_state.update(
                    q_submitted=True,
                    phq_total=phq_total,
                    phq_severity=phq_severity,
                    gad_total=gad_total,
                    gad_severity=gad_severity,
                    final_decision=final_decision,
                )

            # Render results (reads from session state, survives re-renders)
            if st.session_state.q_submitted:
                fd     = st.session_state.final_decision
                ph_sev = (st.session_state.phq_severity or "").lower()
                ga_sev = (st.session_state.gad_severity or "").lower()

                def sev_class(s):
                    if "severe"   in s: return "mj-q-sev"
                    if "moderate" in s: return "mj-q-mod"
                    if "mild"     in s: return "mj-q-mild"
                    return "mj-q-min"

                st.markdown('<div class="mj-divider"></div>', unsafe_allow_html=True)
                st.markdown(
                    f"""
<p style="font-size:1rem;font-weight:700;color:#0f172a;margin-bottom:0.6rem">
  Questionnaire results
</p>
<div class="mj-q-result-row">
  <div class="mj-q-pill {sev_class(ph_sev)}">
    PHQ-9: {st.session_state.phq_total} · {ph_sev.capitalize()}
  </div>
  <div class="mj-q-pill {sev_class(ga_sev)}">
    GAD-7: {st.session_state.gad_total} · {ga_sev.capitalize()}
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )

                if fd.get("suicidalityflag") or fd.get("suicidality_flag"):
                    st.error(
                        "You indicated possible self-harm thoughts on PHQ-9 item 9. "
                        "Please seek immediate support. iCall India: **9152987821**"
                    )
                elif fd.get("recommendation") == "seek_professional_help":
                    st.warning(
                        "These results suggest professional mental health support would be advisable."
                    )
                elif fd.get("recommendation") == "consider_professional_help":
                    st.info("These results suggest that talking to a counsellor or therapist may help.")
                else:
                    st.success("Thank you for completing the check-in. Keep taking care of yourself.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="mj-footer">
  Research prototype &nbsp;·&nbsp; Not a clinical tool &nbsp;·&nbsp; Not for emergency use<br>
  If you are in crisis, contact <strong>iCall India: 9152987821</strong>
  or local emergency services.
</div>
""",
    unsafe_allow_html=True,
)
