import streamlit as st
from journal_model import load_model, predict_entry
from decision_logic import decide_next_step, score_phq9, score_gad7

st.set_page_config(
    page_title="AI Journaling Support App",
    page_icon="🧠",
    layout="centered"
)

MODEL_SOURCE = "paras9o9/journal-distilbert-model"

@st.cache_resource
def get_model():
    tokenizer, model = load_model(
        model_path=MODEL_SOURCE,
        tokenizer_name=MODEL_SOURCE
    )
    return tokenizer, model

tokenizer, model = get_model()

# -----------------------------
# Session state
# -----------------------------
defaults = {
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
    "final_decision": None
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -----------------------------
# Helpers
# -----------------------------
def reset_app():
    for key, value in defaults.items():
        st.session_state[key] = value
    st.rerun()

def get_badge_info(pred_label):
    label_map = {
        "NEU": ("🟢 Neutral", "green"),
        "HUMOR": ("😄 Humor", "blue"),
        "MH": ("🟡 Mental health concern", "yellow"),
        "SI": ("🔴 Suicidal ideation concern", "red")
    }
    return label_map.get(pred_label, (pred_label, "blue"))

def get_support_message(decision):
    recommendation = decision.get("recommendation", "general_positive_feedback")

    if decision.get("showcrisiscard") or decision.get("show_crisis_card"):
        return (
            "error",
            "This entry may reflect significant distress. "
            "If you may be in immediate danger or may act on self-harm thoughts, "
            "contact local emergency services now or reach out to a trusted person immediately. "
            "India crisis support: iCall — 9152987821"
        )
    elif recommendation == "seek_professional_help":
        return (
            "warning",
            "Speaking with a mental health professional would be a good next step."
        )
    elif recommendation == "consider_professional_help":
        return (
            "info",
            "It may help to talk with a counsellor, therapist, or a trusted support person."
        )
    elif recommendation in ["monitorandselfcare", "monitor_and_self_care"]:
        return (
            "info",
            "You may be going through a difficult period. Consider rest, self-care, and checking in with someone you trust."
        )
    else:
        return (
            "success",
            "Thanks for sharing. The app did not detect high concern from this entry, but your feelings still matter."
        )

def render_status_message(msg_type, text):
    if msg_type == "error":
        st.error(text)
    elif msg_type == "warning":
        st.warning(text)
    elif msg_type == "info":
        st.info(text)
    else:
        st.success(text)

# -----------------------------
# Styling
st.markdown("""
<style>
:root {
    --bg-grad-1: #f0f4ff;
    --bg-grad-2: #e8f0fe;
    --card-bg: rgba(255, 255, 255, 0.92);
    --card-border: rgba(15, 23, 42, 0.08);
    --text-main: #0f172a;
    --text-soft: #475569;
    --text-muted: #64748b;
    --primary: #4f46e5;
    --primary-dark: #4338ca;
    --success-bg: #dcfce7;
    --success-text: #166534;
    --warn-bg: #fef3c7;
    --warn-text: #92400e;
    --danger-bg: #fee2e2;
    --danger-text: #991b1b;
    --info-bg: #dbeafe;
    --info-text: #1d4ed8;
}

html, body, [class*="css"] {
    font-family: "Inter", sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at top left, rgba(99, 102, 241, 0.06), transparent 40%),
        radial-gradient(circle at bottom right, rgba(59, 130, 246, 0.05), transparent 40%),
        linear-gradient(180deg, #e8eef8 0%, #dde7f2 100%);
}

.block-container {
    max-width: 860px;
    padding-top: 1.8rem;
    padding-bottom: 3rem;
}

.main-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: var(--text-main);
    margin-bottom: 0.35rem;
    line-height: 1.2;
}

.subtle-text {
    color: var(--text-soft);
    font-size: 1rem;
    line-height: 1.65;
}

.disclaimer-chip {
    display: inline-block;
    margin-top: 0.9rem;
    padding: 0.42rem 0.85rem;
    border-radius: 999px;
    border: 1px solid #fdba74;
    background: #fff7ed;
    color: #9a3412;
    font-size: 0.82rem;
    font-weight: 700;
}

.section-heading {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: var(--primary);
    font-weight: 800;
    margin-bottom: 0.3rem;
}

.block-card {
    background: var(--card-bg);
    border: 1px solid var(--card-border);
    border-radius: 22px;
    padding: 1.4rem 1.4rem 1.1rem 1.4rem;
    box-shadow: 0 4px 6px rgba(15, 23, 42, 0.04),
                0 10px 30px rgba(15, 23, 42, 0.07);
    margin-bottom: 1.1rem;
    backdrop-filter: blur(12px);
}

.mini-note {
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.55rem;
    padding: 0.5rem 0.75rem;
    background: #f8fafc;
    border-radius: 10px;
    border: 1px solid #e8edf3;
}

.badge {
    display: inline-block;
    padding: 0.45rem 1rem;
    border-radius: 999px;
    font-size: 0.9rem;
    font-weight: 800;
    margin: 0.4rem 0 0.8rem 0;
    letter-spacing: 0.01em;
}

.badge.green {
    background: var(--success-bg);
    color: var(--success-text);
    border: 1px solid #86efac;
}

.badge.yellow {
    background: var(--warn-bg);
    color: var(--warn-text);
    border: 1px solid #fcd34d;
}

.badge.red {
    background: var(--danger-bg);
    color: var(--danger-text);
    border: 1px solid #fca5a5;
}

.badge.blue {
    background: var(--info-bg);
    color: var(--info-text);
    border: 1px solid #93c5fd;
}

.metric-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 0.65rem;
    margin: 0.3rem 0 0.9rem 0;
}

.metric-pill {
    background: #f1f5f9;
    color: #1e293b;
    border: 1px solid #dde3ec;
    border-radius: 12px;
    padding: 0.5rem 0.85rem;
    font-size: 0.91rem;
    font-weight: 600;
}

div[data-testid="stTextArea"] textarea {
    border-radius: 16px !important;
    border: 1.5px solid #64748b !important;
    background: #cbd5e1 !important;
    color: #0f172a !important;
    caret_color: black;
    padding: 1rem !important;
    font-size: 1rem !important;
    line-height: 1.65 !important;
    font-weight: 500 !important;
}

div[data-testid="stTextArea"] textarea:focus {
    border-color: #4338ca !important;
    background: #bac7d8 !important;
    color: #020617 !important;
    box-shadow: 0 0 0 3px rgba(67, 56, 202, 0.16) !important;
}

div[data-testid="stTextArea"] textarea::placeholder {
    color: #334155 !important;
    opacity: 1 !important;
}

div[data-testid="stButton"] > button {
    border-radius: 14px !important;
    min-height: 46px !important;
    font-weight: 800 !important;
    font-size: 0.97rem !important;
    letter-spacing: 0.01em !important;
    border: none !important;
    transition: all 0.18s ease !important;
}

div[data-testid="stBaseButton-primary"] > button {
    background: linear-gradient(135deg, #6366f1, #4f46e5) !important;
    color: #ffffff !important;
    box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35) !important;
}

div[data-testid="stBaseButton-primary"] > button:hover {
    filter: brightness(1.07) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

div[data-testid="stBaseButton-secondary"] > button {
    background: #ffffff !important;
    color: #334155 !important;
    border: 1.5px solid #dde3ec !important;
}

div[data-testid="stBaseButton-secondary"] > button:hover {
    background: #f8fafc !important;
    border-color: #cbd5e1 !important;
}

div[data-testid="stExpander"] details {
    background: rgba(255, 255, 255, 0.95) !important;
    border: 1px solid #dde3ec !important;
    border-radius: 18px !important;
    padding: 0.2rem 0.4rem !important;
    box-shadow: 0 4px 12px rgba(15, 23, 42, 0.05) !important;
}

div[data-testid="stExpander"] summary {
    font-weight: 700 !important;
    color: #1e293b !important;
    font-size: 0.97rem !important;
}

div[data-testid="stSelectbox"] > div {
    border-radius: 12px !important;
}

div[data-testid="stAlert"] {
    border-radius: 16px !important;
}

div[data-testid="stProgressBar"] > div {
    border-radius: 999px !important;
}

hr {
    border: none;
    height: 1px;
    background: linear-gradient(90deg, transparent, #c7d2e8, transparent);
    margin: 1.1rem 0 0.5rem 0;
}

.result-label {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-main);
    margin-bottom: 0.2rem;
}

.helper-copy {
    color: var(--text-soft);
    font-size: 0.96rem;
    line-height: 1.65;
    margin-bottom: 0.5rem;
}

.question-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--text-main);
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
    padding-bottom: 0.35rem;
    border-bottom: 2px solid #e8edf3;
}

.footer-note {
    text-align: center;
    color: var(--text-muted);
    font-size: 0.85rem;
    margin-top: 1.5rem;
    padding-top: 1rem;
    border-top: 1px solid #e8edf3;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------
# Hero
# -----------------------------
st.markdown('<div class="block-card">', unsafe_allow_html=True)
st.markdown('<div class="main-title">🧠 AI Journaling Support</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtle-text">Reflect on your thoughts in a private space and receive supportive, research-oriented feedback based on your journal entry.</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="disclaimer-chip">Not a diagnosis tool • Not for emergency use</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Entry section
# -----------------------------
entry_container = st.container()
with entry_container:
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Step 1 • Journal entry</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="helper-copy">Take a breath and write freely about how your day feels, what is weighing on you, or what thoughts keep returning.</div>',
        unsafe_allow_html=True
    )

    user_text = st.text_area(
        "Write about how you are feeling today",
        height=220,
        placeholder="Example: I’ve been feeling emotionally tired lately. I’m trying to manage things, but some days feel heavier than usual.",
        label_visibility="collapsed",
        key="user_text"
    )

    col1, col2 = st.columns([3, 1])
    with col1:
        analyse = st.button("Analyse entry", type="primary", use_container_width=True)
    with col2:
        reset = st.button("Reset", use_container_width=True)

    word_count = len(user_text.split()) if user_text.strip() else 0
    st.markdown(
        f'<div class="mini-note">Word count: {word_count} • Writing 2–5 sentences usually gives more useful feedback.</div>',
        unsafe_allow_html=True
    )
    st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Reset
# -----------------------------
if reset:
    reset_app()

# -----------------------------
# Analyse
# -----------------------------
if analyse:
    if not user_text.strip():
        st.warning("Please enter a journal entry before analysing.")
    else:
        with st.spinner("Reading your entry and preparing feedback..."):
            pred_label, prob_dict = predict_entry(user_text, tokenizer, model)
            decision = decide_next_step(pred_label, prob_dict)

        st.session_state.analysis_done = True
        st.session_state.pred_label = pred_label
        st.session_state.prob_dict = prob_dict
        st.session_state.decision = decision
        st.session_state.q_submitted = False
        st.session_state.phq_total = None
        st.session_state.phq_severity = None
        st.session_state.gad_total = None
        st.session_state.gad_severity = None
        st.session_state.final_decision = None

# -----------------------------
# Results
# -----------------------------
if st.session_state.analysis_done:
    pred_label = st.session_state.pred_label
    prob_dict = st.session_state.prob_dict
    decision = st.session_state.decision

    badge_text, badge_color = get_badge_info(pred_label)

    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Step 2 • Analysis result</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="badge {badge_color}">{badge_text}</div>', unsafe_allow_html=True)
    st.markdown('<div class="result-label">Current model interpretation</div>', unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="metric-strip">
            <div class="metric-pill">Risk tier: {decision['risk_tier'].capitalize()}</div>
            <div class="metric-pill">Top signal: {pred_label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="helper-copy">These confidence bars show how strongly the model leaned toward each class for this journal entry.</div>',
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.write("#### Confidence across classes")

    ordered_labels = ["NEU", "HUMOR", "MH", "SI"]
    friendly_names = {
        "NEU": "Neutral",
        "HUMOR": "Humor",
        "MH": "Mental health concern",
        "SI": "Suicidal ideation concern"
    }

    for label in ordered_labels:
        p = float(prob_dict.get(label, 0.0))
        st.progress(p, text=f"{friendly_names[label]} • {p:.1%}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Support section
    st.markdown('<div class="block-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Step 3 • Support guidance</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="helper-copy">This section suggests a reasonable next step based on the current screening result.</div>',
        unsafe_allow_html=True
    )

    msg_type, msg_text = get_support_message(decision)
    render_status_message(msg_type, msg_text)

    st.markdown('</div>', unsafe_allow_html=True)

    # Questionnaires
    show_prompt = decision.get("showphqgadprompt") or decision.get("show_phq_gad_prompt")
    if show_prompt:
        with st.expander("Optional mood and anxiety questionnaires", expanded=False):
            st.markdown(
                '<div class="helper-copy">You can complete PHQ-9 and GAD-7 for a more structured mood and anxiety check-in.</div>',
                unsafe_allow_html=True
            )

            response_options = {
                0: "Not at all",
                1: "Several days",
                2: "More than half the days",
                3: "Nearly every day"
            }

            with st.form("questionnaire_form"):
                st.markdown('<div class="question-title">PHQ-9</div>', unsafe_allow_html=True)
                phq_questions = [
                    "Little interest or pleasure in doing things",
                    "Feeling down, depressed, or hopeless",
                    "Trouble falling or staying asleep, or sleeping too much",
                    "Feeling tired or having little energy",
                    "Poor appetite or overeating",
                    "Feeling bad about yourself — or that you are a failure or have let yourself or your family down",
                    "Trouble concentrating on things",
                    "Moving or speaking so slowly that other people could have noticed, or the opposite — being so fidgety or restless",
                    "Thoughts that you would be better off dead or of hurting yourself"
                ]

                phq_answers = []
                for i, q in enumerate(phq_questions):
                    val = st.selectbox(
                        f"{i+1}. {q}",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: response_options[x],
                        key=f"phq_{i}"
                    )
                    phq_answers.append(val)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<div class="question-title">GAD-7</div>', unsafe_allow_html=True)
                gad_questions = [
                    "Feeling nervous, anxious, or on edge",
                    "Not being able to stop or control worrying",
                    "Worrying too much about different things",
                    "Trouble relaxing",
                    "Being so restless that it is hard to sit still",
                    "Becoming easily annoyed or irritable",
                    "Feeling afraid as if something awful might happen"
                ]

                gad_answers = []
                for i, q in enumerate(gad_questions):
                    val = st.selectbox(
                        f"{i+1}. {q}",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: response_options[x],
                        key=f"gad_{i}"
                    )
                    gad_answers.append(val)

                submit_q = st.form_submit_button("Submit questionnaires", use_container_width=True)

            if submit_q:
                phq_total, phq_severity = score_phq9(phq_answers)
                gad_total, gad_severity = score_gad7(gad_answers)

                final_decision = decide_next_step(
                    pred_label,
                    prob_dict,
                    phq9_items=phq_answers,
                    gad7_items=gad_answers
                )

                st.session_state.q_submitted = True
                st.session_state.phq_total = phq_total
                st.session_state.phq_severity = phq_severity
                st.session_state.gad_total = gad_total
                st.session_state.gad_severity = gad_severity
                st.session_state.final_decision = final_decision

            if st.session_state.q_submitted:
                final_decision = st.session_state.final_decision

                st.markdown("<hr>", unsafe_allow_html=True)
                st.write("#### Questionnaire results")
                st.markdown(
                    f"""
                    <div class="metric-strip">
                        <div class="metric-pill">PHQ-9: {st.session_state.phq_total} — {st.session_state.phq_severity.capitalize()}</div>
                        <div class="metric-pill">GAD-7: {st.session_state.gad_total} — {st.session_state.gad_severity.capitalize()}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if final_decision.get("suicidalityflag") or final_decision.get("suicidality_flag"):
                    st.error(
                        "You indicated possible self-harm thoughts on PHQ-9 item 9. "
                        "Please seek immediate support from a trusted person, local emergency service, "
                        "or iCall India at 9152987821."
                    )
                elif final_decision.get("recommendation") == "seek_professional_help":
                    st.warning(
                        "These results suggest that professional mental health support would be advisable."
                    )
                elif final_decision.get("recommendation") == "consider_professional_help":
                    st.info(
                        "These results suggest that talking to a counsellor or therapist may help."
                    )
                else:
                    st.success("Thank you for completing the check-in.")

st.markdown(
    '<div class="footer-note">Research prototype for journal-based emotional risk screening.</div>',
    unsafe_allow_html=True
)
