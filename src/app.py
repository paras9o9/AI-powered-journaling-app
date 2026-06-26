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

for key, default in {
    "analysis_done": False,
    "pred_label": None,
    "prob_dict": None,
    "decision": None,
    "user_text": ""
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

st.markdown("""
<style>
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 820px;
}
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(180deg, #f6f8fb 0%, #eef4f7 100%);
}
.hero-card, .section-card, .result-card {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 22px;
    padding: 1.25rem 1.25rem;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    margin-bottom: 1rem;
}
.hero-title {
    font-size: 2rem;
    font-weight: 700;
    color: #0f172a;
    margin-bottom: 0.35rem;
}
.hero-sub {
    color: #475569;
    font-size: 0.98rem;
    line-height: 1.6;
}
.chip {
    display: inline-block;
    padding: 0.35rem 0.75rem;
    border-radius: 999px;
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 0.75rem;
    background: #fff7ed;
    color: #9a3412;
    border: 1px solid #fdba74;
}
.step-label {
    font-size: 0.82rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #64748b;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-badge {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.8rem;
}
.badge-green { background: #dcfce7; color: #166534; }
.badge-yellow { background: #fef3c7; color: #92400e; }
.badge-red { background: #fee2e2; color: #991b1b; }
.badge-blue { background: #dbeafe; color: #1d4ed8; }

div[data-testid="stTextArea"] textarea {
    border-radius: 18px !important;
    border: 1px solid #cbd5e1 !important;
    background: #fcfdff !important;
    padding: 1rem !important;
    font-size: 1rem !important;
}
div[data-testid="stTextArea"] textarea:focus {
    border-color: #7c3aed !important;
    box-shadow: 0 0 0 3px rgba(124, 58, 237, 0.12) !important;
}
.stButton > button {
    border-radius: 14px !important;
    padding: 0.75rem 1rem !important;
    font-weight: 700 !important;
    border: none !important;
}
.metric-row {
    display: flex;
    gap: 0.75rem;
    flex-wrap: wrap;
    margin-top: 0.5rem;
    margin-bottom: 0.5rem;
}
.metric-pill {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #334155;
    padding: 0.5rem 0.8rem;
    border-radius: 14px;
    font-size: 0.9rem;
}
.small-note {
    color: #64748b;
    font-size: 0.9rem;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-card">
    <div class="hero-title">🧠 AI Journaling Support</div>
    <div class="hero-sub">
        Reflect on your thoughts in a private space and receive supportive, research-oriented feedback.
    </div>
    <div class="chip">Not a diagnosis tool • Not for emergency use</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="step-label">Step 1 • Journal entry</div>', unsafe_allow_html=True)

user_text = st.text_area(
    "Write about how you are feeling today",
    height=220,
    placeholder="Take your time. You can write freely about stress, emotions, worries, or anything that feels important today.",
    label_visibility="collapsed",
    key="user_text"
)

col_a, col_b = st.columns([3, 1])
with col_a:
    analyse = st.button("Analyse entry", type="primary", use_container_width=True)
with col_b:
    reset = st.button("Reset", use_container_width=True)

word_count = len(user_text.split()) if user_text.strip() else 0
st.markdown(
    f'<div class="small-note">Word count: {word_count} • Tip: writing 2–5 sentences usually gives more useful feedback.</div>',
    unsafe_allow_html=True
)
st.markdown('</div>', unsafe_allow_html=True)

if reset:
    st.session_state.analysis_done = False
    st.session_state.pred_label = None
    st.session_state.prob_dict = None
    st.session_state.decision = None
    st.session_state.user_text = ""
    st.rerun()

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

if st.session_state.analysis_done:
    pred_label = st.session_state.pred_label
    prob_dict = st.session_state.prob_dict
    decision = st.session_state.decision

    label_map = {
        "NEU": ("🟢 Neutral", "badge-green"),
        "HUMOR": ("😄 Humor", "badge-blue"),
        "MH": ("🟡 Mental health concern", "badge-yellow"),
        "SI": ("🔴 Suicidal ideation concern", "badge-red")
    }

    badge_text, badge_class = label_map.get(pred_label, (pred_label, "badge-blue"))

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-label">Step 2 • Analysis result</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="result-badge {badge_class}">{badge_text}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-pill"><strong>Risk tier:</strong> {decision['risk_tier'].capitalize()}</div>
            <div class="metric-pill"><strong>Top signal:</strong> {pred_label}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("### Confidence across classes")
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

    st.markdown('<div class="result-card">', unsafe_allow_html=True)
    st.markdown('<div class="step-label">Step 3 • Support guidance</div>', unsafe_allow_html=True)

    recommendation = decision.get("recommendation", "general_positive_feedback")

    if decision.get("showcrisiscard") or decision.get("show_crisis_card"):
        st.error(
            "This entry may reflect significant distress. If you may be in immediate danger or may act on self-harm thoughts, contact local emergency services now or reach out to a trusted person immediately. India crisis support: iCall — 9152987821"
        )
    elif recommendation == "seek_professional_help":
        st.warning("Speaking with a mental health professional would be a good next step.")
    elif recommendation == "consider_professional_help":
        st.info("It may help to talk with a counsellor, therapist, or a trusted support person.")
    elif recommendation in ["monitorandselfcare", "monitor_and_self_care"]:
        st.info("You may be going through a difficult period. Consider rest, self-care, and checking in with someone you trust.")
    else:
        st.success("Thanks for sharing. The app did not detect high concern from this entry, but your feelings still matter.")

    st.markdown('</div>', unsafe_allow_html=True)

    show_prompt = decision.get("showphqgadprompt") or decision.get("show_phq_gad_prompt")
    if show_prompt:
        with st.expander("Optional mood and anxiety questionnaires", expanded=False):
            st.write("You can complete PHQ-9 and GAD-7 for a more structured check-in.")

            response_options = {
                0: "Not at all",
                1: "Several days",
                2: "More than half the days",
                3: "Nearly every day"
            }

            with st.form("questionnaire_form"):
                st.markdown("### PHQ-9")
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
                phq_answers = [
                    st.selectbox(
                        f"{i+1}. {q}",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: response_options[x],
                        key=f"phq_{i}"
                    )
                    for i, q in enumerate(phq_questions)
                ]

                st.markdown("### GAD-7")
                gad_questions = [
                    "Feeling nervous, anxious, or on edge",
                    "Not being able to stop or control worrying",
                    "Worrying too much about different things",
                    "Trouble relaxing",
                    "Being so restless that it is hard to sit still",
                    "Becoming easily annoyed or irritable",
                    "Feeling afraid as if something awful might happen"
                ]
                gad_answers = [
                    st.selectbox(
                        f"{i+1}. {q}",
                        options=[0, 1, 2, 3],
                        format_func=lambda x: response_options[x],
                        key=f"gad_{i}"
                    )
                    for i, q in enumerate(gad_questions)
                ]

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

                st.write(f"**PHQ-9 score:** {phq_total} — {phq_severity.capitalize()}")
                st.write(f"**GAD-7 score:** {gad_total} — {gad_severity.capitalize()}")

                if final_decision.get("suicidalityflag") or final_decision.get("suicidality_flag"):
                    st.error("You indicated possible self-harm thoughts on PHQ-9 item 9. Please seek immediate support from a trusted person, local emergency service, or iCall India at 9152987821.")
                elif final_decision.get("recommendation") == "seek_professional_help":
                    st.warning("These results suggest that professional mental health support would be advisable.")
                elif final_decision.get("recommendation") == "consider_professional_help":
                    st.info("These results suggest that talking to a counsellor or therapist may help.")
                else:
                    st.success("Thank you for completing the check-in.")
