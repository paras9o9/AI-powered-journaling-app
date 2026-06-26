import streamlit as st
from journal_model import load_model, predict_entry
from decision_logic import decide_next_step, score_phq9, score_gad7

st.set_page_config(
    page_title="AI Journaling Support App",
    page_icon="🧠",
    layout="centered"
)

MODEL_SOURCE = "paras9o9/journal-distilbert-model"
# If using local files instead, use:
# MODEL_SOURCE = "./model"

@st.cache_resource
def get_model():
    tokenizer, model = load_model(
        model_path=MODEL_SOURCE,
        tokenizer_name=MODEL_SOURCE
    )
    return tokenizer, model

tokenizer, model = get_model()

if "analysis_done" not in st.session_state:
    st.session_state.analysis_done = False
if "pred_label" not in st.session_state:
    st.session_state.pred_label = None
if "prob_dict" not in st.session_state:
    st.session_state.prob_dict = None
if "decision" not in st.session_state:
    st.session_state.decision = None

st.title("🧠 AI Journaling Support App")
st.caption("A research prototype for journal-based emotional risk screening.")
st.warning(
    "This app is not a diagnosis tool, not a replacement for a therapist, "
    "and not an emergency service."
)

with st.container():
    st.subheader("Journal entry")
    user_text = st.text_area(
        "Write about how you are feeling today",
        height=220,
        placeholder="Example: I’ve been feeling overwhelmed lately and I’m not sure how to cope..."
    )

    analyse = st.button("Analyse entry", type="primary", use_container_width=True)

if analyse:
    if not user_text.strip():
        st.warning("Please enter a journal entry before analysing.")
    else:
        with st.spinner("Analysing your entry..."):
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

    st.markdown("---")
    st.subheader("Model result")

    label_map = {
        "NEU": "🟢 Neutral",
        "HUMOR": "😄 Humor",
        "MH": "🟡 Mental health concern",
        "SI": "🔴 Suicidal ideation concern"
    }

    st.write(f"**Predicted class:** {label_map.get(pred_label, pred_label)}")
    st.write(f"**Risk tier:** {decision['risk_tier'].capitalize()}")

    st.write("**Class probabilities**")
    ordered_labels = ["NEU", "HUMOR", "MH", "SI"]
    for label in ordered_labels:
        p = float(prob_dict.get(label, 0.0))
        st.progress(p, text=f"{label}: {p:.2%}")

    st.markdown("---")
    st.subheader("Support guidance")

    recommendation = decision.get("recommendation", "general_positive_feedback")

    if decision.get("showcrisiscard") or decision.get("show_crisis_card"):
        st.error(
            "⚠️ This entry may reflect significant distress.

"
            "If you may be in immediate danger or may act on self-harm thoughts, "
            "contact local emergency services now or reach out to a trusted person immediately.

"
            "India crisis support: iCall — 9152987821"
        )
    elif recommendation == "seek_professional_help":
        st.warning(
            "Your responses suggest that speaking with a mental health professional would be a good next step."
        )
    elif recommendation == "consider_professional_help":
        st.info(
            "It may help to talk with a counsellor, therapist, or a trusted support person."
        )
    elif recommendation == "monitorandselfcare" or recommendation == "monitor_and_self_care":
        st.info(
            "You may be going through a difficult period. Consider self-care, rest, and checking in with someone you trust."
        )
    else:
        st.success(
            "Thanks for sharing. The app did not detect high concern from this entry, but your feelings still matter."
        )

    show_prompt = decision.get("showphqgadprompt") or decision.get("show_phq_gad_prompt")
    if show_prompt:
        st.markdown("---")
        st.subheader("Optional questionnaires")
        st.write(
            "You can complete the PHQ-9 and GAD-7 for a more structured mood and anxiety check-in."
        )

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

            phq_answers = []
            for i, q in enumerate(phq_questions):
                val = st.selectbox(
                    f"{i+1}. {q}",
                    options=[0, 1, 2, 3],
                    format_func=lambda x: response_options[x],
                    key=f"phq_{i}"
                )
                phq_answers.append(val)

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

            st.markdown("---")
            st.subheader("Questionnaire results")
            st.write(f"**PHQ-9 score:** {phq_total} — {phq_severity.capitalize()}")
            st.write(f"**GAD-7 score:** {gad_total} — {gad_severity.capitalize()}")

            if final_decision.get("suicidalityflag") or final_decision.get("suicidality_flag"):
                st.error(
                    "⚠️ You indicated possible self-harm thoughts on PHQ-9 item 9. "
                    "Please seek immediate support from a trusted person, local emergency service, "
                    "or iCall India at 9152987821."
                )
            elif final_decision.get("recommendation") == "seek_professional_help":
                st.warning("These results suggest that professional mental health support would be advisable.")
            elif final_decision.get("recommendation") == "consider_professional_help":
                st.info("These results suggest that talking to a counsellor or therapist may help.")
            else:
                st.success("Thank you for completing the check-in.")

st.markdown("---")
st.caption(
    "For deployment: store the model on Hugging Face Hub or load it from a local ./model folder."
)