import streamlit as st
from journal_model import load_model, predict_entry
from decision_logic import decide_next_step, score_phq9, score_gad7

st.set_page_config(page_title="AI Journaling Support App", page_icon="🧠", layout="centered")

st.markdown("""
<style>
.block-container {
    max-width: 820px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}
.stApp {
    background: linear-gradient(180deg, #f8fafc 0%, #eef4f7 100%);
}
h1, h2, h3, p, label {
    color: #0f172a;
}
div[data-testid="stTextArea"] textarea,
div[data-testid="stTextInput"] input {
    border-radius: 18px !important;
    border: 1px solid #cbd5e1 !important;
    background: #ffffff !important;
    padding: 1rem !important;
}
div[data-testid="stButton"] > button {
    border-radius: 14px !important;
    font-weight: 700 !important;
    min-height: 46px;
}
div[data-testid="stExpander"] details {
    border-radius: 18px !important;
    border: 1px solid #e2e8f0 !important;
    background: rgba(255,255,255,0.9) !important;
}
.result-shell, .hero-shell {
    background: rgba(255,255,255,0.88);
    border: 1px solid rgba(15, 23, 42, 0.08);
    border-radius: 22px;
    padding: 1.25rem;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.06);
    margin-bottom: 1rem;
}
.small-note {
    color: #64748b;
    font-size: 0.92rem;
}
.badge {
    display: inline-block;
    padding: 0.45rem 0.8rem;
    border-radius: 999px;
    font-weight: 700;
    margin-bottom: 0.75rem;
}
.badge-green { background:#dcfce7; color:#166534; }
.badge-yellow { background:#fef3c7; color:#92400e; }
.badge-red { background:#fee2e2; color:#991b1b; }
.badge-blue { background:#dbeafe; color:#1d4ed8; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero-shell">
    <h1 style="margin-bottom:0.35rem;">🧠 AI Journaling Support</h1>
    <p style="margin-bottom:0.6rem; color:#475569;">
        Reflect on your thoughts in a private space and receive supportive, research-oriented feedback.
    </p>
    <div style="display:inline-block;padding:0.35rem 0.75rem;border-radius:999px;
                background:#fff7ed;color:#9a3412;border:1px solid #fdba74;font-size:0.84rem;font-weight:600;">
        Not a diagnosis tool • Not for emergency use
    </div>
</div>
""", unsafe_allow_html=True)

with st.container():
    st.markdown("### Step 1 • Journal entry")
    user_text = st.text_area(
        "Write about how you are feeling today",
        height=220,
        placeholder="Take your time. You can write freely about stress, emotions, worries, or anything that feels important today.",
        label_visibility="collapsed"
    )

    c1, c2 = st.columns([3, 1])
    with c1:
        analyse = st.button("Analyse entry", type="primary", use_container_width=True)
    with c2:
        reset = st.button("Reset", use_container_width=True)

    st.caption(f"Word count: {len(user_text.split()) if user_text.strip() else 0}")
