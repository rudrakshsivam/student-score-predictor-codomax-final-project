"""
Student Score Predictor
A complete machine learning project:
  1. Dashboard      - explore the data
  2. Model          - compare 3 regression models with cross-validation
  3. Predict        - estimate a score from study hours + AI study plan (Groq API)
  4. AI Chat       - chat with an AI assistant by typing or speaking (Groq API),
                      and hear the answers read aloud
"""
import hashlib
import json
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

# ---------------- CONFIG ----------------
DATA_PATH = "data/student_scores.csv"
X_COL, Y_COL = "Hours", "Scores"
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
# Paste your Groq key between the quotes to use it on your own PC.
# WARNING: remove it again before uploading app.py to GitHub!
API_KEY = ""
GROQ_STT_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_MODEL = "llama-3.3-70b-versatile"   # change if Groq retires this model name
GROQ_STT_MODEL = "whisper-large-v3-turbo"  # speech-to-text model for voice input
BLUE, RED, GREY = "#2563eb", "#ef4444", "#94a3b8"
# ----------------------------------------

st.set_page_config(page_title="Student Score Predictor", page_icon="🎓", layout="wide")


# ---------------- Data & models ----------------
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).drop_duplicates().dropna()
    return df.reset_index(drop=True)


def get_models():
    return {
        "Linear Regression": LinearRegression(),
        "Polynomial (degree 2)": make_pipeline(PolynomialFeatures(2), LinearRegression()),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    }


@st.cache_data
def compare_models(df):
    X, y = df[[X_COL]], df[Y_COL]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in get_models().items():
        res = cross_validate(model, X, y, cv=cv, scoring=("r2", "neg_mean_absolute_error"))
        rows.append(
            {
                "Model": name,
                "R² (5-fold CV)": round(res["test_r2"].mean(), 3),
                "MAE (marks)": round(-res["test_neg_mean_absolute_error"].mean(), 2),
            }
        )
    return pd.DataFrame(rows)


@st.cache_data
def cv_predictions(df):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    return cross_val_predict(LinearRegression(), df[[X_COL]], df[Y_COL], cv=cv)


@st.cache_resource
def train_final_model(df):
    return LinearRegression().fit(df[[X_COL]], df[Y_COL])


# ---------------- Groq AI ----------------
def ask_groq(messages, api_key):
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": 0.5, "max_tokens": 600},
        timeout=40,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Groq error {response.status_code}: {response.text[:200]}")
    return response.json()["choices"][0]["message"]["content"]


SYSTEM_PROMPT = (
    "You are a friendly, concise study coach and data assistant. "
    "Use only the data you are given. Scores are estimates, never promises."
)


def transcribe(audio_bytes, api_key):
    """Turn recorded speech into text with Groq's Whisper model."""
    response = requests.post(
        GROQ_STT_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        files={"file": ("speech.wav", audio_bytes, "audio/wav")},
        data={"model": GROQ_STT_MODEL, "response_format": "json"},
        timeout=60,
    )
    if response.status_code != 200:
        raise RuntimeError(f"Speech-to-text error {response.status_code}: {response.text[:200]}")
    return response.json()["text"].strip()


def speak_controls(text, auto=False):
    """Browser text-to-speech: a Speak button and a Stop button (no key needed)."""
    clean = re.sub(r"[*#`_>|~]", "", text)
    safe = json.dumps(clean)
    auto_js = "speak();" if auto else ""
    components.html(
        f"""
        <button onclick="speak()" style="padding:6px 14px;border-radius:8px;border:1px solid #cbd5e1;
            background:white;cursor:pointer;font-size:14px">🔊 Speak reply</button>
        <button onclick="window.speechSynthesis.cancel()" style="padding:6px 14px;border-radius:8px;
            border:1px solid #cbd5e1;background:white;cursor:pointer;font-size:14px">⏹ Stop</button>
        <script>
        function speak() {{
            window.speechSynthesis.cancel();
            const u = new SpeechSynthesisUtterance({safe});
            u.rate = 1;
            window.speechSynthesis.speak(u);
        }}
        {auto_js}
        </script>
        """,
        height=48,
    )


HOURS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", re.I)
FALLBACK = (
    "I can answer questions about predictions (for example *\"What score for 6 hours?\"*), "
    "the equation, correlation, accuracy and averages. Add a Groq API key to unlock "
    "open-ended answers and study advice."
)


def predict_score(hours):
    return float(np.clip(model.predict(pd.DataFrame({X_COL: [hours]}))[0], 0, 100))


def local_answer(question):
    """Built-in answers computed from the data and model. Works with no API key."""
    q = question.lower()
    m = HOURS_RE.search(q)
    if m:
        hours = float(m.group(1))
        note = (
            ""
            if h_min <= hours <= h_max
            else f" (this is outside the {h_min:.1f}-{h_max:.1f} hour range of the data, so it is less reliable)"
        )
        return f"Studying **{hours:g} hours** gives a predicted score of about **{predict_score(hours):.0f}/100**{note}."
    if any(w in q for w in ["equation", "formula", "slope"]):
        return f"The model is **Score ≈ {slope:.2f} × Hours + {intercept:.2f}**, so each extra hour adds about {slope:.0f} marks."
    if any(w in q for w in ["correlation", "relationship", "related"]):
        return f"Hours and scores are strongly related: the correlation is **{df[X_COL].corr(df[Y_COL]):.2f}**."
    if any(w in q for w in ["best model", "accura", "r2", "r²", "error", "compare"]):
        best = results.sort_values("R² (5-fold CV)").iloc[-1]
        return (
            f"The best model is **{best['Model']}** with R² = {best['R² (5-fold CV)']} "
            f"and an average error of about {best['MAE (marks)']} marks."
        )
    if any(w in q for w in ["average", "mean"]):
        return f"On average students studied **{df[X_COL].mean():.2f} hours** and scored **{df[Y_COL].mean():.1f}**."
    if any(w in q for w in ["how many", "dataset", "students", "data"]):
        return f"The dataset has **{len(df)} students** with the columns {X_COL} and {Y_COL}."
    if any(w in q for w in ["hello", "hi ", "hey"]) or q.strip() in ("hi", "hello", "hey"):
        return "Hello! Ask me about the data, or try *\"What score for 7 hours?\"*"
    return None


def chat_reply(history, api_key):
    """Answer the latest user message: built-in logic first, Groq for open-ended questions."""
    question = history[-1]["content"]
    fact = local_answer(question)
    if not api_key:
        return fact or FALLBACK
    context = (
        f"Dataset ({len(df)} students), columns {X_COL} and {Y_COL}:\n{df.to_csv(index=False)}\n"
        f"Model: Score = {slope:.2f} * Hours + {intercept:.2f}\n"
        f"Cross-validation results:\n{results.to_string(index=False)}"
    )
    system = SYSTEM_PROMPT + "\n\n" + context
    if fact:
        system += f"\n\nVerified model fact for the latest question: {fact}"
    try:
        return ask_groq([{"role": "system", "content": system}] + history[-8:], api_key)
    except Exception as e:
        return (fact + "\n\n" if fact else "") + f"⚠️ AI request failed: {e}"


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25)


# ---------------- App ----------------
try:
    df = load_data()
except FileNotFoundError:
    st.error(f"Dataset not found. Place your CSV at `{DATA_PATH}`.")
    st.stop()

model = train_final_model(df)
slope, intercept = float(model.coef_[0]), float(model.intercept_)
results = compare_models(df)
h_min, h_max = float(df[X_COL].min()), float(df[X_COL].max())

def get_api_key():
    """Look for the key in API_KEY above, then Streamlit secrets, then an environment variable."""
    if API_KEY.strip():
        return API_KEY.strip()
    try:
        key = st.secrets["GROQ_API_KEY"]
        if key:
            return str(key)
    except Exception:
        pass
    return os.getenv("GROQ_API_KEY")


st.sidebar.header("🔑 Groq API key")
api_key = get_api_key()
if api_key:
    st.sidebar.success("API key loaded ✅ AI features are on")
else:
    api_key = st.sidebar.text_input("Paste your key", type="password")
    st.sidebar.caption("Needed only for the AI features. The key is never saved.")
st.sidebar.markdown("---")
st.sidebar.caption(f"Dataset: {len(df)} students · columns: {X_COL}, {Y_COL}")

st.title("🎓 Student Score Predictor")
st.caption("Predict exam scores from study hours, compare ML models, and get an AI study plan")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🧠 Model", "🎯 Predict", "💬 AI Chat"])

# ---------- Tab 1: Dashboard ----------
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Students", len(df))
    c2.metric("Average hours", f"{df[X_COL].mean():.2f}")
    c3.metric("Average score", f"{df[Y_COL].mean():.1f}")
    c4.metric("Correlation", f"{df[X_COL].corr(df[Y_COL]):.2f}")

    left, right = st.columns([3, 2])
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(df[X_COL], df[Y_COL], color=BLUE, s=45, label="Students")
    xs = np.linspace(h_min, h_max, 50)
    ax.plot(xs, slope * xs + intercept, color=RED, lw=2, label="Regression line")
    ax.set_xlabel("Hours studied")
    ax.set_ylabel("Score")
    ax.set_title("Study hours vs exam score", fontweight="bold")
    ax.legend()
    style(ax)
    left.pyplot(fig)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(5, 4.2))
    axes[0].hist(df[X_COL], bins=8, color=BLUE)
    axes[0].set_title("Hours distribution", fontweight="bold")
    axes[1].hist(df[Y_COL], bins=8, color=GREY)
    axes[1].set_title("Score distribution", fontweight="bold")
    for a in axes:
        style(a)
    fig.tight_layout()
    right.pyplot(fig)
    plt.close(fig)

    with st.expander("View the dataset"):
        st.dataframe(df, use_container_width=True)

# ---------- Tab 2: Model ----------
with tab2:
    st.subheader("Model comparison")
    st.caption("Evaluated with 5-fold cross-validation, which is more reliable than one split on small data.")
    st.dataframe(results, use_container_width=True, hide_index=True)
    st.info(
        f"Final model: **Linear Regression**. It is simple, explainable and as accurate as the others. "
        f"Equation: Score ≈ {slope:.2f} × Hours + {intercept:.2f}"
    )

    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.bar(results["Model"], results["R² (5-fold CV)"], color=[BLUE, GREY, GREY])
    ax.set_ylim(0, 1.05)
    ax.set_title("R² by model (higher is better)", fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=15, ha="right")
    style(ax)
    fig.tight_layout()
    left.pyplot(fig)
    plt.close(fig)

    preds = cv_predictions(df)
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.scatter(df[Y_COL], preds, color=BLUE, s=40)
    lims = [df[Y_COL].min() - 5, df[Y_COL].max() + 5]
    ax.plot(lims, lims, color=RED, ls="--", label="Perfect prediction")
    ax.set_xlabel("Actual score")
    ax.set_ylabel("Predicted score")
    ax.set_title("Actual vs predicted (cross-validated)", fontweight="bold")
    ax.legend()
    style(ax)
    fig.tight_layout()
    right.pyplot(fig)
    plt.close(fig)

# ---------- Tab 3: Predict ----------
with tab3:
    st.subheader("Predict a student's score")
    hours = st.slider("Hours studied", 0.5, 12.0, 6.0, 0.1)
    predicted = float(np.clip(model.predict(pd.DataFrame({X_COL: [hours]}))[0], 0, 100))

    if hours > h_max or hours < h_min:
        st.warning(
            f"The data only covers {h_min:.1f}–{h_max:.1f} hours, so this estimate is less reliable."
        )

    m1, m2 = st.columns(2)
    m1.metric("Predicted score", f"{predicted:.1f} / 100")
    if predicted >= 75:
        m2.success("Excellent: strong preparation")
    elif predicted >= 50:
        m2.info("Good: room to improve")
    else:
        m2.warning("Needs more study time")
    st.progress(int(predicted) / 100)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.scatter(df[X_COL], df[Y_COL], color=GREY, s=35, label="Students")
    xs = np.linspace(0.5, 12, 50)
    ax.plot(xs, np.clip(slope * xs + intercept, 0, 100), color=BLUE, lw=2, label="Model")
    ax.scatter([hours], [predicted], color=RED, s=140, zorder=5, label="Your prediction")
    ax.set_xlabel("Hours studied")
    ax.set_ylabel("Score")
    ax.legend()
    style(ax)
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("#### 🤖 AI study plan")
    if st.button("Get AI study plan"):
        if not api_key:
            st.warning("Paste your Groq API key in the sidebar first.")
        else:
            prompt = (
                f"A student studied {hours:.1f} hours. A model trained on {len(df)} students "
                f"predicts a score of {predicted:.0f} out of 100. Give: 1) a one-line "
                "interpretation, 2) four practical study tips, 3) a simple 5-day study "
                "schedule. Keep it under 200 words."
            )
            with st.spinner("Your study coach is thinking..."):
                try:
                    reply = ask_groq(
                        [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        api_key,
                    )
                    st.markdown(reply)
                except Exception as e:
                    st.error(f"AI request failed: {e}")

# ---------- Tab 4: AI Chat ----------
with tab4:
    st.subheader("💬 AI Chat")
    st.caption(
        "Type or speak your question. Try: *What score for 7 hours?* · *Which model is best?* · "
        "*Give me study tips for a 60 score.*"
    )
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    top1, top2 = st.columns([3, 1])
    auto_speak = top2.checkbox("Auto-speak replies", value=False)
    if top2.button("🗑 Clear chat"):
        st.session_state["messages"] = []

    # --- voice input ---
    spoken = None
    if hasattr(st, "audio_input"):
        audio = top1.audio_input("🎤 Speak your question")
        if audio is not None:
            audio_bytes = audio.getvalue()
            digest = hashlib.md5(audio_bytes).hexdigest()
            if st.session_state.get("last_audio") != digest:
                st.session_state["last_audio"] = digest
                if not api_key:
                    st.warning("Voice input needs the Groq API key.")
                else:
                    with st.spinner("Listening..."):
                        try:
                            spoken = transcribe(audio_bytes, api_key)
                        except Exception as e:
                            st.error(f"Voice input failed: {e}")
    else:
        top1.info("Update Streamlit (pip install -U streamlit) to use voice input.")

    history_box = st.container()
    typed = st.chat_input("Ask something...")
    question = typed or spoken

    if question:
        st.session_state["messages"].append({"role": "user", "content": question})
        with st.spinner("Thinking..."):
            reply = chat_reply(st.session_state["messages"], api_key)
        st.session_state["messages"].append({"role": "assistant", "content": reply})
        st.session_state["speak_new"] = True

    with history_box:
        messages = st.session_state["messages"]
        if not messages:
            st.info("No messages yet. Ask your first question below 👇")
        for msg in messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        if messages and messages[-1]["role"] == "assistant":
            fresh = st.session_state.pop("speak_new", False)
            speak_controls(messages[-1]["content"], auto=auto_speak and fresh)
