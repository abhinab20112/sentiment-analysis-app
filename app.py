import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from keras.models import load_model  # ✅ Use standalone Keras loader for .h5 models
from plotly import graph_objects as go
import plotly.express as px

# ────────────────────────────────
# 🌍 Environment setup
# ────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress TensorFlow warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Optional: disable oneDNN CPU optimizations

# ────────────────────────────────
# 📘 Streamlit page configuration
# ────────────────────────────────
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────
# 🎨 Custom CSS styling
# ────────────────────────────────
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1DA1F2;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .positive { color: #10b981; font-weight: bold; }
    .negative { color: #ef4444; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ────────────────────────────────
# 📦 Model and tokenizer loader
# ────────────────────────────────
BASE_DIR = os.getcwd()

@st.cache_resource
def load_models_and_tokenizer():
    """Safely load Keras models and tokenizer."""
    lstm_path = os.path.join(BASE_DIR, "sentiment_model.h5")
    ann_path = os.path.join(BASE_DIR, "sentiment_ann.h5")
    tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pickle")

    # Check files exist
    if not all(os.path.exists(p) for p in [lstm_path, ann_path, tokenizer_path]):
        st.warning("⚠️ Model or tokenizer files not found. Place them in the same folder as app.py.")
        return None, None, None

    try:
        # Try loading using standalone Keras (works best with older .h5)
        lstm_model = load_model(lstm_path, compile=False)
        ann_model = load_model(ann_path, compile=False)

        # Load tokenizer
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        st.success("✅ Models and tokenizer loaded successfully!")
        return lstm_model, ann_model, tokenizer

    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("If you trained with tf.keras, you can switch back to tf.keras.models.load_model() instead.")
        return None, None, None

# ────────────────────────────────
# 🧹 Text preprocessing
# ────────────────────────────────
def preprocess_tweet(text: str) -> str:
    """Clean and normalize tweets before feeding to model."""
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"(.)\\1+", r"\\1", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.\U0001F300-\U0001F9FF]", "", text)
    return " ".join(text.lower().split())

# ────────────────────────────────
# 🤖 Prediction
# ────────────────────────────────
def predict_sentiment(text, model, tokenizer, max_len=50):
    """Predict sentiment using the selected model."""
    if model is None or tokenizer is None:
        return "Unavailable", 0.0, text

    cleaned = preprocess_tweet(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")

    try:
        pred = float(model.predict(padded, verbose=0)[0][0])
    except Exception:
        return "Unavailable", 0.0, cleaned

    sentiment = "Positive" if pred > 0.5 else "Negative"
    confidence = pred if pred > 0.5 else (1 - pred)
    return sentiment, confidence, cleaned

# ────────────────────────────────
# 📊 Visualization
# ────────────────────────────────
def create_gauge_chart(confidence, sentiment):
    color = "#10b981" if sentiment == "Positive" else "#ef4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"{sentiment} Sentiment"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "darkgray"}
            ]
        }
    ))
    fig.update_layout(height=300)
    return fig

# ────────────────────────────────
# 🏠 Main app
# ────────────────────────────────
def main():
    st.markdown('<h1 class="main-header">🐦 Twitter Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("Analyze social media sentiment using **Deep Learning (LSTM & ANN)**")

    # Load models & tokenizer
    lstm_model, ann_model, tokenizer = load_models_and_tokenizer()

    # Sidebar navigation
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Choose a page:", ["🏠 Single Tweet", "📝 Batch Analysis", "🔬 Model Comparison", "ℹ️ About"])
    st.sidebar.markdown("---")
    st.sidebar.metric("LSTM Accuracy", "85%", "↑5%")
    st.sidebar.metric("ANN Accuracy", "80%", "→ Stable")

    # 🏠 Single Tweet Analysis
    if page == "🏠 Single Tweet":
        st.subheader("Analyze a Single Tweet")
        tweet = st.text_area("Enter a tweet:", placeholder="Type or paste a tweet...", height=100)

        if st.button("🔍 Analyze") and tweet.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence, cleaned = predict_sentiment(tweet, lstm_model, tokenizer)
            emoji = "😊" if sentiment == "Positive" else "😞"
            st.success(f"**Result:** {sentiment} {emoji}")
            st.markdown(f"**Confidence:** {confidence*100:.2f}%")
            with st.expander("View Preprocessed Tweet"):
                st.code(cleaned)
            st.plotly_chart(create_gauge_chart(confidence, sentiment), use_container_width=True)

    # 📝 Batch Analysis
    elif page == "📝 Batch Analysis":
        st.subheader("Batch Sentiment Analysis")
        tab1, tab2 = st.tabs(["📄 Paste Text", "📁 Upload CSV"])

        with tab1:
            text_data = st.text_area("Tweets (one per line):", height=200)
            if st.button("Analyze Batch"):
                tweets = [t.strip() for t in text_data.splitlines() if t.strip()]
                if not tweets:
                    st.warning("Please enter some text!")
                else:
                    results = []
                    with st.spinner(f"Analyzing {len(tweets)} tweets..."):
                        for t in tweets:
                            s, c, _ = predict_sentiment(t, lstm_model, tokenizer)
                            results.append({"Tweet": t[:50], "Sentiment": s, "Confidence": f"{c*100:.2f}%"})
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    st.download_button("📥 Download CSV", df.to_csv(index=False), "batch_results.csv", "text/csv")

        with tab2:
            uploaded = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded:
                df = pd.read_csv(uploaded)
                col = st.selectbox("Select text column:", df.columns)
                if st.button("Analyze CSV"):
                    results = []
                    with st.spinner("Analyzing file..."):
                        for t in df[col]:
                            s, c, _ = predict_sentiment(str(t), lstm_model, tokenizer)
                            results.append({"Sentiment": s, "Confidence": f"{c*100:.2f}%"})
                    df["Sentiment"] = [r["Sentiment"] for r in results]
                    df["Confidence"] = [r["Confidence"] for r in results]
                    st.dataframe(df)
                    st.download_button("📥 Download CSV", df.to_csv(index=False), "analyzed.csv", "text/csv")

    # 🔬 Model Comparison
    elif page == "🔬 Model Comparison":
        st.subheader("Compare LSTM vs ANN")
        tweet = st.text_input("Enter a tweet:")
        if st.button("Compare Models") and tweet.strip():
            s1, c1, _ = predict_sentiment(tweet, lstm_model, tokenizer)
            s2, c2, _ = predict_sentiment(tweet, ann_model, tokenizer)
            col1, col2 = st.columns(2)
            for label, s, c, col in [("LSTM", s1, c1, col1), ("ANN", s2, c2, col2)]:
                with col:
                    emoji = "😊" if s == "Positive" else "😞"
                    st.markdown(f"### {label}: {s} {emoji}")
                    st.progress(float(c))
                    st.caption(f"Confidence: {c*100:.2f}%")

    # ℹ️ About Page
    else:
        st.subheader("About This Project")
        st.markdown("""
        This is a **Twitter Sentiment Analysis Dashboard** built with:
        - 🧠 Deep Learning (LSTM & ANN)  
        - 🐍 Keras / TensorFlow backend  
        - 🎨 Streamlit for UI  
        - 📊 Plotly for interactive visualization
        """)

# ────────────────────────────────
# 🚀 Run the app
# ────────────────────────────────
if __name__ == "__main__":
    main()
