import os
import re
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model
from keras.utils import pad_sequences
import plotly.graph_objects as go
import plotly.express as px

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # 0 = all logs, 1 = warnings, 2 = errors only
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"  # keep optimized ops on


# ────────────────────────────────
# 📘 PAGE CONFIGURATION
# ────────────────────────────────
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ────────────────────────────────
# 🎨 CUSTOM CSS STYLING
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
# 📦 LOAD MODELS AND TOKENIZER
# ────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_models_and_tokenizer():
    """Load models and tokenizer with safe paths."""
    try:
        lstm_model_path = os.path.join(BASE_DIR, "sentiment_model.h5")
        ann_model_path = os.path.join(BASE_DIR, "sentiment_ann.h5")
        tokenizer_path = os.path.join(BASE_DIR, "tokenizer.pickle")

        lstm_model = load_model(lstm_model_path)
        ann_model = load_model(ann_model_path)
        with open(tokenizer_path, "rb") as f:
            tokenizer = pickle.load(f)

        return lstm_model, ann_model, tokenizer
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        st.info("✅ Ensure 'sentiment_model.h5', 'sentiment_ann.h5', and 'tokenizer.pickle' are in the same folder as app.py.")
        return None, None, None

# ────────────────────────────────
# 🧹 PREPROCESSING
# ────────────────────────────────
def preprocess_tweet(text: str) -> str:
    """Preprocess tweet text (same as during training)."""
    text = re.sub(r"http\S+|www\S+|https\S+", "[URL]", text)
    text = re.sub(r"@\w+", "[USER]", text)
    text = re.sub(r"(.)\1+", r"\1", text)
    text = re.sub(r"[^a-zA-Z0-9\s!?.\U0001F300-\U0001F9FF]", "", text)
    return " ".join(text.lower().split())

# ────────────────────────────────
# 🤖 PREDICTION
# ────────────────────────────────
def predict_sentiment(text, model, tokenizer, max_len=50):
    """Predict sentiment for a given text."""
    cleaned = preprocess_tweet(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)[0][0]

    sentiment = "Positive" if pred > 0.5 else "Negative"
    confidence = pred if pred > 0.5 else (1 - pred)

    # 🔧 Convert to native float to avoid Streamlit type error
    return sentiment, float(confidence), cleaned


# ────────────────────────────────
# 📊 VISUALIZATION
# ────────────────────────────────
def create_gauge_chart(confidence, sentiment):
    """Create gauge chart for confidence visualization."""
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
# 🏠 MAIN APP FUNCTION
# ────────────────────────────────
def main():
    st.markdown('<h1 class="main-header">🐦 Twitter Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze social media sentiment using Deep Learning (LSTM)**")

    lstm_model, ann_model, tokenizer = load_models_and_tokenizer()
    if lstm_model is None:
        st.stop()

    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Choose a page:", 
                            ["🏠 Single Tweet Analysis", "📝 Batch Analysis", "🔬 Model Comparison", "ℹ️ About"])
    st.sidebar.markdown("---")
    st.sidebar.metric("LSTM Accuracy", "85%", "5%")
    st.sidebar.metric("ANN Accuracy", "80%", "-")

    # ──────────────── PAGE 1 ────────────────
    if page == "🏠 Single Tweet Analysis":
        st.header("Analyze a Single Tweet")
        col1, col2 = st.columns([2, 1])

        with col1:
            user_input = st.text_area("Enter a tweet:", placeholder="Type or paste a tweet...", height=100)
            analyze_btn = st.button("🔍 Analyze Sentiment", type="primary")

        if analyze_btn and user_input.strip():
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence, cleaned = predict_sentiment(user_input, lstm_model, tokenizer)
                st.success("Analysis Complete!")
                res1, res2 = st.columns(2)

                with res1:
                    emoji = "😊" if sentiment == "Positive" else "😞"
                    st.markdown(f"### Result: {sentiment} {emoji}")
                    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                    with st.expander("View Preprocessed Tweet"):
                        st.code(cleaned)

                with res2:
                    st.plotly_chart(create_gauge_chart(confidence, sentiment), use_container_width=True)

    # ──────────────── PAGE 2 ────────────────
    elif page == "📝 Batch Analysis":
        st.header("Batch Sentiment Analysis")
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
                            results.append({'Tweet': t[:50], 'Sentiment': s, 'Confidence': f"{c*100:.2f}%"})
                    df = pd.DataFrame(results)
                    st.dataframe(df)
                    st.download_button("📥 Download CSV", df.to_csv(index=False), "results.csv", "text/csv")

        with tab2:
            uploaded = st.file_uploader("Upload CSV file", type=["csv"])
            if uploaded:
                df = pd.read_csv(uploaded)
                col = st.selectbox("Select text column:", df.columns)
                if st.button("Analyze CSV"):
                    results = []
                    for t in df[col]:
                        s, c, _ = predict_sentiment(str(t), lstm_model, tokenizer)
                        results.append({'Sentiment': s, 'Confidence': f"{c*100:.2f}%"})
                    df["Sentiment"] = [r["Sentiment"] for r in results]
                    df["Confidence"] = [r["Confidence"] for r in results]
                    st.dataframe(df)
                    st.download_button("📥 Download CSV", df.to_csv(index=False), "analyzed.csv", "text/csv")

    # ──────────────── PAGE 3 ────────────────
    elif page == "🔬 Model Comparison":
        st.header("Compare LSTM vs ANN")
        tweet = st.text_input("Enter a tweet:")
        if st.button("Compare Models") and tweet.strip():
            s1, c1, _ = predict_sentiment(tweet, lstm_model, tokenizer)
            s2, c2, _ = predict_sentiment(tweet, ann_model, tokenizer)
            col1, col2 = st.columns(2)
            for label, s, c, col in [("LSTM", s1, c1, col1), ("ANN", s2, c2, col2)]:
                with col:
                    emoji = "😊" if s == "Positive" else "😞"
                    st.markdown(f"### {label}: {s} {emoji}")
                    st.progress((float(c))
)
                    st.caption(f"Confidence: {c*100:.2f}%")

    # ──────────────── PAGE 4 ────────────────
    else:
        st.header("About This Project")
        st.markdown("""
        **Twitter Sentiment Analysis** using Deep Learning (LSTM and ANN).  
        Built with TensorFlow/Keras + Streamlit + Plotly for visualization.
        """)

# ────────────────────────────────
# 🚀 RUN APP
# ────────────────────────────────
if __name__ == "__main__":
    main()
