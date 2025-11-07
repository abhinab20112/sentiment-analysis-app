import streamlit as st
import numpy as np
import pandas as pd
import pickle
import re
import plotly.graph_objects as go
import plotly.express as px
from collections import Counter
import os

# Handle TensorFlow import and compatibility
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    
    # Disable TensorFlow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    tf.get_logger().setLevel('ERROR')
except ImportError as e:
    st.error(f"TensorFlow import error: {e}")
    st.info("Please ensure TensorFlow is installed: pip install tensorflow")

# Page configuration
st.set_page_config(
    page_title="Twitter Sentiment Analysis",
    page_icon="🐦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
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
    .positive {
        color: #10b981;
        font-weight: bold;
    }
    .negative {
        color: #ef4444;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models_and_tokenizer():
    """Load both models and tokenizer (cached for performance)"""
    try:
        lstm_model = load_model('sentiment_lstm.h5')
        ann_model = load_model('sentiment_ann.h5')
        with open('tokenizer.pickle', 'rb') as f:
            tokenizer = pickle.load(f)
        return lstm_model, ann_model, tokenizer
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure 'sentiment_lstm.h5', 'sentiment_ann.h5', and 'tokenizer.pickle' are in the same directory!")
        return None, None, None


def preprocess_tweet(text):
    """Preprocess tweet text (same as training)"""
    text = re.sub(r'http\S+|www\S+|https\S+', '[URL]', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '[USER]', text)
    text = re.sub(r'(.)\1+', r'\1', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.\U0001F300-\U0001F9FF]', '', text)
    text = text.lower()
    text = ' '.join(text.split())
    return text


def predict_sentiment(text, model, tokenizer, max_len=50):
    """Predict sentiment for a single tweet"""
    cleaned = preprocess_tweet(text)
    sequence = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded, verbose=0)[0][0]
    
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    confidence = prediction if prediction > 0.5 else (1 - prediction)
    
    return sentiment, confidence, cleaned


def create_gauge_chart(confidence, sentiment):
    """Create a gauge chart for confidence visualization"""
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
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def main():
    # Header
    st.markdown('<h1 class="main-header">🐦 Twitter Sentiment Analysis Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("**Analyze social media sentiment using Deep Learning (LSTM)**")
    
    # Load models
    lstm_model, ann_model, tokenizer = load_models_and_tokenizer()
    
    if lstm_model is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("📊 Navigation")
    page = st.sidebar.radio("Choose a page:", 
                           ["🏠 Single Tweet Analysis", 
                            "📝 Batch Analysis", 
                            "🔬 Model Comparison",
                            "ℹ️ About"])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📈 Model Performance")
    st.sidebar.metric("LSTM Accuracy", "85%", "5%")
    st.sidebar.metric("ANN Accuracy", "80%", "-")
    
    # Page 1: Single Tweet Analysis
    if page == "🏠 Single Tweet Analysis":
        st.header("Analyze a Single Tweet")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            user_input = st.text_area(
                "Enter a tweet to analyze:",
                placeholder="Type or paste a tweet here...",
                height=100
            )
            
            analyze_btn = st.button("🔍 Analyze Sentiment", type="primary")
        
        if analyze_btn and user_input:
            with st.spinner("Analyzing sentiment..."):
                sentiment, confidence, cleaned_text = predict_sentiment(
                    user_input, lstm_model, tokenizer
                )
                
                # Display results
                st.success("Analysis Complete!")
                
                # Result columns
                res_col1, res_col2 = st.columns(2)
                
                with res_col1:
                    emoji = "😊" if sentiment == "Positive" else "😞"
                    st.markdown(f"### Result: {sentiment} {emoji}")
                    st.markdown(f"**Confidence:** {confidence*100:.2f}%")
                    
                    # Show cleaned tweet
                    with st.expander("View Preprocessed Tweet"):
                        st.code(cleaned_text)
                
                with res_col2:
                    # Gauge chart
                    fig = create_gauge_chart(confidence, sentiment)
                    st.plotly_chart(fig, use_container_width=True)
        
        # Example tweets
        st.markdown("---")
        st.subheader("Try Example Tweets:")
        examples = {
            "Positive": "I absolutely love this product! Best purchase ever! 😊",
            "Negative": "This is terrible. Worst experience of my life 😡",
            "Mixed": "The service was okay but not great"
        }
        
        ex_col1, ex_col2, ex_col3 = st.columns(3)
        if ex_col1.button("Try Positive Example"):
            st.session_state.example = examples["Positive"]
        if ex_col2.button("Try Negative Example"):
            st.session_state.example = examples["Negative"]
        if ex_col3.button("Try Mixed Example"):
            st.session_state.example = examples["Mixed"]
    
    # Page 2: Batch Analysis
    elif page == "📝 Batch Analysis":
        st.header("Batch Sentiment Analysis")
        
        st.markdown("Analyze multiple tweets at once by pasting them below (one per line) or uploading a CSV file.")
        
        tab1, tab2 = st.tabs(["📄 Paste Text", "📁 Upload CSV"])
        
        with tab1:
            batch_input = st.text_area(
                "Enter tweets (one per line):",
                placeholder="Tweet 1\nTweet 2\nTweet 3...",
                height=200
            )
            
            if st.button("Analyze Batch", type="primary"):
                if batch_input:
                    tweets = [t.strip() for t in batch_input.split('\n') if t.strip()]
                    
                    with st.spinner(f"Analyzing {len(tweets)} tweets..."):
                        results = []
                        for tweet in tweets:
                            sentiment, confidence, _ = predict_sentiment(
                                tweet, lstm_model, tokenizer
                            )
                            results.append({
                                'Tweet': tweet[:50] + '...' if len(tweet) > 50 else tweet,
                                'Sentiment': sentiment,
                                'Confidence': f"{confidence*100:.2f}%"
                            })
                        
                        df = pd.DataFrame(results)
                        
                        # Statistics
                        st.success(f"Analyzed {len(tweets)} tweets!")
                        
                        col1, col2, col3 = st.columns(3)
                        positive_count = len([r for r in results if r['Sentiment'] == 'Positive'])
                        negative_count = len(results) - positive_count
                        
                        col1.metric("Total Tweets", len(tweets))
                        col2.metric("Positive 😊", positive_count, 
                                   f"{positive_count/len(tweets)*100:.1f}%")
                        col3.metric("Negative 😞", negative_count,
                                   f"{negative_count/len(tweets)*100:.1f}%")
                        
                        # Pie chart
                        fig = px.pie(
                            values=[positive_count, negative_count],
                            names=['Positive', 'Negative'],
                            title='Sentiment Distribution',
                            color_discrete_map={'Positive': '#10b981', 'Negative': '#ef4444'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Results table
                        st.dataframe(df, use_container_width=True)
                        
                        # Download button
                        csv = df.to_csv(index=False)
                        st.download_button(
                            "📥 Download Results",
                            csv,
                            "sentiment_results.csv",
                            "text/csv"
                        )
        
        with tab2:
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            if uploaded_file:
                df_upload = pd.read_csv(uploaded_file)
                st.write("Preview:", df_upload.head())
                
                text_column = st.selectbox("Select the column containing tweets:", df_upload.columns)
                
                if st.button("Analyze CSV", type="primary"):
                    with st.spinner("Processing..."):
                        results = []
                        for tweet in df_upload[text_column]:
                            sentiment, confidence, _ = predict_sentiment(
                                str(tweet), lstm_model, tokenizer
                            )
                            results.append({
                                'Sentiment': sentiment,
                                'Confidence': confidence
                            })
                        
                        df_upload['Sentiment'] = [r['Sentiment'] for r in results]
                        df_upload['Confidence'] = [f"{r['Confidence']*100:.2f}%" for r in results]
                        
                        st.success("Analysis complete!")
                        st.dataframe(df_upload, use_container_width=True)
                        
                        csv = df_upload.to_csv(index=False)
                        st.download_button("📥 Download Results", csv, "results.csv", "text/csv")
    
    # Page 3: Model Comparison
    elif page == "🔬 Model Comparison":
        st.header("Compare LSTM vs Simple ANN")
        
        st.markdown("""
        This project compared two architectures:
        - **Simple ANN:** Embedding → Global Average Pooling → Dense
        - **LSTM:** Embedding → LSTM layer → Dense
        """)
        
        # Performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🏆 LSTM Model")
            st.metric("Test Accuracy", "85%")
            st.markdown("**Advantages:**")
            st.markdown("- Captures sequential patterns")
            st.markdown("- Better context understanding")
            st.markdown("- Handles word order")
        
        with col2:
            st.markdown("### 📊 Simple ANN")
            st.metric("Test Accuracy", "80%")
            st.markdown("**Advantages:**")
            st.markdown("- Faster training")
            st.markdown("- Fewer parameters")
            st.markdown("- Good baseline")
        
        # Side-by-side comparison
        st.markdown("---")
        st.subheader("Test Both Models")
        
        compare_input = st.text_input("Enter a tweet to compare predictions:")
        
        if st.button("Compare Models") and compare_input:
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### LSTM Prediction")
                sentiment_lstm, conf_lstm, _ = predict_sentiment(
                    compare_input, lstm_model, tokenizer
                )
                emoji = "😊" if sentiment_lstm == "Positive" else "😞"
                st.markdown(f"**{sentiment_lstm}** {emoji}")
                st.progress(conf_lstm)
                st.caption(f"Confidence: {conf_lstm*100:.2f}%")
            
            with col2:
                st.markdown("### Simple ANN Prediction")
                sentiment_ann, conf_ann, _ = predict_sentiment(
                    compare_input, ann_model, tokenizer
                )
                emoji = "😊" if sentiment_ann == "Positive" else "😞"
                st.markdown(f"**{sentiment_ann}** {emoji}")
                st.progress(conf_ann)
                st.caption(f"Confidence: {conf_ann*100:.2f}%")
    
    # Page 4: About
    else:
        st.header("About This Project")
        
        st.markdown("""
        ### 🎯 Project Overview
        This is a **Twitter Sentiment Analysis** system built using Deep Learning (LSTM) 
        to classify tweets as positive or negative sentiment.
        
        ### 📊 Dataset
        - **Source:** Sentiment140 dataset
        - **Size:** 100,000 tweets (sampled from 1.6M)
        - **Classes:** Binary (Positive/Negative)
        - **Split:** 80% training, 20% testing
        
        ### 🧠 Model Architecture
        **LSTM Model:**
        - Embedding Layer (vocab=10000, dim=128)
        - LSTM Layer (64 units)
        - Dropout (0.5)
        - Dense Output (sigmoid activation)
        
        ### 📈 Performance
        - **LSTM Accuracy:** 85%
        - **Simple ANN Accuracy:** 80%
        - **Improvement:** 5% by using sequential modeling
        
        ### 🛠️ Tech Stack
        - **Framework:** TensorFlow/Keras
        - **Deployment:** Streamlit
        - **Visualization:** Plotly
        - **Language:** Python
        
        ### 👨‍💻 Developer
        Built as a B.Tech project demonstrating:
        - End-to-end ML pipeline
        - Model comparison and evaluation
        - Production deployment
        - Interactive visualization
        
        ### 📝 Key Learnings
        1. Preprocessing is crucial for text data
        2. LSTM captures context better than simple averaging
        3. Model comparison validates architectural choices
        4. Deployment makes projects tangible and shareable
        
        ---
        
        **GitHub:** [Your GitHub Link]  
        **LinkedIn:** [Your LinkedIn]  
        **Email:** [Your Email]
        """)
        
        st.balloons()


if __name__ == "__main__":
    main()