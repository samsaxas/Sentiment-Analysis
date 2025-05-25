import streamlit as st
from transformers import pipeline
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Set page config
st.set_page_config(
    page_title="🎭 Sentiment Analyzer",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for attractive styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #02111c 0%, #0a1929 50%, #1a2332 100%);
        font-family: 'Poppins', sans-serif;
    }

    /* Main container */
    .main-container {
        background: transparent;
        border-radius: 0px;
        padding: 2rem;
        margin: 0rem;
        box-shadow: none;
        backdrop-filter: none;
        border: none;
        color: #fff; /* Light text for dark background */
    }

    /* Header styling */
    .app-header {
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem;
    }

    .app-title {
        font-size: 3.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        background: linear-gradient(45deg, #ff6b6b, #4ecdc4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-align: center;
        text-shadow: 0px 1px 2px rgba(0,0,0,0.1);
    }

    .app-subtitle {
        font-size: 1.2rem;
        color: #666;
        font-weight: 300;
        margin-bottom: 1rem;
        text-align: center;
    }

    /* Input area styling */
    .stTextArea > div > div > textarea {
        border-radius: 15px;
        border: 2px solid #e0e0e0;
        padding: 1rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        background: rgba(255,255,255,0.9);
        color: #333 !important;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.3);
        transform: translateY(-2px);
    }

    .stTextArea > div > div > textarea::placeholder {
        color: #666 !important;
        opacity: 1;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        width: 100%;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
    }

    /* Results container */
    .results-container {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        margin-top: 1rem;
        border-left: 5px solid #667eea;
    }

    /* Sentiment badges */
    .sentiment-positive {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(86, 171, 47, 0.3);
    }

    .sentiment-negative {
        background: linear-gradient(45deg, #ff416c, #ff4b2b);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.3);
    }

    .sentiment-neutral {
        background: linear-gradient(45deg, #ffecd2, #fcb69f);
        color: #333;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(252, 182, 159, 0.3);
    }

    /* Table styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    /* Success message styling */
    .stSuccess {
        background: linear-gradient(45deg, #56ab2f, #a8e6cf);
        border-radius: 10px;
        border: none;
    }

    /* Animation for loading */
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }

    .analyzing {
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# Create main container (removed white background)
# st.markdown('<div class="main-container">', unsafe_allow_html=True)

# Header
st.markdown("""
<div class="app-header">
    <h1 class="app-title">Sentiment Analyzer</h1>
    <p class="app-subtitle">Powered by BERT & VADER • Advanced Natural Language Processing</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")
st.write("✨ **Discover the emotional tone** of any text using state-of-the-art AI models. Get instant insights!")

# User input section
st.markdown("### 📝 Enter Your Text")
col1, col2 = st.columns([3, 1])

with col1:
    # Check if there's text in session state first
    if 'text' in st.session_state:
        text = st.text_area(
            "Type or paste your text below:",
            value=st.session_state.text,
            height=150,
            placeholder="Example: 'I love this new product! It works perfectly and exceeded my expectations.'"
        )
    else:
        text = st.text_area(
            "Type or paste your text below:",
            height=150,
            placeholder="Example: 'I love this new product! It works perfectly and exceeded my expectations.'"
        )

with col2:
    st.markdown("#### 💡 Try These Examples:")

    # Define example texts
    positive_examples = [
        "I absolutely love this new restaurant! The food is amazing and the service is outstanding.",
        "This movie was fantastic! The acting was superb and the storyline kept me engaged throughout.",
        "I'm so excited about my new job! The team is wonderful and the work is exactly what I wanted.",
        "What a beautiful day! The weather is perfect and I feel incredibly happy and grateful."
    ]

    neutral_examples = [
        "The meeting started at 10 AM and ended at 11:30 AM. We discussed the quarterly reports.",
        "The book has 300 pages and was published in 2020. It covers various topics in detail.",
        "The store is located on Main Street and opens at 9 AM. They sell office supplies.",
        "The temperature today is 72 degrees Fahrenheit. It's partly cloudy with light winds."
    ]

    negative_examples = [
        "This product is terrible. It broke after just one day and customer service was unhelpful.",
        "I'm really disappointed with this purchase. The quality is poor and it doesn't work as advertised.",
        "The service at this restaurant was awful. The food was cold and the staff was rude.",
        "I hate waiting in long lines. This experience has been frustrating and time-consuming."
    ]

    # Initialize example counters in session state
    if 'positive_counter' not in st.session_state:
        st.session_state.positive_counter = 0
    if 'neutral_counter' not in st.session_state:
        st.session_state.neutral_counter = 0
    if 'negative_counter' not in st.session_state:
        st.session_state.negative_counter = 0

    # Positive example button
    if st.button("😊 Positive Example", use_container_width=True, key="pos_btn"):
        st.session_state.text = positive_examples[st.session_state.positive_counter]
        st.session_state.positive_counter = (st.session_state.positive_counter + 1) % len(positive_examples)
        st.rerun()

    # Neutral example button
    if st.button("😐 Neutral Example", use_container_width=True, key="neu_btn"):
        st.session_state.text = neutral_examples[st.session_state.neutral_counter]
        st.session_state.neutral_counter = (st.session_state.neutral_counter + 1) % len(neutral_examples)
        st.rerun()

    # Negative example button
    if st.button("😞 Negative Example", use_container_width=True, key="neg_btn"):
        st.session_state.text = negative_examples[st.session_state.negative_counter]
        st.session_state.negative_counter = (st.session_state.negative_counter + 1) % len(negative_examples)
        st.rerun()

# Use session state to persist text between reruns
if 'text' in st.session_state:
    text = st.session_state.text

# Load BERT pipeline with error handling
@st.cache_resource
def load_bert_pipeline():
    try:
        return pipeline("sentiment-analysis")
    except Exception:
        return None

bert_pipeline = load_bert_pipeline()

# Load VADER analyzer
@st.cache_resource
def load_nltk_vader():
    nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

nltk_analyzer = load_nltk_vader()

st.markdown("---")

# Analysis button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    analyze_button = st.button("� Analyze Sentiment", use_container_width=True, type="primary")

if analyze_button:
    if text.strip() == "":
        st.error("⚠️ Please enter some text to analyze!")
    else:
        # Create results container (removed white background)
        # st.markdown('<div class="results-container">', unsafe_allow_html=True)

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        # First, analyze with VADER
        status_text.text("🔍 Analyzing with VADER...")
        progress_bar.progress(25)

        scores = nltk_analyzer.polarity_scores(text)
        compound = scores['compound']

        progress_bar.progress(50)

        # Prepare and show enhanced table
        st.markdown("### 📊 VADER Sentiment Analysis")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("😞 Negative", f"{scores['neg']:.3f}", delta=None)
        with col2:
            st.metric("😐 Neutral", f"{scores['neu']:.3f}", delta=None)
        with col3:
            st.metric("� Positive", f"{scores['pos']:.3f}", delta=None)
        with col4:
            st.metric("🎯 Compound", f"{scores['compound']:.3f}", delta=None)

        # Then, analyze with BERT (if available)
        if bert_pipeline is not None:
            status_text.text("🤖 Analyzing with BERT...")
            progress_bar.progress(75)

            result = bert_pipeline(text)
            label = result[0]['label']
            score = result[0]['score']

            # Adjust BERT classification for neutral detection
            original_label = label

            # If VADER shows high neutral (>0.8) and BERT disagrees, trust VADER
            if scores['neu'] > 0.8 and label != "NEUTRAL":
                label = "NEUTRAL"
            elif score < 0.75:  # Low confidence threshold for neutral
                if label == "POSITIVE" and score < 0.65:
                    label = "NEUTRAL"
                elif label == "NEGATIVE" and score < 0.65:
                    label = "NEUTRAL"

            progress_bar.progress(100)
            status_text.text("✅ Analysis Complete!")

            # Display BERT results with enhanced styling
            st.markdown("### 🤖 BERT Classification")

            # Create sentiment badge
            if label == "POSITIVE":
                badge_class = "sentiment-positive"
                emoji = "😊"
                color = "#56ab2f"
            elif label == "NEGATIVE":
                badge_class = "sentiment-negative"
                emoji = "😞"
                color = "#ff416c"
            else:
                badge_class = "sentiment-neutral"
                emoji = "😐"
                color = "#fcb69f"

            # Display result with beautiful styling
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div class="{badge_class}" style="font-size: 1.5rem; margin-bottom: 1rem;">
                    {emoji} {label} ({score:.2f})
                </div>
                <div style="background: linear-gradient(45deg, {color}20, {color}10);
                            border-radius: 10px; padding: 1rem; margin-top: 1rem;">
                    <h4>🎯 Final Sentiment: <span style="color: {color};">{label}</span></h4>
                    <p>Confidence Score: <strong>{score:.1%}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            # Use VADER results when BERT is not available
            progress_bar.progress(100)
            status_text.text("✅ Analysis Complete!")

            # Determine sentiment from VADER scores
            if compound >= 0.05:
                label = "POSITIVE"
                badge_class = "sentiment-positive"
                emoji = "😊"
                color = "#56ab2f"
            elif compound <= -0.05:
                label = "NEGATIVE"
                badge_class = "sentiment-negative"
                emoji = "😞"
                color = "#ff416c"
            else:
                label = "NEUTRAL"
                badge_class = "sentiment-neutral"
                emoji = "😐"
                color = "#fcb69f"

            # Display VADER-based final result
            st.markdown("### 🎯 Final Sentiment Analysis")
            st.markdown(f"""
            <div style="text-align: center; padding: 1rem;">
                <div class="{badge_class}" style="font-size: 1.5rem; margin-bottom: 1rem;">
                    {emoji} {label}
                </div>
                <div style="background: linear-gradient(45deg, {color}20, {color}10);
                            border-radius: 10px; padding: 1rem; margin-top: 1rem;">
                    <h4>🎯 Final Sentiment: <span style="color: {color};">{label}</span></h4>
                    <p>Based on VADER Compound Score: <strong>{compound:.3f}</strong></p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()

        # Visual effects
        if label == "POSITIVE":
            st.balloons()
        elif label == "NEGATIVE":
            st.snow()
        else:  # NEUTRAL or other labels
            # Custom full-screen balance scale animation
            st.markdown("""<style>.balance-container{position:fixed!important;top:0!important;left:0!important;width:100vw!important;height:100vh!important;z-index:999999!important;pointer-events:none!important;overflow:hidden!important}@keyframes balance-float-1{0%{transform:translateY(120vh) rotate(0deg);opacity:0}10%{opacity:1}90%{opacity:1}100%{transform:translateY(-20vh) rotate(360deg);opacity:0}}@keyframes balance-float-2{0%{transform:translateY(110vh) rotate(0deg);opacity:0}15%{opacity:1}85%{opacity:1}100%{transform:translateY(-15vh) rotate(-360deg);opacity:0}}@keyframes balance-float-3{0%{transform:translateY(130vh) rotate(0deg);opacity:0}8%{opacity:1}92%{opacity:1}100%{transform:translateY(-25vh) rotate(720deg);opacity:0}}.balance-scale{position:absolute!important;font-size:4rem!important;pointer-events:none!important}.balance-1{animation:balance-float-1 6s ease-in-out!important}.balance-2{animation:balance-float-2 7s ease-out!important}.balance-3{animation:balance-float-3 5s ease-in!important}</style><div class="balance-container"><div class="balance-scale balance-1" style="left:5vw;animation-delay:0s">⚖️</div><div class="balance-scale balance-3" style="left:25vw;animation-delay:0.8s">⚖️</div><div class="balance-scale balance-2" style="left:50vw;animation-delay:1.6s">⚖️</div><div class="balance-scale balance-1" style="left:75vw;animation-delay:2.4s">⚖️</div><div class="balance-scale balance-3" style="left:90vw;animation-delay:3.2s">⚖️</div><div class="balance-scale balance-2" style="left:15vw;animation-delay:0.4s">⚖️</div><div class="balance-scale balance-1" style="left:35vw;animation-delay:1.2s">⚖️</div><div class="balance-scale balance-3" style="left:60vw;animation-delay:2.0s">⚖️</div><div class="balance-scale balance-2" style="left:80vw;animation-delay:2.8s">⚖️</div><div class="balance-scale balance-3" style="left:8vw;animation-delay:0.6s">⚖️</div><div class="balance-scale balance-2" style="left:28vw;animation-delay:1.4s">⚖️</div><div class="balance-scale balance-1" style="left:45vw;animation-delay:2.2s">⚖️</div><div class="balance-scale balance-3" style="left:68vw;animation-delay:3.0s">⚖️</div><div class="balance-scale balance-2" style="left:85vw;animation-delay:3.8s">⚖️</div><div class="balance-scale balance-1" style="left:12vw;animation-delay:1.0s">⚖️</div><div class="balance-scale balance-3" style="left:38vw;animation-delay:1.8s">⚖️</div><div class="balance-scale balance-2" style="left:55vw;animation-delay:2.6s">⚖️</div><div class="balance-scale balance-1" style="left:72vw;animation-delay:3.4s">⚖️</div><div class="balance-scale balance-3" style="left:92vw;animation-delay:4.2s">⚖️</div></div>""", unsafe_allow_html=True)

        # st.markdown('</div>', unsafe_allow_html=True)

# Close main container and add footer (removed white background)
# st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #666;">
    <p>💡 <em>Analyze any text to discover its emotional sentiment instantly!</em></p>
</div>
""", unsafe_allow_html=True)
