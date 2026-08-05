import streamlit as st
import joblib
import re
import string
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# Download NLTK resources
# -----------------------------
nltk.download("punkt")
nltk.download("punkt_tab")   
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# -----------------------------
# Initialize NLP Objects
# -----------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# -----------------------------
# Text Preprocessing Function
# -----------------------------
def preprocess_text(text):

    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove words containing numbers
    text = re.sub(r"\b\w*\d\w*\b", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords & non-alphabetic words
    tokens = [
        word
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]

    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]

    return " ".join(tokens)


# -----------------------------
# Prediction Function
# -----------------------------
def predict_news(news):

    clean_text = preprocess_text(news)

    vector = vectorizer.transform([clean_text])

    prediction = model.predict(vector)[0]

    score = model.decision_function(vector)[0]

    return prediction, score, clean_text


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

st.title("📰 Fake News Detection using Machine Learning")

st.markdown(
    """
This application uses **TF-IDF + Linear SVM** to classify a news article as **Fake** or **Real**.
"""
)

st.write("---")

news = st.text_area(
    "📝 Enter News Article or Headline",
    height=250,
    placeholder="Paste any news article or headline here..."
)

if st.button("🔍 Predict"):

    if news.strip() == "":
        st.warning("⚠️ Please enter some news text.")
    else:

        prediction, score, cleaned = predict_news(news)

        st.write("## Prediction")

        if prediction == 0:
            st.error("❌ Fake News")
        else:
            st.success("✅ Real News")

        st.write(f"**Decision Score:** `{score:.2f}`")
        st.write(f"**Words Processed:** `{len(cleaned.split())}`")

        with st.expander("View Cleaned Text"):
            st.write(cleaned)

st.write("---")

st.caption("Developed using Python, Scikit-learn, TF-IDF, Linear SVM and Streamlit.")