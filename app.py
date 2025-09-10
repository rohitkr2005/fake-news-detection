import streamlit as st
import joblib
import pandas as pd
import re
import nltk
from nltk.stem import WordNetLemmatizer

# ------------------- Load Model & Vectorizer -------------------
model = joblib.load("fake_news_model.pkl")      # Logistic Regression (or Naive Bayes)
tfidf = joblib.load("tfidf_vectorizer.pkl")

# ------------------- Preprocessing -------------------
nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()

def clean_text(text, max_words=200):
    text = str(text).lower()
    text = re.sub(r'<.*?>', ' ', text)                # remove HTML
    text = re.sub(r'http\S+|www\.\S+', ' ', text)     # remove URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)          # keep only letters/numbers
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()[:max_words]
    words = [lemmatizer.lemmatize(w) for w in words]
    return ' '.join(words)

# ------------------- Prediction Functions -------------------
def predict_news(text):
    cleaned = clean_text(text)
    vectorized = tfidf.transform([cleaned])
    pred = model.predict(vectorized)[0]
    prob = model.predict_proba(vectorized)[0]
    return ("FAKE" if pred == 1 else "REAL"), prob

def predict_bulk(df, text_col="text"):
    df["clean_text"] = df[text_col].apply(clean_text)
    vectors = tfidf.transform(df["clean_text"])
    preds = model.predict(vectors)
    probs = model.predict_proba(vectors)
    df["Prediction"] = ["FAKE" if p == 1 else "REAL" for p in preds]
    df["Confidence"] = probs.max(axis=1)
    return df

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="wide")

st.title("📰 Fake News Detection System")
st.write("Check if news articles are **Real or Fake** using Machine Learning (TF-IDF + ML Model).")

# Tabs for Single vs Bulk
tab1, tab2 = st.tabs(["🔍 Single News Check", "📂 Bulk CSV Upload"])

# --- Tab 1: Single News ---
with tab1:
    user_input = st.text_area("✍️ Enter News Text Here", height=150)

    if st.button("Check News"):
        if user_input.strip() == "":
            st.warning("⚠️ Please enter some text first.")
        else:
            label, prob = predict_news(user_input)
            st.subheader(f"Prediction: {label}")
            st.write(f"Confidence: {prob.max():.2f}")

# --- Tab 2: Bulk CSV Upload ---
with tab2:
    uploaded_file = st.file_uploader("📂 Upload a CSV file (must have a 'text' column)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        if "text" not in df.columns:
            st.error("❌ The CSV must contain a 'text' column with news articles.")
        else:
            st.success(f"✅ Loaded {len(df)} news articles.")
            result_df = predict_bulk(df, text_col="text")

            st.subheader("📊 Predictions")
            st.dataframe(result_df[["text", "Prediction", "Confidence"]])

            # Option to download results
            csv = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv,
                file_name="fake_news_predictions.csv",
                mime="text/csv"
            )