# 📰 Fake News Detection System

A Fake News Detection System built using Python, NLP, and Machine Learning.
This project uses TF-IDF vectorization + Logistic Regression / Naive Bayes to classify news articles as REAL or FAKE.
It also provides a Streamlit web interface for both single news check and bulk CSV uploads.

## ✨ Features

✅ Preprocessing pipeline (cleaning, lemmatization, tokenization)

✅ TF-IDF vectorization (unigrams & bigrams)

✅ Machine Learning models (Logistic Regression, Naive Bayes)

✅ Interactive Streamlit UI

✅ Bulk CSV file upload for mass predictions

✅ Download results with predictions

## 🚀 Tech Stack

Python 3.8+

Pandas, NumPy (data handling)

NLTK (text preprocessing)

scikit-learn (TF-IDF + ML models)

Joblib (model persistence)

Streamlit (web UI)

## 📂 Dataset

We used the Fake and Real News Dataset from Kaggle:
👉 Fake and Real News Dataset

The dataset was manually downloaded, preprocessed, and used for training.


## Install dependencies:

pip install -r requirements.txt


## Train or load the pre-trained model:

Training script: train_model.py (optional if you want retraining)

Pre-trained model: fake_news_model.pkl + tfidf_vectorizer.pkl

Run the Streamlit app:

streamlit run app.py

## 🎯 Usage
### 🔍 Single News Check

Enter a news headline or article.

Click Check News.

The app will display Prediction (REAL/FAKE) + Confidence Score.

### 📂 Bulk CSV Upload

Upload a CSV file with a text column.

The app will classify all news articles.

Download results as a CSV file with added predictions.


🔮 Future Improvements

Add more advanced deep learning models (LSTMs, Transformers).

Improve dataset with more real-world examples.

Deploy online (Heroku, Streamlit Cloud).


## ✅ Ready to push! Just make sure you include:

requirements.txt

app.py

fake_news_model.pkl & tfidf_vectorizer.pkl

Optionally, assets/ folder for screenshots.
