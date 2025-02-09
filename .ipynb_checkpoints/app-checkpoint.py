import streamlit as st
import pickle
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Load trained model and vectorizer
with open("naive_bayes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Streamlit UI
st.title("📄 Document Classifier")
st.write("Upload a text file and get its classification!")

uploaded_file = st.file_uploader("Choose a file", type=["txt"])

if uploaded_file:
    text = uploaded_file.read().decode("utf-8")
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    predicted_label = model.predict(text_tfidf)[0]

    st.subheader("📌 Classification Result")
    st.write(f"**Predicted Label:** {predicted_label}")

    # Save result
    df = pd.DataFrame([[uploaded_file.name, text, predicted_label]], columns=["file_name", "text", "predicted_label"])
    df.to_csv("classified_documents.csv", mode="a", header=not os.path.exists("classified_documents.csv"), index=False)

    st.success("✅ Result saved to 'classified_documents.csv'")

