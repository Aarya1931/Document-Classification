import os
from flask import Flask, request, render_template, redirect, url_for
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

import cv2
import pandas as pd
import re
import pickle
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Ensure the upload folder exists
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Load stopwords
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
with open("naive_bayes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Set the path to the Tesseract executable (if required)
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'  # Update this path as needed

# OCR function for extracting text from images
def extract_text_from_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(gray)
    return text

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = ' '.join([word for word in text.split() if word not in stop_words])  # Remove stopwords
    return text

# Document classification function
def classify_document(text):
    processed_text = preprocess_text(text)
    text_tfidf = vectorizer.transform([processed_text])
    return model.predict(text_tfidf)[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file is uploaded
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            # Save the uploaded file
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            # Extract text based on file type
            if file_path.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                extracted_text = extract_text_from_image(file_path)
            elif file_path.endswith('.txt'):
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
            else:
                return "Unsupported file type"

            # Classify document
            predicted_label = classify_document(extracted_text)

            # Save result
            df = pd.DataFrame([[file.filename, extracted_text, predicted_label]], columns=["file_name", "text", "predicted_label"])
            df.to_csv("classified_documents.csv", mode="a", header=not os.path.exists("classified_documents.csv"), index=False)

            return render_template('result.html', label=predicted_label, text=extracted_text)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
