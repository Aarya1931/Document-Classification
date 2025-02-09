import os
import time
import pytesseract
import cv2
import pandas as pd
import re
import nltk
import pickle
from nltk.corpus import stopwords
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from sklearn.feature_extraction.text import TfidfVectorizer

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load trained model and vectorizer
with open("naive_bayes_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

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

# Handler for detecting new files
class DocumentHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            print(f"📄 New file detected: {file_path}")

            # Extract text based on file type
            if file_path.endswith(('.png', '.jpg', '.jpeg', '.tiff')):
                extracted_text = extract_text_from_image(file_path)
            elif file_path.endswith('.txt'):
                with open(file_path, "r", encoding="utf-8") as f:
                    extracted_text = f.read()
            else:
                print("❌ Unsupported file type. Skipping...")
                return

            # Classify document
            predicted_label = classify_document(extracted_text)
            print(f"✅ Classified as: {predicted_label}")

            # Save result
            df = pd.DataFrame([[file_path, extracted_text, predicted_label]], columns=["file_path", "text", "predicted_label"])
            df.to_csv("classified_documents.csv", mode="a", header=not os.path.exists("classified_documents.csv"), index=False)

            print("✅ Document classification saved.")

# Monitor folder for new files
def monitor_folder():
    folder_path = "new_docs"
    event_handler = DocumentHandler()
    observer = Observer()
    observer.schedule(event_handler, folder_path, recursive=False)
    observer.start()
    
    print("👀 Watching for new files in 'new_docs/'...")
    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Start monitoring
if __name__ == "__main__":
    monitor_folder()
