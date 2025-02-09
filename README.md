Video Link : https://drive.google.com/file/d/17UzYC-UXX1ZLbFjSPtKr3aqCIY5tR2ox/view?usp=sharing


📝 Document Classification using OCR & Naïve Bayes
📌 Project Overview
This project is a Flask-based web application that allows users to upload images or text files, extract text using OCR (Tesseract), classify the document using a Naïve Bayes model, and display the classification result.

✅ Key Features:

🖼️ Supports Image & Text Files (.png, .jpg, .jpeg, .tiff, .txt)
🔍 Extracts Text from Images using Tesseract OCR
📊 Classifies Documents using Naïve Bayes & TF-IDF
🎨 Modern UI with Attractive Styling
🌐 Flask-based Web Application
📁 Automatically Saves Classified Results
🚀 Tech Stack Used
🔹 Backend: Flask, Python
🔹 Machine Learning: Scikit-Learn (Naïve Bayes, TF-IDF)
🔹 OCR: Tesseract-OCR, pytesseract
🔹 Frontend: HTML, CSS
🔹 File Monitoring: Watchdog
🔹 Data Handling: Pandas

📂 Project Structure
📁 Document_Classification_OCR
│── 📁 uploads/                # Stores uploaded files
│── 📁 templates/              # HTML Templates
│   ├── index.html             # Upload Page
│   ├── result.html            # Classification Result Page
│── 📄 app.py                  # Flask Application
│── 📄 naive_bayes_model.pkl    # Pre-trained ML Model
│── 📄 tfidf_vectorizer.pkl     # TF-IDF Vectorizer
│── 📄 classified_documents.csv # Saves Classification Results
│── 📄 requirements.txt         # Python Dependencies
│── 📄 README.md                # Project Documentation
⚡ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/yourusername/Document-Classification-OCR.git
cd Document-Classification-OCR
2️⃣ Create a Virtual Environment (Optional but Recommended)
python -m venv venv
source venv/bin/activate   # On macOS/Linux
venv\Scripts\activate      # On Windows
3️⃣ Install Dependencies 
pip install -r requirements.txt
4️⃣ Download & Install Tesseract-OCR
🔗 Download: Tesseract-OCR
After installation, update the Tesseract path in app.py:
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
5️⃣ Run the Application
python app.py
🔹 Open the app in your browser: http://127.0.0.1:5000/
🎯 How to Use?
1️⃣ Upload an Image or Text File 📤
2️⃣ OCR Extracts Text from Images 🔍
3️⃣ Naïve Bayes Classifies the Document 🤖
4️⃣ View & Save Classification Results 📊
5️⃣ Try Another File! 🔄

📌 Future Enhancements
✨ Deploy Model with Flask API
✨ Integrate Deep Learning OCR (EasyOCR, PaddleOCR)
✨ Multi-Language OCR Support
✨ More Machine Learning Models for Classification


⭐ Like this project? Give it a star! ⭐
💙 Developed with Passion 🚀
