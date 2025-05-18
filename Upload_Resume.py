import streamlit as st 
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import pickle
import mysql.connector
from pathlib import Path
import tempfile
from docx import Document
import os
import logging
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import fitz  # PyMuPDF

# Optional: Set path to Tesseract executable (Windows only)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# --- Logging Setup ---
logging.basicConfig(level=logging.ERROR)

# --- Preprocessing Utility ---
def preprocess_text(text, stop_words=None, lemmatizer=None):
    if stop_words is None:
        stop_words = set(stopwords.words('english'))
    if lemmatizer is None:
        lemmatizer = WordNetLemmatizer()
    text = re.sub('[^a-zA-Z]', ' ', text.lower())
    return ' '.join(lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words)

# --- Model Training ---
def train_model():
    try:
        df = pd.read_csv('Resumedataset.csv')
        print("Column names in Resumedataset.csv:", df.columns.tolist())
    except FileNotFoundError:
        st.error("Error: Resume dataset not found!")
        logging.error("Resume dataset not found.")
        return None, None

    df['Category'] = df['Category'].str.replace(' ', '_', regex=False)

    nltk.download('stopwords')
    nltk.download('wordnet')
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()

    cleaned_sentences = [preprocess_text(sentence, stop_words, lemmatizer) for sentence in df['Resume'].values]

    cv = CountVectorizer(max_features=20000)
    X = cv.fit_transform(cleaned_sentences).toarray()
    y = df['Category'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    return cv, model

# --- Database Initialization ---
def initialize_db():
    try:
        return mysql.connector.connect(
            user='root',
            password='password',
            host='127.0.0.1',
            database='resumes_database',
            auth_plugin='mysql_native_password'
        )
    except mysql.connector.Error as e:
        st.error(f"Database connection error: {e}")
        logging.error(f"Database connection error: {e}")
        return None

# --- Data Fetching ---
def fetch_positions(mydb):
    try:
        cur = mydb.cursor()
        cur.execute("SELECT position FROM skills")
        return [row[0] for row in cur.fetchall()]
    except mysql.connector.Error as e:
        st.error(f"Database error: {e}")
        logging.error(f"Database error: {e}")
        return []

def fetch_skills(mydb, position):
    try:
        cur = mydb.cursor()
        cur.execute("SELECT skills FROM skills WHERE LOWER(position) = %s", (position.lower(),))
        result = cur.fetchone()
        return result[0] if result else "Skills not found for this position."
    except mysql.connector.Error as e:
        st.error(f"Database error: {e}")
        logging.error(f"Database error: {e}")
        return ""

# --- Text Utilities with OCR fallback ---
def extract_text(file_path, file_type):
    try:
        if file_type == "text/plain":
            with open(file_path, "r", encoding="utf-8", errors='ignore') as f:
                return f.read()

        elif file_type == "application/pdf":
            text = ""
            with fitz.open(file_path) as doc:
                for page in doc:
                    text += page.get_text()

            if text.strip():
                return text  # Return if standard extraction works

            # Fallback: OCR scanned PDF pages
            st.warning("No text found using standard method. Running OCR...")
            with open(file_path, "rb") as f:
                images = convert_from_bytes(f.read(), dpi=300)

            ocr_text = ""
            for img in images:
                gray = img.convert('L')  # grayscale
                bw = gray.point(lambda x: 0 if x < 140 else 255, '1')  # binarize
                ocr_text += pytesseract.image_to_string(bw)

            return ocr_text.strip()

        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = Document(file_path)
            return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    except Exception as e:
        st.error(f"Error reading file: {e}")
        logging.error(f"OCR or file read error: {e}")
        return None

def is_valid_pdf(file_path):
    try:
        with open(file_path, "rb") as f:
            fitz.open(stream=f.read(), filetype="pdf")
        return True
    except Exception:
        return False

def calculate_score(resume_text, skills_text):
    resume_words = set(resume_text.lower().split())
    keywords = {kw.strip().lower() for kw in skills_text.split(',') if kw.strip()}
    matched = resume_words & keywords
    return (len(matched) / len(keywords)) * 100 if keywords else 0

# --- Save Resume ---
def save_resume(mydb, name, email, resume, score, location, category):
    try:
        cur = mydb.cursor()
        cur.execute(
            "INSERT INTO employees (Name, Email, Resume, score, location, category) VALUES (%s, %s, %s, %s, %s, %s)",
            (name, email, resume, score, location, category)
        )
        mydb.commit()
        st.success("Resume saved successfully!")
    except mysql.connector.Error as e:
        mydb.rollback()
        st.error(f"Database error: {e}")
        logging.error(f"Save error: {e}")

# --- Main App ---
def main():
    st.title("AI Resume Screening")

    cv = model = None
    if st.checkbox("Train Model (Takes time)"):
        cv, model = train_model()
        if cv and model:
            joblib.dump(model, 'RF.joblib')
            with open('cv.pickle', 'wb') as f:
                pickle.dump(cv, f)
            st.success("Model trained and saved!")

    if not (cv and model):
        try:
            model = joblib.load('RF.joblib')
            with open('cv.pickle', 'rb') as f:
                cv = pickle.load(f)
            st.info("Loaded pre-trained model.")
        except FileNotFoundError:
            st.error("Model files not found. Train the model first.")
            return

    mydb = initialize_db()
    if not mydb:
        return

    positions = fetch_positions(mydb)
    selected_position = st.selectbox("Select Position", positions)
    skills_text = fetch_skills(mydb, selected_position)
    if skills_text:
        st.write(f"Required Skills: {skills_text}")

    with st.form("resume_form"):
        name = st.text_input("Full Name")
        email = st.text_input("Email")
        location = st.text_input("Location")
        uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])
        submit = st.form_submit_button("Submit")

    if submit:
        if uploaded_file:
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    temp_path = tmp.name

                file_type = uploaded_file.type
                if file_type == "application/pdf" and not is_valid_pdf(temp_path):
                    raise ValueError("Uploaded file is not a valid PDF.")

                resume_text = extract_text(temp_path, file_type)
                if resume_text is None:
                    return

                nltk.download('stopwords')
                nltk.download('wordnet')
                stop_words = set(stopwords.words('english'))
                lemmatizer = WordNetLemmatizer()
                cleaned = preprocess_text(resume_text, stop_words, lemmatizer)
                score = calculate_score(cleaned, skills_text)

                resume_vector = cv.transform([cleaned])
                predicted_category = model.predict(resume_vector)[0]
                saved_category = selected_position.replace(" ", "_")

                st.write(f"Model Predicted Category: {predicted_category}")
                st.write(f"Saved Category: {saved_category}")
                st.write(f"Resume Score: {score:.2f}%")

                with open(temp_path, "rb") as f:
                    resume_data = f.read()

                save_resume(mydb, name, email, resume_data, score, location, saved_category)
            except ValueError as e:
                st.error(f"Error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
                logging.error(f"Error: {e}")
            finally:
                if temp_path:
                    os.unlink(temp_path)
        else:
            st.warning("Please upload a resume.")

    if mydb:
        mydb.close()

if __name__ == "__main__":
    main()
