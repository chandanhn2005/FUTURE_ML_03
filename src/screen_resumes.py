import os
import joblib
import pdfplumber


def extract_text(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# load model
model = joblib.load("models/random_forest_model.pkl")
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


folder = "data/test_resumes"

for file in os.listdir(folder):

    if file.endswith(".pdf"):

        path = os.path.join(folder, file)

        text = extract_text(path)

        vector = tfidf.transform([text])

        prediction = model.predict(vector)

        print(file, "→", prediction[0])