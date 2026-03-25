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


# load trained model
model = joblib.load("models/random_forest_model.pkl")

# load vectorizer
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


def predict_resume(pdf_path):

    text = extract_text(pdf_path)

    vector = tfidf.transform([text])

    prediction = model.predict(vector)

    return prediction[0]


if __name__ == "__main__":

    pdf_file = "data/test_resume.pdf"

    result = predict_resume(pdf_file)

    print("\nPredicted Job Category:", result)