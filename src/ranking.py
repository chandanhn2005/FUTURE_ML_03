import os
import joblib
import pdfplumber
from sklearn.metrics.pairwise import cosine_similarity


def extract_text(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# load vectorizer
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


def rank_resumes(job_description, folder):

    resumes = []
    names = []

    for file in os.listdir(folder):

        if file.endswith(".pdf"):

            path = os.path.join(folder, file)

            text = extract_text(path)

            resumes.append(text)

            names.append(file)

    # transform resumes
    resume_vectors = tfidf.transform(resumes)

    # transform job description
    job_vector = tfidf.transform([job_description])

    scores = cosine_similarity(resume_vectors, job_vector)

    ranking = []

    for i in range(len(names)):

        ranking.append((names[i], scores[i][0]))

    ranking.sort(key=lambda x: x[1], reverse=True)

    return ranking