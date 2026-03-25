import streamlit as st
import pdfplumber
import re
import plotly.graph_objects as go
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from skills_db import skills_db
from category_skills import category_skills

# Make layout wide (dashboard look)
st.set_page_config(layout="wide")


# -------------------- FUNCTIONS --------------------

def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
    return text.lower()


def detect_skills(resume_text):
    found_skills = []
    for skill in skills_db:
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, resume_text):
            found_skills.append(skill)
    return sorted(found_skills)


def rank_resumes(uploaded_text, required_skills):

    df = pd.read_csv("data/processed/resume_dataset.csv")
    df["Resume_str"] = df["Resume_str"].astype(str)

    resumes = df["Resume_str"].tolist()
    resumes.insert(0, uploaded_text)

    vectorizer = TfidfVectorizer(stop_words="english")
    vectors = vectorizer.fit_transform(resumes)

    similarity = cosine_similarity(vectors[0:1], vectors[1:]).flatten()

    results = []

    for i, row in df.iterrows():

        resume_text = row["Resume_str"].lower()

        matched_skills = [skill for skill in required_skills if skill in resume_text]
        missing_skills = list(set(required_skills) - set(matched_skills))

        skill_score = (len(matched_skills) / len(required_skills)) * 100 if required_skills else 0
        similarity_score = similarity[i] * 100

        final_score = (0.6 * similarity_score) + (0.4 * skill_score)

        results.append({
            "Category": row["Category"],
            "Similarity Score": round(similarity_score, 2),
            "Skill Score": round(skill_score, 2),
            "Final Score": round(final_score, 2),
            "Matched Skills": ", ".join(matched_skills),
            "Missing Skills": ", ".join(missing_skills)
        })

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(by="Final Score", ascending=False)

    return result_df.head(25)


# -------------------- UI --------------------

st.title("AI Resume Screening System")

uploaded_file = st.file_uploader("Upload Resume", type=["pdf"])

job_role = st.selectbox(
    "Select Job Role",
    list(category_skills.keys()),
    key="job_role_select"
)

# Engineering branch selection
if job_role == "ENGINEERING":

    branch = st.selectbox(
        "Select Engineering Branch",
        list(category_skills["ENGINEERING"].keys()),
        key="engineering_branch_select"
    )

    required_skills = category_skills["ENGINEERING"][branch]

else:
    required_skills = category_skills[job_role]


# -------------------- PREDICT --------------------

if st.button("Predict"):

    if uploaded_file is None:
        st.warning("Please upload a resume first.")

    else:

        resume_text = extract_text(uploaded_file)
        detected_skills = detect_skills(resume_text)

        # ---------------- Ranking ----------------
        top_resumes = rank_resumes(resume_text, required_skills)

        top_resumes = top_resumes.reset_index(drop=True)
        top_resumes.index += 1

        st.subheader("Top Matching Resumes")
        st.dataframe(top_resumes)

        # ---------------- DASHBOARD ----------------
        st.subheader("📊 Resume Ranking Dashboard")

        top_candidate = top_resumes.iloc[0]

        # First row (2 columns)
        col1, col2 = st.columns(2)

        # LEFT → Final Score
        with col1:
            st.markdown("### 📊 Final Score Comparison")
            bar_data = top_resumes[["Category", "Final Score"]].set_index("Category")
            st.bar_chart(bar_data)

        # RIGHT → Skill Distribution
        with col2:
            st.markdown("### 🥧 Skill Distribution")

            matched = len(top_candidate["Matched Skills"].split(", ")) if top_candidate["Matched Skills"] else 0
            missing = len(top_candidate["Missing Skills"].split(", ")) if top_candidate["Missing Skills"] else 0

            pie_data = pd.DataFrame({
                "Type": ["Matched Skills", "Missing Skills"],
                "Count": [matched, missing]
            })

            st.bar_chart(pie_data.set_index("Type"))

        # FULL WIDTH → Similarity vs Skill Score
        st.markdown("### 📈 Similarity vs Skill Score")

        line_data = top_resumes[["Category", "Similarity Score", "Skill Score"]]
        line_data = line_data.set_index("Category")

        st.line_chart(line_data)

        # ---------------- Skill Analysis ----------------

        present_skills = [
            skill for skill in required_skills if skill in detected_skills
        ]

        missing_skills = list(set(required_skills) - set(present_skills))

        match_score = (len(present_skills) / len(required_skills)) * 100 if required_skills else 0

        st.subheader("Detected Skills from Resume")
        st.write(detected_skills)

        st.subheader("Skills Present for Selected Role")
        st.success(present_skills)

        st.subheader("Missing Skills")
        st.error(missing_skills)

        st.subheader("Skill Match Score")
        st.progress(int(match_score))
        st.write(round(match_score, 2), "% match")

        # ---------------- ATS SCORE ----------------

        ats_score = round(match_score, 2)

        st.subheader("ATS Score Meter")

        chart_placeholder = st.empty()

        for value in range(0, int(ats_score) + 1, 2):

            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=value,
                title={'text': "ATS Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "red"},
                        {'range': [50, 75], 'color': "orange"},
                        {'range': [75, 100], 'color': "green"}
                    ],
                }
            ))

            chart_placeholder.plotly_chart(fig, width="stretch")
            time.sleep(0.02)

