# 🚀 Resume Screening & Candidate Ranking System

## 📄 End-to-End Machine Learning Project for Automated Resume Analysis and Candidate Selection

This project builds a complete machine learning pipeline to automatically screen, analyze, and rank resumes based on job descriptions and required skills.

The system uses Natural Language Processing (NLP) and Machine Learning techniques to streamline the hiring process and reduce manual effort through an interactive Streamlit dashboard.

---

## 🌐 Live Application

🔗 Try the dashboard here:
https://futureml03-xoq47y6p3xhgn4ccqj772j.streamlit.app/

---

## 📌 Project Overview

Recruiters often spend significant time reviewing resumes manually. This project automates that process using AI.

The system:

* Analyzes resumes using NLP techniques
* Extracts relevant skills and keywords
* Compares resumes with job descriptions
* Ranks candidates based on similarity scores

The final application allows users to:

* Upload resumes
* Evaluate candidate profiles
* Rank candidates automatically
* Visualize insights through dashboards

---

## 🎯 Problem Statement

Manual resume screening is:

* Time-consuming
* Prone to human bias
* Inefficient for large-scale hiring

This project aims to build an intelligent system that:

* Automates resume evaluation
* Improves candidate selection accuracy
* Reduces recruitment time

---

## 📂 Dataset Information

The dataset contains resume data with:

* Candidate information
* Skills and experience text
* Job roles/categories
* Keywords and technical skills

Text data is processed using NLP techniques to extract meaningful insights.

---

## 🧠 Machine Learning Approach

The system uses text-based similarity and feature extraction:

* TF-IDF Vectorization
* Cosine Similarity
* Keyword Matching
* Skill-based scoring

These techniques help quantify how well a resume matches a job description.

---

## ⚙️ Machine Learning Pipeline

The project follows a structured workflow:

* Data Collection
* Text Cleaning & Preprocessing
* Feature Extraction (TF-IDF)
* Resume Parsing
* Similarity Calculation
* Candidate Ranking
* Visualization & Dashboard Deployment

---

## 📊 Dashboard Features

The deployed dashboard provides powerful functionalities:

### 📂 Resume Upload & Parsing

* Upload resumes (PDF)
* Extract text using PDF processing
* Clean and preprocess resume content

---

### 📊 Candidate Ranking System

* Compare resumes with job description
* Generate similarity scores
* Rank candidates automatically

---

### 📈 Data Visualization

* Skill distribution charts
* Similarity vs skill score analysis
* Final score comparison
* Interactive graphs using Plotly

---

## 📁 Project Structure

```
Resume_Screening/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── dataset/
│   └── processed/
│
├── src/
│   ├── preprocessing.py
│   ├── feature_extraction.py
│   ├── similarity.py
│   └── ranking.py
│
├── models/
└── README.md
```

---

## 📊 Model Evaluation

The system evaluates candidates based on:

* Similarity Score (TF-IDF + Cosine Similarity)
* Skill Matching Score
* Final Combined Score

These metrics help rank candidates effectively based on relevance.

---

## 💻 Technologies Used

### Programming Language

* Python

### Libraries

* Pandas
* NumPy
* Scikit-learn
* Plotly
* PDFPlumber

### Framework

* Streamlit

### Deployment

* Streamlit Cloud

---

## ▶️ How to Run the Project Locally

Clone the repository

```
git clone https://github.com/Deepakchakra/FUTURE_ML_03.git
```

Navigate to the project folder

```
cd FUTURE_ML_03
```

Install dependencies

```
pip install -r requirements.txt
```

Run the application

```
streamlit run app.py
```

---

## 📈 Future Improvements

* Advanced NLP models (BERT / Transformers)
* Better resume parsing accuracy
* Real-time job description matching
* Integration with ATS systems
* Multi-user authentication system

---

## 👨‍💻 Author

### Deepak Chakrasali

Machine Learning & AI Enthusiast

### GitHub

https://github.com/Deepakchakra

---
