import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer

# load dataset
df = pd.read_csv("data/processed/resume_dataset.csv")


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text


df["Resume_str"] = df["Resume_str"].apply(clean_text)


category_skills = {}

for category in df["Category"].unique():

    resumes = df[df["Category"] == category]["Resume_str"]

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=200,
        ngram_range=(1,2)
    )

    X = vectorizer.fit_transform(resumes)

    skills = vectorizer.get_feature_names_out()

    category_skills[category] = list(skills)


# save to python file
with open("app/category_skills.py", "w") as f:

    f.write("category_skills = {\n")

    for category, skills in category_skills.items():

        f.write(f'    "{category}": [\n')

        for skill in skills:

            f.write(f'        "{skill}",\n')

        f.write("    ],\n")

    f.write("}\n")


print("Category skill database created successfully")
print("Total categories:", len(category_skills))