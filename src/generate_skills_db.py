import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer


# load dataset
df = pd.read_csv("data/processed/resume_dataset.csv")


# basic cleaning
def clean(text):

    text = str(text).lower()

    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    text = re.sub(r'\s+', ' ', text)

    return text


df["Resume_str"] = df["Resume_str"].apply(clean)


# TF-IDF extraction
vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=2000,
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(df["Resume_str"])


skills = vectorizer.get_feature_names_out()


# filter useful skills
skills_list = []

for skill in skills:

    if len(skill) > 2:
        skills_list.append(skill)


# save as python file
with open("app/skills_db.py", "w") as f:

    f.write("skills_db = [\n")

    for skill in skills_list:

        f.write(f'    "{skill}",\n')

    f.write("]")


print("skills_db created successfully")
print("Total skills:", len(skills_list))