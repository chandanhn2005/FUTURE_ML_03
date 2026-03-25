import pandas as pd
import re


def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # remove new lines
    text = re.sub(r'\n', ' ', text)

    # remove numbers
    text = re.sub(r'\d+', ' ', text)

    # remove urls
    text = re.sub(r'http\S+', ' ', text)

    # remove special characters
    text = re.sub(r'[^a-zA-Z ]', ' ', text)

    # remove extra spaces
    text = re.sub(r'\s+', ' ', text)

    return text


def load_dataset():

    df = pd.read_csv("data/processed/resume_dataset.csv")

    df["Resume_str"] = df["Resume_str"].apply(clean_text)

    # remove extremely short resumes
    df = df[df["Resume_str"].str.len() > 100]

    return df