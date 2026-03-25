import os
import pdfplumber
import pandas as pd

# dataset path
dataset_path = "data/pdf_resumes"

data = []

def extract_text(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text

    return text


# go through each category folder
for category in os.listdir(dataset_path):

    category_folder = os.path.join(dataset_path, category)

    if os.path.isdir(category_folder):

        for file in os.listdir(category_folder):

            if file.endswith(".pdf"):

                file_path = os.path.join(category_folder, file)

                try:
                    resume_text = extract_text(file_path)

                    data.append({
                        "Category": category,
                        "Resume_str": resume_text
                    })

                except:
                    print("Error reading:", file)


# convert to dataframe
df = pd.DataFrame(data)

# create processed folder if not exists
os.makedirs("data/processed", exist_ok=True)

# save dataset
df.to_csv("data/processed/resume_dataset.csv", index=False)

print("Dataset created successfully")
print("Total resumes:", len(df))