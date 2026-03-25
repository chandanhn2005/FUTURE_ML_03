import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB

from preprocess import load_dataset
from feature_extraction import create_features


# load dataset
df = load_dataset()

X, tfidf = create_features(df["Resume_str"])
y = df["Category"]


# split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "svm": LinearSVC(class_weight="balanced"),
    "logistic": LogisticRegression(max_iter=2000),
    "random_forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ),
    "naive_bayes": MultinomialNB()
}


for name, model in models.items():

    print("Training", name)

    model.fit(X_train, y_train)

    joblib.dump(model, f"models/{name}_model.pkl")


# save vectorizer
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

print("\nModels trained successfully")