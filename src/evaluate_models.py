import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from preprocess import load_dataset
from feature_extraction import create_features


# load dataset
df = load_dataset()

X, tfidf = create_features(df["Resume_str"])
y = df["Category"]


# split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


models = {
    "SVM": joblib.load("models/svm_model.pkl"),
    "Logistic Regression": joblib.load("models/logistic_model.pkl"),
    "Random Forest": joblib.load("models/random_forest_model.pkl"),
    "Naive Bayes": joblib.load("models/naive_bayes_model.pkl")
}


print("\nModel Accuracy Comparison\n")

for name, model in models.items():

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(name, "Accuracy:", round(accuracy * 100, 2), "%")