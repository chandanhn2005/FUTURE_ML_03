from sklearn.feature_extraction.text import TfidfVectorizer


def create_features(resumes):

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )

    X = tfidf.fit_transform(resumes)

    return X, tfidf