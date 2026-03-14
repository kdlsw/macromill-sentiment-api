from __future__ import annotations


def list_models() -> tuple[str, ...]:
    return ("tfidf_lr", "tfidf_linearsvm", "roberta")


def build_model(model_name: str):
    if model_name == "tfidf_lr":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        stop_words = set(ENGLISH_STOP_WORDS)
        stop_words -= {"no", "nor", "not"}
        stop_words_list = sorted(stop_words)

        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=5,
                        max_features=50_000,
                        strip_accents="unicode",
                        stop_words=stop_words_list,
                    ),
                ),
                ("clf", LogisticRegression(max_iter=1000, solver="liblinear")),
            ]
        )
    if model_name == "tfidf_linearsvm":
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        stop_words = set(ENGLISH_STOP_WORDS)
        stop_words -= {"no", "nor", "not"}
        stop_words_list = sorted(stop_words)

        return Pipeline(
            steps=[
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=5,
                        max_features=50_000,
                        strip_accents="unicode",
                        stop_words=stop_words_list,
                    ),
                ),
                ("clf", LinearSVC()),
            ]
        )
    raise ValueError(f"Unknown model: {model_name!r}. Available: {list_models()}")
