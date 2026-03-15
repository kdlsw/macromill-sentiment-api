from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from macromill_sentiment.config import PreprocessConfig
from macromill_sentiment.data.load import load_imdb_csv
from macromill_sentiment.data.preprocess import clean_text, describe_preprocess


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    cfg = PreprocessConfig(strip_html=args.strip_html, lowercase=args.lowercase)

    df = load_imdb_csv(args.data_path)
    df["sentiment"] = df["sentiment"].map(_normalize_label)

    raw_reviews = df["review"].astype(str)
    cleaned_reviews = raw_reviews.map(lambda s: clean_text(s, cfg))

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "run_args.json").write_text(
        json.dumps(
            {
                "data_path": str(args.data_path),
                "output_dir": str(args.output_dir),
                "strip_html": bool(args.strip_html),
                "lowercase": bool(args.lowercase),
                "skip_ngrams": bool(args.skip_ngrams),
                "top_ngrams": int(args.top_ngrams),
                "max_features": int(args.max_features),
                "min_df": int(args.min_df),
                "ngram_sample_size": int(args.ngram_sample_size),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    summary = {
        "data_path": str(args.data_path),
        "n_rows": int(len(df)),
        "label_counts": df["sentiment"].value_counts().to_dict(),
        "preprocess": describe_preprocess(cfg),
        "contains_br_tag_rate": float(raw_reviews.str.contains(r"<br", case=False, regex=True).mean()),
    }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    _plot_class_balance(df["sentiment"], out_dir / "class_balance.png")
    _plot_length_histograms(cleaned_reviews, df["sentiment"], out_dir / "length_chars.png", kind="chars")
    _plot_length_histograms(cleaned_reviews, df["sentiment"], out_dir / "length_tokens.png", kind="tokens")
    _plot_html_presence(raw_reviews, out_dir / "html_presence.png")

    if not args.skip_ngrams:
        try:
            (out_dir / "ngram_config.json").write_text(
                json.dumps(
                    {
                        "top_ngrams": int(args.top_ngrams),
                        "max_features": int(args.max_features),
                        "min_df": int(args.min_df),
                        "ngram_sample_size": int(args.ngram_sample_size),
                        "ngram_range": [1, 2],
                        "stop_words": "sklearn_english_minus_negators",
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )

            if args.ngram_sample_size and args.ngram_sample_size < len(df):
                tmp = df[["sentiment"]].copy()
                tmp["cleaned_review"] = cleaned_reviews
                tmp = tmp.sample(n=args.ngram_sample_size, random_state=42)
                texts = tmp["cleaned_review"].to_list()
                labels = tmp["sentiment"].to_list()
            else:
                texts = cleaned_reviews.to_list()
                labels = df["sentiment"].to_list()

            ngram_report = _top_ngrams_by_class(
                texts,
                labels,
                n=args.top_ngrams,
                max_features=args.max_features,
                min_df=args.min_df,
            )
        except Exception as e:
            (out_dir / "top_ngrams_error.txt").write_text(str(e), encoding="utf-8")
        else:
            (out_dir / "top_ngrams.json").write_text(
                json.dumps(ngram_report, indent=2, sort_keys=False), encoding="utf-8"
            )
            _plot_wordclouds(ngram_report, out_dir, mode=args.wordcloud_mode, max_words=args.wordcloud_max_words)

    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="eda")
    p.add_argument("--data-path", type=Path, default=Path("work") / "IMDB Dataset.csv")
    p.add_argument("--output-dir", type=Path, default=Path("work") / "eda")
    p.add_argument("--strip-html", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--lowercase", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--skip-ngrams", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--top-ngrams", type=int, default=25)
    p.add_argument("--max-features", type=int, default=50_000)
    p.add_argument("--min-df", type=int, default=5)
    p.add_argument("--ngram-sample-size", type=int, default=10_000)
    p.add_argument("--wordcloud-mode", type=str, default="diff", choices=["by_class", "diff"])
    p.add_argument("--wordcloud-max-words", type=int, default=100)
    return p


def _normalize_label(label: str) -> str:
    s = str(label).strip().lower()
    if s in {"pos", "positive", "1", "true"}:
        return "positive"
    if s in {"neg", "negative", "0", "false"}:
        return "negative"
    return s


def _plot_class_balance(labels, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    vc = labels.value_counts()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(vc.index.astype(str), vc.values)
    ax.set_title("Class Balance")
    ax.set_xlabel("Sentiment")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


_WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def _plot_length_histograms(texts, labels, out_path: Path, *, kind: str) -> None:
    import matplotlib.pyplot as plt

    if kind == "chars":
        lengths = texts.map(len)
        title = "Review Length (Characters)"
    elif kind == "tokens":
        lengths = texts.map(lambda s: len(_WORD_RE.findall(s)))
        title = "Review Length (Tokens)"
    else:
        raise ValueError(f"Unknown kind: {kind!r}")

    fig, ax = plt.subplots(figsize=(7, 4))
    for cls in ["negative", "positive"]:
        subset = lengths[labels == cls].to_numpy()
        ax.hist(subset, bins=60, alpha=0.5, label=cls, density=True)
    ax.set_title(title)
    ax.set_xlabel("Length")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_html_presence(raw_texts, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    has_html = raw_texts.str.contains(r"<[^>]+>", regex=True)
    counts = {
        "contains_html": int(has_html.sum()),
        "no_html": int((~has_html).sum()),
    }
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(counts.keys()), list(counts.values()))
    ax.set_title("HTML Tag Presence in Raw Reviews")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _top_ngrams_by_class(
    texts: list[str],
    labels: list[str],
    *,
    n: int,
    max_features: int,
    min_df: int,
) -> dict:
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

    stop_words = set(ENGLISH_STOP_WORDS)
    stop_words -= {"no", "nor", "not"}
    stop_words_list = sorted(stop_words)

    vec = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        strip_accents="unicode",
        stop_words=stop_words_list,
    )
    X = vec.fit_transform(texts)
    y = np.asarray(labels)

    vocab = np.asarray(vec.get_feature_names_out())
    report: dict[str, object] = {
        "params": {
            "ngram_range": [1, 2],
            "min_df": int(min_df),
            "max_features": int(max_features),
            "stop_words": "sklearn_english_minus_negators",
        },
        "tfidf_mean_by_class": {},
    }

    mean_by_class: dict[str, np.ndarray] = {}
    for cls in ["negative", "positive"]:
        mask = y == cls
        if not np.any(mask):
            mean_by_class[cls] = np.zeros(len(vocab), dtype=float)
        else:
            mean_tfidf = X[mask].mean(axis=0)
            mean_by_class[cls] = np.asarray(mean_tfidf).ravel()

        top_idx = np.argsort(mean_by_class[cls])[::-1][:n]
        report["tfidf_mean_by_class"][cls] = [
            {"ngram": str(vocab[i]), "mean_tfidf": float(mean_by_class[cls][i])}
            for i in top_idx
            if mean_by_class[cls][i] > 0
        ]

    diff = mean_by_class["positive"] - mean_by_class["negative"]
    top_pos_idx = np.argsort(diff)[::-1][:n]
    top_neg_idx = np.argsort(diff)[:n]
    report["tfidf_mean_pos_minus_mean_neg"] = {
        "positive": [
            {
                "ngram": str(vocab[i]),
                "mean_diff": float(diff[i]),
                "mean_pos": float(mean_by_class["positive"][i]),
                "mean_neg": float(mean_by_class["negative"][i]),
            }
            for i in top_pos_idx
            if diff[i] > 0
        ],
        "negative": [
            {
                "ngram": str(vocab[i]),
                "mean_diff": float(diff[i]),
                "mean_pos": float(mean_by_class["positive"][i]),
                "mean_neg": float(mean_by_class["negative"][i]),
            }
            for i in top_neg_idx
            if diff[i] < 0
        ],
    }

    return report


def _plot_wordclouds(
    ngram_report: dict,
    out_dir: Path,
    *,
    mode: str,
    max_words: int = 100,
) -> None:
    """Generate word cloud images from n-gram analysis.

    Args:
        ngram_report: The n-gram analysis report dict (from _top_ngrams_by_class)
        out_dir: Output directory for images
        mode: 'by_class' or 'diff'
            - 'by_class': use tfidf_mean_by_class (raw TF-IDF scores)
            - 'diff': use tfidf_mean_pos_minus_mean_neg (discriminative power)
        max_words: Maximum number of words in each word cloud
    """
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt

    if mode == "by_class":
        data_key = "tfidf_mean_by_class"
        suffix = "by_class"
    elif mode == "diff":
        data_key = "tfidf_mean_pos_minus_mean_neg"
        suffix = "diff"
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    if data_key not in ngram_report:
        return

    data = ngram_report[data_key]

    # Color functions
    def green_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        import random
        r = random.randint(0, 50)
        g = random.randint(150, 220)
        b = random.randint(0, 50)
        return f"rgb({r}, {g}, {b})"

    def red_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        import random
        r = random.randint(200, 255)
        g = random.randint(0, 80)
        b = random.randint(0, 50)
        return f"rgb({r}, {g}, {b})"

    # Generate for positive and negative
    for sentiment, color_func in [("positive", green_color_func), ("negative", red_color_func)]:
        if sentiment not in data:
            continue

        items = data[sentiment]

        if mode == "by_class":
            # Use mean_tfidf as weight
            word_freq = {item["ngram"]: item["mean_tfidf"] for item in items}
            title = f"Word Cloud - {sentiment.title()} (TF-IDF Mean)"
        else:
            # Use absolute mean_diff as weight (for visibility)
            word_freq = {item["ngram"]: abs(item["mean_diff"]) for item in items}
            title = f"Word Cloud - {sentiment.title()} (Discriminative Power)"

        if not word_freq:
            continue

        wc = WordCloud(
            width=800,
            height=400,
            max_words=max_words,
            background_color="white",
            prefer_horizontal=0.7,
        )
        wc.generate_from_frequencies(word_freq)
        wc.recolor(color_func=color_func)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        ax.set_title(title, fontsize=14, pad=10)
        fig.tight_layout()
        fig.savefig(out_dir / f"wordcloud_{suffix}_{sentiment}.png", dpi=150)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
