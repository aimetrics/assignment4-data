from __future__ import annotations

import argparse
import os
import gzip
import random
from pathlib import Path
from typing import Any

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType

# Import your existing utilities.
from filtering_helper import (
    extract_text_from_html_bytes,
    identify_language,
    gopher_quality_filter,
)

# https://huggingface.co/ryen12/cs336-data-collection
DEFAULT_MODEL_PATH = os.path.expanduser("~/fanchi/proj/opensource/model/quality_classifier/quality_classifier.bin")
DEFAULT_POSITIVE_WARC = "subsampled_positive_urls.warc.gz"
DEFAULT_NEGATIVE_WARC = "CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
DEFAULT_TRAIN_FILE = "~/fanchi/proj/opensource/model/quality_classifier/quality_train.txt"


def _open_maybe_gzip(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def normalize_text(text: str) -> str:
    return " ".join(text.split())


def keep_positive_training_text(text: str, min_chars: int = 200) -> bool:
    """
    Apply light filtering before using a page as a training example.
    """
    text = normalize_text(text)
    if len(text) < min_chars:
        return False

    try:
        lang, score = identify_language(text)
    except Exception:
        return False

    if lang != "en" or score < 0.7:
        return False

    try:
        if not gopher_quality_filter(text):
            return False
    except Exception:
        return False

    return True


def keep_negative_training_text(
    text: str,
    min_chars: int = 50,
    require_english: bool = True,
) -> bool:
    """
    Keep *representative* negatives:
    - apply only light normalization/length checks,
    - optionally keep to English to avoid the classifier learning language
      instead of quality,
    - do NOT run Gopher quality filter (otherwise we erase low-quality signal).
    """
    text = normalize_text(text)
    if len(text) < min_chars:
        return False

    if not require_english:
        return True

    try:
        lang, score = identify_language(text)
    except Exception:
        return False

    return lang == "en" and score >= 0.5


def read_warc_texts(
    warc_paths: list[Path],
    max_docs: int | None = None,
    min_chars: int = 200,
    label_type: str = "positive",
    require_english_for_negative: bool = True,
) -> list[str]:
    """
    Extract English, reasonably clean texts from WARC files.
    """
    docs: list[str] = []

    for warc_path in warc_paths:
        with _open_maybe_gzip(warc_path) as f:
            for record in ArchiveIterator(f, parse_http=False):
                if record.record_type != WarcRecordType.response:
                    continue

                try:
                    payload = record.reader.read()
                    text = extract_text_from_html_bytes(payload)
                except Exception:
                    continue

                if not text:
                    continue

                text = normalize_text(text)
                if label_type == "positive":
                    keep = keep_positive_training_text(text, min_chars=min_chars)
                elif label_type == "negative":
                    keep = keep_negative_training_text(
                        text,
                        min_chars=min_chars,
                        require_english=require_english_for_negative,
                    )
                else:
                    raise ValueError("label_type must be one of: positive, negative")

                if keep:
                    docs.append(text)

                if max_docs is not None and len(docs) >= max_docs:
                    return docs

    return docs


def write_fasttext_training_file(
    positive_docs: list[str],
    negative_docs: list[str],
    out_path: Path,
    seed: int = 42,
) -> None:
    """
    Write fastText supervised training data:
      __label__hq <text>
      __label__lq <text>
    """
    rng = random.Random(seed)

    rows = [f"__label__hq {x}" for x in positive_docs]
    rows += [f"__label__lq {x}" for x in negative_docs]
    rng.shuffle(rows)

    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(row + "\n")


def train_quality_model(
    train_file: Path,
    model_out: Path,
) -> None:
    model = fasttext.train_supervised(
        input=str(train_file),
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minn=2,
        maxn=5,
        dim=100,
        loss="softmax",
        thread=8,
    )
    model.save_model(str(model_out))


def train_from_warcs(
    positive_warcs: list[Path],
    negative_warcs: list[Path],
    train_out: Path = Path(DEFAULT_TRAIN_FILE),
    model_out: Path = Path(DEFAULT_MODEL_PATH),
    max_positive: int = 50000,
    max_negative: int = 50000,
    min_positive_chars: int = 200,
    min_negative_chars: int = 50,
    require_english_for_negative: bool = True,
) -> tuple[int, int]:
    """
    End-to-end pipeline:
      1) read/clean positive + negative WARC docs
      2) write fastText training file
      3) train and save classifier model
    Returns (num_positive_docs, num_negative_docs).
    """
    positive_docs = read_warc_texts(
        positive_warcs,
        max_docs=max_positive,
        min_chars=min_positive_chars,
        label_type="positive",
    )
    negative_docs = read_warc_texts(
        negative_warcs,
        max_docs=max_negative,
        min_chars=min_negative_chars,
        label_type="negative",
        require_english_for_negative=require_english_for_negative,
    )
    n = min(len(positive_docs), len(negative_docs))
    positive_docs = positive_docs[:n]
    negative_docs = negative_docs[:n]
    write_fasttext_training_file(
        positive_docs=positive_docs,
        negative_docs=negative_docs,
        out_path=train_out,
    )
    train_quality_model(
        train_file=train_out,
        model_out=model_out,
    )
    return len(positive_docs), len(negative_docs)


def _load_quality_model(model_path: str | Path = DEFAULT_MODEL_PATH):
    return fasttext.load_model(str(model_path))


def classify_quality(
    text: str,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    threshold: float = 0.5,
) -> tuple[Any, float]:
    """
    Return:
      ('high_quality', score) or ('low_quality', score)
    """
    text = normalize_text(text)
    if len(text) < 50:
        return "low_quality", 1.0

    model = _load_quality_model(model_path)
    labels, probs = model.predict(text.replace("\n", " "), k=2)

    label = labels[0]
    score = float(probs[0])

    if label.startswith("__label__"):
        label = label[len("__label__"):]

    mapping = {
        "hq": "high_quality",
        "high_quality": "high_quality",
        "1": "high_quality",
        "lq": "low_quality",
        "low_quality": "low_quality",
        "0": "low_quality",
    }

    normalized_label = mapping.get(label, label)
    if normalized_label == "high_quality" and score < threshold:
        return "low_quality", 1.0 - score
    if normalized_label == "low_quality" and (1.0 - score) < threshold:
        return "high_quality", 1.0 - score
    return normalized_label, score


def run_classify_quality(text: str) -> tuple[Any, float]:
    """
    Adapter-style helper for tests.
    Assumes quality_classifier.bin exists in the working directory.
    """
    return classify_quality(text)


def parse_args():
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="command", required=True)

    prep = subparsers.add_parser("prepare_data")
    prep.add_argument("--positive_warcs", nargs="+", required=True)
    prep.add_argument("--negative_warcs", nargs="+", required=True)
    prep.add_argument("--max_positive", type=int, default=50000)
    prep.add_argument("--max_negative", type=int, default=50000)
    prep.add_argument("--min_positive_chars", type=int, default=200)
    prep.add_argument("--min_negative_chars", type=int, default=50)
    prep.add_argument(
        "--no_require_english_for_negative",
        action="store_true",
        help="Keep negative examples even when language ID is not English.",
    )
    prep.add_argument("--output", default=DEFAULT_TRAIN_FILE)

    train = subparsers.add_parser("train")
    train.add_argument("--train_file", required=True)
    train.add_argument("--model_out", default=DEFAULT_MODEL_PATH)

    end2end = subparsers.add_parser("train_from_warcs")
    end2end.add_argument(
        "--positive_warcs",
        nargs="+",
        default=[DEFAULT_POSITIVE_WARC],
    )
    end2end.add_argument(
        "--negative_warcs",
        nargs="+",
        default=[DEFAULT_NEGATIVE_WARC],
    )
    end2end.add_argument("--max_positive", type=int, default=50000)
    end2end.add_argument("--max_negative", type=int, default=50000)
    end2end.add_argument("--min_positive_chars", type=int, default=200)
    end2end.add_argument("--min_negative_chars", type=int, default=50)
    end2end.add_argument(
        "--no_require_english_for_negative",
        action="store_true",
        help="Keep negative examples even when language ID is not English.",
    )
    end2end.add_argument("--train_out", default=DEFAULT_TRAIN_FILE)
    end2end.add_argument("--model_out", default=DEFAULT_MODEL_PATH)

    infer = subparsers.add_parser("infer")
    infer.add_argument("--text", required=True)
    infer.add_argument("--model_path", default=DEFAULT_MODEL_PATH)
    infer.add_argument("--threshold", type=float, default=0.5)

    return parser.parse_args()


def main():
    args = parse_args()

    if args.command == "prepare_data":
        positive_docs = read_warc_texts(
            [Path(p) for p in args.positive_warcs],
            max_docs=args.max_positive,
            min_chars=args.min_positive_chars,
            label_type="positive",
        )
        negative_docs = read_warc_texts(
            [Path(p) for p in args.negative_warcs],
            max_docs=args.max_negative,
            min_chars=args.min_negative_chars,
            label_type="negative",
            require_english_for_negative=not args.no_require_english_for_negative,
        )

        write_fasttext_training_file(
            positive_docs=positive_docs,
            negative_docs=negative_docs,
            out_path=Path(args.output),
        )

        print(f"positives: {len(positive_docs)}")
        print(f"negatives: {len(negative_docs)}")
        print(f"wrote training file to: {args.output}")

    elif args.command == "train":
        train_quality_model(
            train_file=Path(args.train_file),
            model_out=Path(args.model_out),
        )
        print(f"saved model to: {args.model_out}")

    elif args.command == "train_from_warcs":
        pos_count, neg_count = train_from_warcs(
            positive_warcs=[Path(p) for p in args.positive_warcs],
            negative_warcs=[Path(p) for p in args.negative_warcs],
            train_out=Path(args.train_out),
            model_out=Path(args.model_out),
            max_positive=args.max_positive,
            max_negative=args.max_negative,
            min_positive_chars=args.min_positive_chars,
            min_negative_chars=args.min_negative_chars,
            require_english_for_negative=not args.no_require_english_for_negative,
        )
        print(f"positives kept: {pos_count}")
        print(f"negatives kept: {neg_count}")
        print(f"wrote training file to: {args.train_out}")
        print(f"saved model to: {args.model_out}")

    elif args.command == "infer":
        label, score = classify_quality(
            args.text,
            model_path=args.model_path,
            threshold=args.threshold,
        )
        print(label, score)


if __name__ == "__main__":
    main()