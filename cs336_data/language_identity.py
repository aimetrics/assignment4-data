from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any

from fastwarc.warc import ArchiveIterator, WarcRecordType

# Use the utility functions directly from your experiment code.
from filtering_helper import (
    extract_text_from_html_bytes,
    identify_language,
)


def _open_warc(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def collect_language_predictions(
    warc_paths: list[Path],
    min_text_length: int = 1,
) -> list[dict[str, Any]]:
    """
    Extract text from WARC responses using the utility extraction function,
    then run language identification using the utility language-ID function.

    This intentionally does not add extra filtering heuristics beyond skipping
    empty/very short extracted texts, so that the experiment follows the
    experiment setup closely.
    """
    results: list[dict[str, Any]] = []

    for warc_path in warc_paths:
        with _open_warc(warc_path) as f:
            for record in ArchiveIterator(f, parse_http=False):
                if record.record_type != WarcRecordType.response:
                    continue
                try:
                    payload = record.reader.read()
                except Exception:
                    continue

                text = extract_text_from_html_bytes(payload)
                if text is None:
                    continue

                text = " ".join(text.split())
                if len(text) < min_text_length:
                    continue

                try:
                    lang, score = identify_language(text)
                except Exception:
                    continue

                try:
                    record_id = record.record_id
                except Exception:
                    record_id = None

                results.append(
                    {
                        "warc_file": str(warc_path),
                        "record_id": record_id,
                        "lang": lang,
                        "score": float(score),
                        "text": text,
                    }
                )

    return results


def sample_examples(
    predictions: list[dict[str, Any]],
    sample_size: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if len(predictions) <= sample_size:
        return list(predictions)
    return rng.sample(predictions, sample_size)


def english_fraction(predictions: list[dict[str, Any]]) -> float:
    if not predictions:
        return 0.0
    num_english = sum(1 for x in predictions if x["lang"] == "en")
    return num_english / len(predictions)


def write_annotation_template(
    sampled: list[dict[str, Any]],
    output_path: Path,
    preview_chars: int = 500,
) -> None:
    """
    Write a JSONL file for manual annotation.

    The user should fill in 'manual_lang' for each example.
    """
    with output_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(sampled, start=1):
            row = {
                "example_id": i,
                "warc_file": item["warc_file"],
                "record_id": item["record_id"],
                "predicted_lang": item["lang"],
                "score": item["score"],
                "manual_lang": "",
                "text_preview": item["text"][:preview_chars].replace("\n", " "),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manual_annotations(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compare_predictions_with_manual(
    sampled: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(sampled) != len(annotations):
        raise ValueError("Number of annotations must match number of sampled examples.")

    errors: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    correct = 0

    for sample, ann in zip(sampled, annotations):
        predicted = sample["lang"]
        manual = ann["manual_lang"].strip().lower()

        if not manual:
            raise ValueError("Found empty manual_lang in annotation file.")

        is_correct = predicted == manual
        if is_correct:
            correct += 1
        else:
            errors.append(
                {
                    "warc_file": sample["warc_file"],
                    "record_id": sample["record_id"],
                    "predicted_lang": predicted,
                    "manual_lang": manual,
                    "score": sample["score"],
                    "text_preview": sample["text"][:300].replace("\n", " "),
                }
            )

        comparisons.append(
            {
                "predicted_lang": predicted,
                "manual_lang": manual,
                "score": sample["score"],
                "correct": is_correct,
            }
        )

    total = len(sampled)
    accuracy = correct / total if total else 0.0

    return {
        "num_total": total,
        "num_correct": correct,
        "accuracy": accuracy,
        "errors": errors,
        "comparisons": comparisons,
    }


def suggest_english_threshold(
    comparisons: list[dict[str, Any]],
    quantile: float = 0.1,
    conservative_floor: float = 0.7,
) -> float:
    """
    Suggest a threshold based on manually verified English examples.

    We compute a low quantile over correctly labeled English pages, but clamp the
    result to a conservative floor because low-score pages are often noisy or
    low-information even if their language label is technically correct.
    """
    good_english_scores = sorted(
        x["score"]
        for x in comparisons
        if x["predicted_lang"] == "en" and x["manual_lang"] == "en"
    )

    if not good_english_scores:
        return conservative_floor

    idx = int(quantile * (len(good_english_scores) - 1))
    empirical_threshold = good_english_scores[idx]
    return max(empirical_threshold, conservative_floor)


def write_report(
    predictions: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
    output_path: Path,
) -> None:
    frac_en = english_fraction(predictions)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"Total extracted documents: {len(predictions)}\n")
        f.write(f"Fraction predicted English: {frac_en:.4f}\n")

        if comparison is not None:
            f.write(f"Manual sample size: {comparison['num_total']}\n")
            f.write(f"Manual sample accuracy: {comparison['accuracy']:.4f}\n")
            f.write(f"Classifier errors in sample: {len(comparison['errors'])}\n")

            threshold = suggest_english_threshold(comparison["comparisons"])
            f.write(f"Suggested English confidence threshold: {threshold:.4f}\n\n")

            f.write("Errors:\n")
            for err in comparison["errors"]:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warc_paths",
        nargs="+",
        required=True,
        help="Paths to WARC or WARC.GZ files",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=20,
        help="Number of random examples to manually inspect",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=1,
        help="Skip extracted texts shorter than this",
    )
    parser.add_argument(
        "--annotation_file",
        type=str,
        default="language_id_sample.jsonl",
        help="Path to write/read the manual annotation template",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="language_id_report.txt",
        help="Path to write the final report summary",
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "report"],
        required=True,
        help=(
            "sample: run predictions and write a sample annotation file; "
            "report: run predictions again and compare against completed annotations"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    warc_paths = [Path(p) for p in args.warc_paths]
    annotation_path = Path(args.annotation_file)
    report_path = Path(args.report_file)

    predictions = collect_language_predictions(
        warc_paths=warc_paths,
        min_text_length=args.min_text_length,
    )

    if args.mode == "sample":
        sampled = sample_examples(
            predictions=predictions,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        write_annotation_template(sampled, annotation_path)

        print(f"Wrote manual annotation template to: {annotation_path}")
        print(f"Total extracted documents: {len(predictions)}")
        print(f"Fraction predicted English: {english_fraction(predictions):.4f}")

    elif args.mode == "report":
        sampled = sample_examples(
            predictions=predictions,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        annotations = load_manual_annotations(annotation_path)
        comparison = compare_predictions_with_manual(sampled, annotations)
        write_report(predictions, comparison, report_path)

        print(f"Wrote report to: {report_path}")
        print(f"Total extracted documents: {len(predictions)}")
        print(f"Fraction predicted English: {english_fraction(predictions):.4f}")
        print(f"Manual sample accuracy: {comparison['accuracy']:.4f}")
        print(f"Classifier errors in sample: {len(comparison['errors'])}")
        print(
            "Suggested English confidence threshold: "
            f"{suggest_english_threshold(comparison['comparisons']):.4f}"
        )


if __name__ == "__main__":
    main()