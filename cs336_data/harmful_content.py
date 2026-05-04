from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any

from fastwarc.warc import ArchiveIterator, WarcRecordType

# Use the utility functions you already implemented.
from filtering_helper import (
    extract_text_from_html_bytes,
    classify_nsfw,
    classify_toxic_speech,
)


def _open_warc(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def collect_harmful_predictions(
    warc_paths: list[Path],
    min_text_length: int = 1,
) -> list[dict[str, Any]]:
    """
    Extract text from WARC responses and run NSFW / toxic classifiers.
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
                    nsfw_label, nsfw_score = classify_nsfw(text)
                    toxic_label, toxic_score = classify_toxic_speech(text)
                except Exception:
                    continue

                try:
                    record_id = record.record_id
                except Exception:
                    record_id = None

                harmful_pred = (nsfw_label == "nsfw" or toxic_label == "toxic")

                results.append(
                    {
                        "warc_file": str(warc_path),
                        "record_id": record_id,
                        "text": text,
                        "nsfw_label": nsfw_label,
                        "nsfw_score": float(nsfw_score),
                        "toxic_label": toxic_label,
                        "toxic_score": float(toxic_score),
                        "harmful_pred": harmful_pred,
                    }
                )

    return results


def harmful_fraction(predictions: list[dict[str, Any]]) -> float:
    if not predictions:
        return 0.0
    return sum(1 for x in predictions if x["harmful_pred"]) / len(predictions)


def sample_balanced_examples(
    predictions: list[dict[str, Any]],
    harmful_count: int = 10,
    normal_count: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Force a balanced sample: N harmful + N non-harmful.
    If one side has fewer than requested, return as many as available.
    """
    rng = random.Random(seed)

    harmful = [x for x in predictions if x["harmful_pred"]]
    normal = [x for x in predictions if not x["harmful_pred"]]

    sampled_harmful = rng.sample(harmful, min(harmful_count, len(harmful)))
    sampled_normal = rng.sample(normal, min(normal_count, len(normal)))

    combined = sampled_harmful + sampled_normal
    rng.shuffle(combined)
    return combined


def write_annotation_template(
    sampled: list[dict[str, Any]],
    output_path: Path,
    preview_chars: int = 700,
) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(sampled, start=1):
            row = {
                "example_id": i,
                "warc_file": item["warc_file"],
                "record_id": item["record_id"],
                "predicted_nsfw": item["nsfw_label"],
                "predicted_nsfw_score": item["nsfw_score"],
                "predicted_toxic": item["toxic_label"],
                "predicted_toxic_score": item["toxic_score"],
                "predicted_harmful": item["harmful_pred"],
                "manual_nsfw": "",
                "manual_toxic": "",
                "text_preview": item["text"][:preview_chars],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_manual_annotations(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compare_with_manual(
    sampled: list[dict[str, Any]],
    annotations: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(sampled) != len(annotations):
        raise ValueError("Number of annotations must match number of sampled examples.")

    nsfw_errors: list[dict[str, Any]] = []
    toxic_errors: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []

    nsfw_correct = 0
    toxic_correct = 0

    for sample, ann in zip(sampled, annotations):
        pred_nsfw = sample["nsfw_label"]
        pred_toxic = sample["toxic_label"]

        manual_nsfw = ann["manual_nsfw"].strip().lower()
        manual_toxic = ann["manual_toxic"].strip().lower()

        if not manual_nsfw or not manual_toxic:
            raise ValueError("Found empty manual_nsfw or manual_toxic in annotation file.")

        nsfw_ok = pred_nsfw == manual_nsfw
        toxic_ok = pred_toxic == manual_toxic

        if nsfw_ok:
            nsfw_correct += 1
        else:
            nsfw_errors.append(
                {
                    "warc_file": sample["warc_file"],
                    "record_id": sample["record_id"],
                    "predicted_nsfw": pred_nsfw,
                    "manual_nsfw": manual_nsfw,
                    "score": sample["nsfw_score"],
                    "text_preview": sample["text"][:300],
                }
            )

        if toxic_ok:
            toxic_correct += 1
        else:
            toxic_errors.append(
                {
                    "warc_file": sample["warc_file"],
                    "record_id": sample["record_id"],
                    "predicted_toxic": pred_toxic,
                    "manual_toxic": manual_toxic,
                    "score": sample["toxic_score"],
                    "text_preview": sample["text"][:300],
                }
            )

        comparisons.append(
            {
                "predicted_nsfw": pred_nsfw,
                "manual_nsfw": manual_nsfw,
                "nsfw_score": sample["nsfw_score"],
                "predicted_toxic": pred_toxic,
                "manual_toxic": manual_toxic,
                "toxic_score": sample["toxic_score"],
            }
        )

    total = len(sampled)
    return {
        "num_total": total,
        "nsfw_accuracy": nsfw_correct / total if total else 0.0,
        "toxic_accuracy": toxic_correct / total if total else 0.0,
        "nsfw_errors": nsfw_errors,
        "toxic_errors": toxic_errors,
        "comparisons": comparisons,
    }


def suggest_threshold(
    comparisons: list[dict[str, Any]],
    pred_key: str,
    manual_key: str,
    score_key: str,
    positive_label: str,
    conservative_floor: float = 0.7,
    quantile: float = 0.1,
) -> float:
    good_scores = sorted(
        x[score_key]
        for x in comparisons
        if x[pred_key] == positive_label and x[manual_key] == positive_label
    )

    if not good_scores:
        return conservative_floor

    idx = int(quantile * (len(good_scores) - 1))
    empirical = good_scores[idx]
    return max(empirical, conservative_floor)


def write_report(
    predictions: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
    output_path: Path,
) -> None:
    frac_harmful = harmful_fraction(predictions)

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"Total extracted documents: {len(predictions)}\n")
        f.write(f"Fraction predicted harmful: {frac_harmful:.4f}\n")

        if comparison is not None:
            f.write(f"Manual sample size: {comparison['num_total']}\n")
            f.write(f"NSFW accuracy on sample: {comparison['nsfw_accuracy']:.4f}\n")
            f.write(f"Toxic accuracy on sample: {comparison['toxic_accuracy']:.4f}\n")
            f.write(f"NSFW classifier errors: {len(comparison['nsfw_errors'])}\n")
            f.write(f"Toxic classifier errors: {len(comparison['toxic_errors'])}\n")

            nsfw_threshold = suggest_threshold(
                comparison["comparisons"],
                pred_key="predicted_nsfw",
                manual_key="manual_nsfw",
                score_key="nsfw_score",
                positive_label="nsfw",
                conservative_floor=0.7,
            )
            toxic_threshold = suggest_threshold(
                comparison["comparisons"],
                pred_key="predicted_toxic",
                manual_key="manual_toxic",
                score_key="toxic_score",
                positive_label="toxic",
                conservative_floor=0.7,
            )

            f.write(f"Suggested NSFW threshold: {nsfw_threshold:.4f}\n")
            f.write(f"Suggested toxic threshold: {toxic_threshold:.4f}\n\n")

            f.write("NSFW errors:\n")
            for err in comparison["nsfw_errors"]:
                f.write(json.dumps(err, ensure_ascii=False) + "\n")

            f.write("\nToxic errors:\n")
            for err in comparison["toxic_errors"]:
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
        "--harmful_count",
        type=int,
        default=10,
        help="Number of predicted harmful examples to sample",
    )
    parser.add_argument(
        "--normal_count",
        type=int,
        default=10,
        help="Number of predicted non-harmful examples to sample",
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
        default="harmful_content_sample.jsonl",
        help="Path to write/read the manual annotation template",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="harmful_content_report.txt",
        help="Path to write the final report summary",
    )
    parser.add_argument(
        "--mode",
        choices=["sample", "report"],
        required=True,
        help=(
            "sample: run predictions and write a balanced annotation file; "
            "report: run predictions again and compare against completed annotations"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    warc_paths = [Path(p) for p in args.warc_paths]
    annotation_path = Path(args.annotation_file)
    report_path = Path(args.report_file)

    predictions = collect_harmful_predictions(
        warc_paths=warc_paths,
        min_text_length=args.min_text_length,
    )

    sampled = sample_balanced_examples(
        predictions=predictions,
        harmful_count=args.harmful_count,
        normal_count=args.normal_count,
        seed=args.seed,
    )

    harmful_available = sum(1 for x in predictions if x["harmful_pred"])
    normal_available = sum(1 for x in predictions if not x["harmful_pred"])

    if args.mode == "sample":
        write_annotation_template(sampled, annotation_path)

        print(f"Wrote balanced annotation template to: {annotation_path}")
        print(f"Total extracted documents: {len(predictions)}")
        print(f"Fraction predicted harmful: {harmful_fraction(predictions):.4f}")
        print(f"Available harmful examples: {harmful_available}")
        print(f"Available non-harmful examples: {normal_available}")
        print(f"Sampled harmful examples: {sum(1 for x in sampled if x['harmful_pred'])}")
        print(f"Sampled non-harmful examples: {sum(1 for x in sampled if not x['harmful_pred'])}")

    elif args.mode == "report":
        annotations = load_manual_annotations(annotation_path)
        comparison = compare_with_manual(sampled, annotations)
        write_report(predictions, comparison, report_path)

        print(f"Wrote report to: {report_path}")
        print(f"Total extracted documents: {len(predictions)}")
        print(f"Fraction predicted harmful: {harmful_fraction(predictions):.4f}")
        print(f"NSFW accuracy on sample: {comparison['nsfw_accuracy']:.4f}")
        print(f"Toxic accuracy on sample: {comparison['toxic_accuracy']:.4f}")
        print(f"NSFW classifier errors: {len(comparison['nsfw_errors'])}")
        print(f"Toxic classifier errors: {len(comparison['toxic_errors'])}")


if __name__ == "__main__":
    main()