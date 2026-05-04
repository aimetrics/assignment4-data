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
    gopher_quality_filter,
)


def _open_warc(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def collect_quality_predictions(
    warc_paths: list[Path],
    min_text_length: int = 1,
) -> list[dict[str, Any]]:
    """
    Extract text from WARC responses using the utility extraction function,
    then run the utility gopher_quality_filter on each extracted document.
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
                    passed = bool(gopher_quality_filter(text))
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
                        "text": text,
                        "passed_filter": passed,
                    }
                )

    return results


def sample_balanced_examples(
    predictions: list[dict[str, Any]],
    pass_count: int = 10,
    fail_count: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Force a balanced sample for manual inspection:
      - pass_count examples that pass the quality filter
      - fail_count examples that fail the quality filter
    """
    rng = random.Random(seed)

    passed = [x for x in predictions if x["passed_filter"]]
    failed = [x for x in predictions if not x["passed_filter"]]

    sampled_passed = rng.sample(passed, min(pass_count, len(passed)))
    sampled_failed = rng.sample(failed, min(fail_count, len(failed)))

    combined = sampled_passed + sampled_failed
    rng.shuffle(combined)
    return combined


def write_annotation_template(
    sampled: list[dict[str, Any]],
    output_path: Path,
    preview_chars: int = 700,
) -> None:
    """
    Write a JSONL file for manual annotation.

    Fill in:
      - manual_quality: "pass" or "fail"
      - notes: brief reason if your judgment differs from the rule-based filter
    """
    with output_path.open("w", encoding="utf-8") as f:
        for i, item in enumerate(sampled, start=1):
            row = {
                "example_id": i,
                "warc_file": item["warc_file"],
                "record_id": item["record_id"],
                "predicted_quality": "pass" if item["passed_filter"] else "fail",
                "manual_quality": "",
                "notes": "",
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

    errors: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    correct = 0

    for sample, ann in zip(sampled, annotations):
        predicted = "pass" if sample["passed_filter"] else "fail"
        manual = ann["manual_quality"].strip().lower()

        if manual not in {"pass", "fail"}:
            raise ValueError("manual_quality must be 'pass' or 'fail' for every example.")

        is_correct = predicted == manual
        if is_correct:
            correct += 1
        else:
            errors.append(
                {
                    "warc_file": sample["warc_file"],
                    "record_id": sample["record_id"],
                    "predicted_quality": predicted,
                    "manual_quality": manual,
                    "notes": ann.get("notes", ""),
                    "text_preview": sample["text"][:300],
                }
            )

        comparisons.append(
            {
                "predicted_quality": predicted,
                "manual_quality": manual,
            }
        )

    total = len(sampled)
    return {
        "num_total": total,
        "num_correct": correct,
        "accuracy": correct / total if total else 0.0,
        "errors": errors,
        "comparisons": comparisons,
    }


def write_report(
    predictions: list[dict[str, Any]],
    comparison: dict[str, Any] | None,
    output_path: Path,
) -> None:
    pass_fraction = (
        sum(1 for x in predictions if x["passed_filter"]) / len(predictions)
        if predictions else 0.0
    )

    with output_path.open("w", encoding="utf-8") as f:
        f.write(f"Total extracted documents: {len(predictions)}\n")
        f.write(f"Fraction passing quality filter: {pass_fraction:.4f}\n")

        if comparison is not None:
            f.write(f"Manual sample size: {comparison['num_total']}\n")
            f.write(f"Manual agreement: {comparison['accuracy']:.4f}\n")
            f.write(f"Disagreements: {len(comparison['errors'])}\n\n")

            f.write("Disagreements:\n")
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
        "--pass_count",
        type=int,
        default=10,
        help="Number of predicted-pass examples to sample",
    )
    parser.add_argument(
        "--fail_count",
        type=int,
        default=10,
        help="Number of predicted-fail examples to sample",
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
        default="gopher_quality_sample.jsonl",
        help="Path to write/read the manual annotation template",
    )
    parser.add_argument(
        "--report_file",
        type=str,
        default="gopher_quality_report.txt",
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

    predictions = collect_quality_predictions(
        warc_paths=warc_paths,
        min_text_length=args.min_text_length,
    )

    sampled = sample_balanced_examples(
        predictions=predictions,
        pass_count=args.pass_count,
        fail_count=args.fail_count,
        seed=args.seed,
    )

    passed_available = sum(1 for x in predictions if x["passed_filter"])
    failed_available = sum(1 for x in predictions if not x["passed_filter"])

    if args.mode == "sample":
        write_annotation_template(sampled, annotation_path)

        print(f"Wrote balanced annotation template to: {annotation_path}")
        print(f"Total extracted documents: {len(predictions)}")
        print(f"Available pass examples: {passed_available}")
        print(f"Available fail examples: {failed_available}")
        print(f"Sampled pass examples: {sum(1 for x in sampled if x['passed_filter'])}")
        print(f"Sampled fail examples: {sum(1 for x in sampled if not x['passed_filter'])}")

    elif args.mode == "report":
        annotations = load_manual_annotations(annotation_path)
        comparison = compare_with_manual(sampled, annotations)
        write_report(predictions, comparison, report_path)

        print(f"Wrote report to: {report_path}")
        print(f"Total extracted documents: {len(predictions)}")
        print(f"Manual agreement: {comparison['accuracy']:.4f}")
        print(f"Disagreements: {len(comparison['errors'])}")


if __name__ == "__main__":
    main()