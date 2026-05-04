from __future__ import annotations

import argparse
import gzip
import json
import random
from pathlib import Path
from typing import Any

from fastwarc.warc import ArchiveIterator, WarcRecordType

from filtering_helper import (
    extract_text_from_html_bytes,
    mask_emails,
    mask_phone_numbers,
    mask_ips,
)


def _open_warc(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def collect_masked_examples(
    warc_paths: list[Path],
    min_text_length: int = 1,
) -> list[dict[str, Any]]:
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

                masked_text = text

                masked_text, email_count = mask_emails(masked_text)
                masked_text, phone_count = mask_phone_numbers(masked_text)
                masked_text, ip_count = mask_ips(masked_text)

                total_replacements = email_count + phone_count + ip_count
                if total_replacements == 0:
                    continue

                try:
                    record_id = record.record_id
                except Exception:
                    record_id = None

                results.append(
                    {
                        "warc_file": str(warc_path),
                        "record_id": record_id,
                        "email_count": email_count,
                        "phone_count": phone_count,
                        "ip_count": ip_count,
                        "total_replacements": total_replacements,
                        "original_text_preview": text[:1000],
                        "masked_text_preview": masked_text[:1000],
                    }
                )

    return results


def sample_examples(
    examples: list[dict[str, Any]],
    sample_size: int = 20,
    seed: int = 42,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    if len(examples) <= sample_size:
        return list(examples)
    return rng.sample(examples, sample_size)


def write_samples(samples: list[dict[str, Any]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for i, sample in enumerate(samples, start=1):
            row = {
                "example_id": i,
                "warc_file": sample["warc_file"],
                "record_id": sample["record_id"],
                "email_count": sample["email_count"],
                "phone_count": sample["phone_count"],
                "ip_count": sample["ip_count"],
                "total_replacements": sample["total_replacements"],
                "false_positive_notes": "",
                "false_negative_notes": "",
                "original_text_preview": sample["original_text_preview"],
                "masked_text_preview": sample["masked_text_preview"],
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


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
        help="Number of random masked examples to inspect",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--min_text_length",
        type=int,
        default=1,
        help="Skip texts shorter than this",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="pii_masking_sample.jsonl",
        help="Path to save sampled examples",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    warc_paths = [Path(p) for p in args.warc_paths]
    output_path = Path(args.output)

    examples = collect_masked_examples(
        warc_paths=warc_paths,
        min_text_length=args.min_text_length,
    )
    samples = sample_examples(
        examples=examples,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    write_samples(samples, output_path)

    print(f"Total examples with at least one replacement: {len(examples)}")
    print(f"Wrote {len(samples)} sampled examples to: {output_path}")


if __name__ == "__main__":
    main()