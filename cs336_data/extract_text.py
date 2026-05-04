from __future__ import annotations

import argparse
import gzip
from pathlib import Path
"""
python extract_text.py --warc CC-MAIN-20250417135010-20250417165010-00065.warc.gz --wet CC-MAIN-20250417135010-20250417165010-00065.warc.wet.gz
"""
from fastwarc.warc import ArchiveIterator, WarcRecordType

from filtering_helper import extract_text_from_html_bytes


def read_warc_samples(warc_path: Path, limit: int) -> list[str]:
    samples: list[str] = []

    with gzip.open(warc_path, "rb") as f:
        for record in ArchiveIterator(f, parse_http=False):
            if record.record_type != WarcRecordType.response:
                continue
            try:
                payload = record.reader.read()
                text = extract_text_from_html_bytes(payload)
            except Exception:
                continue
            if text:
                text = " ".join(text.split())
                samples.append(text)
                if len(samples) >= limit:
                    break
    return samples


def read_wet_samples(wet_path: Path, limit: int) -> list[str]:
    samples: list[str] = []
    with gzip.open(wet_path, "rb") as f:
        for record in ArchiveIterator(f, parse_http=False):
            if record.record_type != WarcRecordType.conversion:
                continue
            try:
                payload = record.reader.read()
                text = payload.decode("utf-8", errors="replace")
                text = " ".join(text.split())
            except Exception:
                continue
            if text:
                samples.append(text)
                if len(samples) >= limit:
                    break
    return samples


def write_detailed_report(
    warc_samples: list[str],
    wet_samples: list[str],
    output_path: Path,
) -> None:
    """
    Write side-by-side sample outputs for manual inspection.
    """
    n = min(len(warc_samples), len(wet_samples))
    with output_path.open("w", encoding="utf-8") as f:
        for i in range(n):
            f.write("=" * 100 + "\n")
            f.write(f"SAMPLE {i + 1}\n\n")
            f.write("[MY WARC EXTRACTION]\n")
            f.write(warc_samples[i][:2000] + "\n\n")
            f.write("[WET EXTRACTION]\n")
            f.write(wet_samples[i][:2000] + "\n\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warc", required=True, help="Path to input .warc.gz")
    parser.add_argument("--wet", required=True, help="Path to corresponding .warc.wet.gz")
    parser.add_argument("--limit", type=int, default=5, help="Number of non-empty samples to compare")
    parser.add_argument(
        "--output",
        default="warc_wet_comparison.txt",
        help="Path to save detailed side-by-side comparison",
    )
    args = parser.parse_args()

    warc_path = Path(args.warc)
    wet_path = Path(args.wet)
    output_path = Path(args.output)

    warc_samples = read_warc_samples(warc_path, args.limit)
    wet_samples = read_wet_samples(wet_path, args.limit)

    write_detailed_report(warc_samples, wet_samples, output_path)

    print(f"\nDetailed comparison written to: {output_path}")


if __name__ == "__main__":
    main()