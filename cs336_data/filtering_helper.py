from __future__ import annotations

import os
import re
import unicodedata
from typing import Any

from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.extract.html2text import extract_plain_text
from resiliparse.parse.encoding import detect_encoding
import fasttext
import hashlib
from pathlib import Path
from collections import Counter, defaultdict

HTTP_SPLIT_RE = re.compile(br"\r?\n\r?\n", re.MULTILINE)


def extract_text_from_html_bytes(html_bytes: bytes) -> str | None:
    try:
        if not html_bytes:
            return None
        payload = html_bytes
        # --- 关键修复：去掉 HTTP header ---
        if payload.startswith(b"HTTP/"):
            parts = HTTP_SPLIT_RE.split(payload, maxsplit=1)
            if len(parts) == 2:
                payload = parts[1]
        # --- 正常 extraction ---
        encoding = detect_encoding(payload)
        html_str = payload.decode(encoding, errors="replace")
        text = extract_plain_text(html_str)
        return text if text else None
    except Exception:
        return None

_LID_MODEL = None
_NSFW_MODEL = None
_TOXIC_MODEL = None


def _get_lid_model(model_path: str = '~/fanchi/proj/opensource/model/lid.176.bin'):
    global _LID_MODEL
    if _LID_MODEL is None:
        _LID_MODEL = fasttext.load_model(os.path.expanduser(model_path))
    return _LID_MODEL


def _get_nsfw_model(model_path: str = '~/fanchi/proj/opensource/model/dolma_fasttext_nsfw_jigsaw_model.bin'):
    global _NSFW_MODEL
    if _NSFW_MODEL is None:
        _NSFW_MODEL = fasttext.load_model(os.path.expanduser(model_path))
    return _NSFW_MODEL


def _get_toxic_model(model_path: str = '~/fanchi/proj/opensource/model/dolma_fasttext_hatespeech_jigsaw_mode.bin'):
    global _TOXIC_MODEL
    if _TOXIC_MODEL is None:
        _TOXIC_MODEL = fasttext.load_model(os.path.expanduser(model_path))
    return _TOXIC_MODEL


def identify_language(text: str) -> tuple[Any, float]:
    text = " ".join(text.split())

    if not text:
        return "unknown", 0.0

    labels, probs = _get_lid_model().predict(text, k=1)

    lang = labels[0]
    score = float(probs[0])

    if lang.startswith("__label__"):
        lang = lang[len("__label__"):]

    mapping = {
        "eng": "en",
        "english": "en",
        "cmn": "zh",
        "zho": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "zh-hans": "zh",
        "zh-hant": "zh",
    }
    lang = mapping.get(lang.lower(), lang.lower())

    return lang, score

EMAIL_RE = re.compile(
    r'(?<![\w.+-])'
    r'[A-Za-z0-9.!#$%&\'*+/=?^_`{|}~-]+'
    r'@'
    r'[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+'
    r'(?![\w.-])'
)
def mask_emails(text: str) -> tuple[str, int]:
    masked_text, count = EMAIL_RE.subn("|||EMAIL_ADDRESS|||", text)
    return masked_text, count

PHONE_RE = re.compile(
    r"""
    (?<!\w)                           # avoid matching inside words
    (?:\+?1[\s.\-]*)?                 # optional country code
    (?:\(?\d{3}\)?[\s.\-]*)           # area code, with optional parentheses
    \d{3}[\s.\-]*                     # central office code
    \d{4}                             # line number
    (?!\w)                            # avoid matching inside words
    """,
    re.VERBOSE,
)

def mask_phone_numbers(text: str) -> tuple[str, int]:
    masked_text, count = PHONE_RE.subn("|||PHONE_NUMBER|||", text)
    return masked_text, count

IPV4_RE = re.compile(
    r"""
    (?<!\d)
    (?:
        (?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)
        \.
    ){3}
    (?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)
    (?!\d)
    """,
    re.VERBOSE,
)

def mask_ips(text: str) -> tuple[str, int]:
    masked_text, count = IPV4_RE.subn("|||IP_ADDRESS|||", text)
    return masked_text, count


def _predict_fasttext_label(model, text: str):
    labels, probs = model.predict(text.replace("\n", " "), k=1)
    label = labels[0]
    if label.startswith("__label__"):
        label = label[len("__label__"):]
    return label.lower(), float(probs[0])

def classify_nsfw(text: str) -> tuple[Any, float]:
    label, score = _predict_fasttext_label(_get_nsfw_model(), text)

    mapping = {
        "nsfw": "nsfw",
        "not_nsfw": "not_nsfw",
        "non_nsfw": "not_nsfw",
        "safe": "not_nsfw",
        "0": "not_nsfw",
        "1": "nsfw",
    }
    return mapping.get(label, label), score


def classify_toxic_speech(text: str) -> tuple[Any, float]:
    label, score = _predict_fasttext_label(_get_toxic_model(), text)

    mapping = {
        "toxic": "toxic",
        "non_toxic": "non_toxic",
        "not_toxic": "non_toxic",
        "normal": "non_toxic",
        "0": "non_toxic",
        "1": "toxic",
        "hatespeech": "toxic",
        "hate": "toxic",
    }
    return mapping.get(label, label), score

WORD_RE = re.compile(r"\b\S+\b")
ALPHA_RE = re.compile(r"[A-Za-z]")
_QUALITY_MODEL = None


def _load_quality_model_if_available():
    global _QUALITY_MODEL
    if _QUALITY_MODEL is not None:
        return _QUALITY_MODEL
    if os.path.exists(os.path.expanduser("~/fanchi/proj/opensource/model/quality_classifier/quality_classifier.bin")):
        _QUALITY_MODEL = fasttext.load_model(os.path.expanduser("~/fanchi/proj/opensource/model/quality_classifier/quality_classifier.bin"))
    return _QUALITY_MODEL


def _train_quality_model_if_possible(
    train_file: str = "quality_train.txt", model_file: str = "quality_classifier.bin"
):
    if not os.path.exists(train_file):
        return None
    model = fasttext.train_supervised(
        input=train_file,
        lr=0.5,
        epoch=25,
        wordNgrams=2,
        minn=2,
        maxn=5,
        dim=100,
        loss="softmax",
        thread=8,
    )
    model.save_model(model_file)
    return model


def classify_quality(text: str) -> tuple[Any, float]:
    normalized = " ".join(text.split())
    if not normalized:
        return "cc", 1.0

    model = _load_quality_model_if_available()
    if model is None:
        model = _train_quality_model_if_possible()
    if model is None:
        raise FileNotFoundError(
            "quality classifier not found. "
            "Please train one first. Example: "
            "python -m pipeline.quality_classifier train_from_warcs "
            "--positive_warcs subsampled_positive_urls.warc.gz "
            "--negative_warcs CC-MAIN-20250417135010-20250417165010-00065.warc.gz"
        )

    labels, probs = model.predict(normalized, k=1)
    label = labels[0].replace("__label__", "").lower()
    score = float(probs[0])
    if label in {"wiki", "hq", "high_quality", "1"}:
        return "wiki", score
    if label in {"cc", "lq", "low_quality", "0"}:
        return "cc", score
    return label, score

def gopher_quality_filter(text: str) -> bool:
    words = WORD_RE.findall(text)
    num_words = len(words)

    # Rule 1: document length
    if num_words < 50 or num_words > 100_000:
        return False

    # Rule 2: mean word length
    mean_word_length = sum(len(w) for w in words) / num_words
    if mean_word_length < 3 or mean_word_length > 10:
        return False

    # Rule 3: >30% of lines end with ellipsis
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        ellipsis_fraction = sum(line.endswith("...") for line in lines) / len(lines)
        if ellipsis_fraction > 0.30:
            return False

    # Rule 4: <80% of words contain at least one alphabetic character
    alpha_word_fraction = sum(bool(ALPHA_RE.search(w)) for w in words) / num_words
    if alpha_word_fraction < 0.80:
        return False

    return True

def _hash_line(line: str) -> bytes:
    return hashlib.sha1(line.encode("utf-8")).digest()


def exact_line_deduplication(
    input_files: list[os.PathLike], output_directory: os.PathLike
):
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    line_counts = Counter()

    # First pass: count line frequencies across all input files
    for input_file in input_files:
        input_path = Path(input_file)
        with input_path.open("r", encoding="utf-8") as f:
            for line in f:
                line_counts[_hash_line(line)] += 1

    # Second pass: write only globally unique lines
    for input_file in input_files:
        input_path = Path(input_file)
        output_path = output_dir / input_path.name

        with input_path.open("r", encoding="utf-8") as fin, output_path.open(
            "w", encoding="utf-8"
        ) as fout:
            for line in fin:
                if line_counts[_hash_line(line)] == 1:
                    fout.write(line)

def _normalize_text(text: str) -> str:
    # NFD normalize first
    text = unicodedata.normalize("NFD", text)

    # remove accents / combining marks
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    # lowercase
    text = text.lower()

    # remove punctuation / symbols by replacing with spaces
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or cat.startswith("S"):
            chars.append(" ")
        else:
            chars.append(ch)
    text = "".join(chars)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _word_ngrams(text: str, n: int) -> set[str]:
    words = text.split()
    if not words:
        return set()

    if len(words) < n:
        # Treat the whole short document as one shingle so empty documents
        # are not created just because they are shorter than n.
        return {" ".join(words)}

    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def _stable_hash_with_seed(s: str, seed: int) -> int:
    h = hashlib.blake2b(digest_size=8, person=seed.to_bytes(8, "little", signed=False))
    h.update(s.encode("utf-8"))
    return int.from_bytes(h.digest(), "big", signed=False)


def _minhash_signature(ngrams_set: set[str], num_hashes: int) -> tuple[int, ...]:
    if not ngrams_set:
        # empty signature sentinel
        return tuple([2**64 - 1] * num_hashes)

    sig = []
    for seed in range(num_hashes):
        min_val = min(_stable_hash_with_seed(ng, seed) for ng in ngrams_set)
        sig.append(min_val)
    return tuple(sig)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def minhash_deduplication(
    input_files: list[os.PathLike],
    num_hashes: int,
    num_bands: int,
    ngrams: int,
    jaccard_threshold: float,
    output_directory: os.PathLike,
):
    input_paths = [Path(p) for p in input_files]
    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    if num_hashes % num_bands != 0:
        raise ValueError("num_hashes must be evenly divisible by num_bands")

    rows_per_band = num_hashes // num_bands

    # Read documents and precompute normalized n-gram sets + minhash signatures
    docs = []
    for path in input_paths:
        text = path.read_text(encoding="utf-8")
        normalized = _normalize_text(text)
        ng_set = _word_ngrams(normalized, ngrams)
        signature = _minhash_signature(ng_set, num_hashes)
        docs.append(
            {
                "path": path,
                "text": text,
                "normalized": normalized,
                "ngrams": ng_set,
                "signature": signature,
            }
        )

    # LSH: bucket documents by band
    candidate_pairs = set()
    for band_idx in range(num_bands):
        start = band_idx * rows_per_band
        end = start + rows_per_band
        buckets = defaultdict(list)

        for i, doc in enumerate(docs):
            band = doc["signature"][start:end]
            bucket_key = (band_idx, band)
            buckets[bucket_key].append(i)

        for bucket_docs in buckets.values():
            if len(bucket_docs) > 1:
                for i in range(len(bucket_docs)):
                    for j in range(i + 1, len(bucket_docs)):
                        a, b = bucket_docs[i], bucket_docs[j]
                        if a > b:
                            a, b = b, a
                        candidate_pairs.add((a, b))

    # Verify candidates with true Jaccard similarity and cluster duplicates
    uf = _UnionFind(len(docs))
    for a, b in candidate_pairs:
        sim = _jaccard(docs[a]["ngrams"], docs[b]["ngrams"])
        if sim >= jaccard_threshold:
            uf.union(a, b)

    # Build clusters
    clusters = defaultdict(list)
    for i in range(len(docs)):
        clusters[uf.find(i)].append(i)

    # Retain one document per cluster.
    # Deterministic choice is better for tests than random choice.
    keep = set()
    for members in clusters.values():
        chosen = min(members, key=lambda idx: str(docs[idx]["path"].name))
        keep.add(chosen)

    # Write only retained documents to the output directory with the same filename
    for idx in keep:
        out_path = output_dir / docs[idx]["path"].name
        out_path.write_text(docs[idx]["text"], encoding="utf-8")