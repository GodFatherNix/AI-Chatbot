#!/usr/bin/env python3
"""Process crawled MOSDAC items, generate embeddings, and insert into Qdrant.

Usage:
    python processing/process_documents.py --input data/mosdac_crawl_20240501T120000Z.jl

Environment variables:
    QDRANT_HOST: Hostname of the Qdrant instance (default: localhost)
    QDRANT_PORT: Port number (default: 6333)
    QDRANT_COLLECTION: Collection name (default: mosdac_embeddings)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, List

import requests
from bs4 import BeautifulSoup
from pdfminer.high_level import extract_text as extract_pdf_text
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

DEFAULT_MAX_TOKENS = 500
OVERLAP_TOKENS = 50

# ----------------------------------------------------------------------------
# Utility functions
# ----------------------------------------------------------------------------


def clean_html(html: str) -> str:
    """Remove scripts/styles and collapse whitespace."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    text = soup.get_text(separator=" ")
    return re.sub(r"\s+", " ", text).strip()


def chunk_text(text: str, max_tokens: int = DEFAULT_MAX_TOKENS, overlap: int = OVERLAP_TOKENS) -> List[str]:
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_tokens)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # next window with overlap
        start += max_tokens - overlap
    return chunks


def download_file(url: str, dest_dir: Path) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    fname = dest_dir / Path(url).name
    try:
        with requests.get(url, timeout=30, stream=True) as r:
            r.raise_for_status()
            with open(fname, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        return fname
    except Exception as err:  # noqa: BLE001
        print(f"[WARN] Failed to download {url}: {err}")
        return None


def extract_text_from_file(filepath: Path) -> str:
    suffix = filepath.suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(str(filepath))
    elif suffix in {".doc", ".docx"}:
        try:
            import docx2txt  # type: ignore

            return docx2txt.process(str(filepath))
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("docx2txt is required for DOCX extraction") from e
    elif suffix in {".xls", ".xlsx"}:
        try:
            import pandas as pd  # type: ignore

            dfs = pd.read_excel(filepath, sheet_name=None, dtype=str)
            text_parts = []
            for sheet, df in dfs.items():
                text_parts.append(f"Sheet: {sheet}\n")
                text_parts.append(df.to_csv(index=False))
            return "\n".join(text_parts)
        except Exception as err:  # noqa: BLE001
            print(f"[WARN] Failed to parse Excel {filepath}: {err}")
            return ""
    else:
        return ""  # Unsupported type for now


# ----------------------------------------------------------------------------
# Main processing logic
# ----------------------------------------------------------------------------


def ensure_collection(client: QdrantClient, collection_name: str, dim: int) -> None:
    if collection_name in [c.name for c in client.get_collections().collections]:
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=qdrant_models.VectorParams(size=dim, distance=qdrant_models.Distance.COSINE),
    )
    print(f"[INFO] Created Qdrant collection '{collection_name}' with dim {dim}")


def upsert_batches(
    client: QdrantClient,
    collection_name: str,
    embeddings: List[List[float]],
    payloads: List[dict],
    batch_size: int = 64,
) -> None:
    # Split into batches
    for i in range(0, len(embeddings), batch_size):
        batch_vectors = embeddings[i : i + batch_size]
        batch_payloads = payloads[i : i + batch_size]
        points = [
            qdrant_models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload=payload,
            )
            for vec, payload in zip(batch_vectors, batch_payloads, strict=True)
        ]
        client.upsert(collection_name=collection_name, points=points)


# ----------------------------------------------------------------------------
# CLI entrypoint
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Process crawled MOSDAC data and embed to Qdrant")
    parser.add_argument("--input", required=True, help="Path to .jl file produced by crawler")
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--overlap", type=int, default=OVERLAP_TOKENS)
    args = parser.parse_args()

    jl_path = Path(args.input)
    if not jl_path.exists():
        raise FileNotFoundError(jl_path)

    # Init embedding model
    model = SentenceTransformer("all-MiniLM-L12-v2")
    dim = model.get_sentence_embedding_dimension()

    # Init Qdrant client
    client = QdrantClient(
        host=os.getenv("QDRANT_HOST", "localhost"),
        port=int(os.getenv("QDRANT_PORT", 6333)),
    )
    collection_name = os.getenv("QDRANT_COLLECTION", "mosdac_embeddings")
    ensure_collection(client, collection_name, dim)

    # Temporary directory for downloaded docs
    tmp_dir = Path(tempfile.mkdtemp(prefix="mosdac_docs_"))

    texts: List[str] = []
    metadata: List[dict] = []

    print("[INFO] Reading and processing items…")
    with jl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, unit="item"):
            item = json.loads(line)

            if "html" in item:  # PageItem
                text = clean_html(item.get("html", ""))
                source = item.get("url")
                if not text:
                    continue
                for idx, chunk in enumerate(chunk_text(text, args.max_tokens, args.overlap)):
                    texts.append(chunk)
                    metadata.append(
                        {
                            "source": source,
                            "chunk": idx,
                            "type": "html",
                            "title": item.get("title", ""),
                            "text": chunk,
                        }
                    )
            elif "file_url" in item:  # FileItem
                file_url = item.get("file_url")
                file_path = download_file(file_url, tmp_dir)
                if file_path is None:
                    continue
                extracted_text = extract_text_from_file(file_path)
                if not extracted_text:
                    continue
                for idx, chunk in enumerate(chunk_text(extracted_text, args.max_tokens, args.overlap)):
                    texts.append(chunk)
                    metadata.append(
                        {
                            "source": file_url,
                            "chunk": idx,
                            "type": file_path.suffix.lower().lstrip("."),
                            "text": chunk,
                        }
                    )
            else:
                # Unknown item type
                continue

    if not texts:
        print("[WARN] No text chunks produced. Exiting.")
        return

    print(f"[INFO] Generating embeddings for {len(texts)} chunks…")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True).tolist()

    print("[INFO] Upserting into Qdrant…")
    upsert_batches(client, collection_name, embeddings, metadata)
    print("[DONE] Processing complete.")


if __name__ == "__main__":
    main()