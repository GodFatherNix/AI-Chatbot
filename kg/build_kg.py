#!/usr/bin/env python3
"""Build a simple knowledge graph from MOSDAC crawled data.

This script reads the JSON-Lines file produced by the crawler, extracts entities and simple
co-occurrence relationships using spaCy NER, and populates a Neo4j graph.

Usage:
    python kg/build_kg.py --input data/mosdac_crawl_YYYYMMDDTXXXXXXZ.jl

Environment variables:
    NEO4J_URI (bolt://localhost:7687)
    NEO4J_USER (neo4j)
    NEO4J_PASSWORD (password)

Note: This is a *starter* KG builder. For production, replace the naive co-occurrence logic
with domain-specific relation extraction.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import spacy
from py2neo import Graph, Node, Relationship
from tqdm import tqdm
from bs4 import BeautifulSoup

# ----------------------------------------------------------------------------
# Config & utilities
# ----------------------------------------------------------------------------


ENTITY_LABELS = {
    "ORG",
    "GPE",
    "PRODUCT",  # spacy might not have PRODUCT; kept for future custom model
    "FAC",
    "PERSON",  # may capture mission scientists etc.
}

DEFAULT_RELATION = "RELATED_TO"

nlp = spacy.load("en_core_web_sm", disable=["lemmatizer", "parser"])

# Precompiled regex for known entity hints (domain-specific)
SATELLITE_REGEX = re.compile(r"INSAT|SCATSAT|MEGHA\-TROPIQUES|OCEANSAT", re.IGNORECASE)
PRODUCT_REGEX = re.compile(r"[A-Z]{2,}_?\d{0,2}\w*", re.IGNORECASE)


# ----------------------------------------------------------------------------
# Helper functions
# ----------------------------------------------------------------------------


def clean_html(text: str) -> str:
    soup = BeautifulSoup(text, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav"]):
        tag.decompose()
    return soup.get_text(separator=" ")


def extract_entities(text: str) -> List[Tuple[str, str]]:
    """Return list of (entity_text, label) pairs, using spaCy and regex hints."""
    ents: List[Tuple[str, str]] = []
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ENTITY_LABELS:
            ents.append((ent.text.strip(), ent.label_))
    # Regex-based heuristics
    for match in SATELLITE_REGEX.finditer(text):
        ents.append((match.group(), "SATELLITE"))
    for match in PRODUCT_REGEX.finditer(text):
        ents.append((match.group(), "PRODUCT"))
    return ents


def upsert_node(graph: Graph, cache: Dict[str, Node], name: str, label: str) -> Node:
    key = f"{label}|{name.lower()}"
    if key in cache:
        return cache[key]
    node = Node(label, name=name)
    graph.merge(node, label, "name")
    cache[key] = node
    return node


# ----------------------------------------------------------------------------
# Main loading logic
# ----------------------------------------------------------------------------


def build_graph(input_path: Path, graph: Graph) -> None:
    node_cache: Dict[str, Node] = {}
    pair_counter: Dict[Tuple[str, str], int] = defaultdict(int)

    with input_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing items"):
            item = json.loads(line)
            if "html" in item:
                raw_text = clean_html(item["html"])
                sent_texts = raw_text.split(". ")  # naive sentence split
                for sent in sent_texts:
                    if not sent.strip():
                        continue
                    ents = extract_entities(sent)
                    # Map to unique (name,label)
                    uniq = {(n, l) for n, l in ents}
                    uniq_list = list(uniq)
                    # Count pair co-occurrences
                    for i in range(len(uniq_list)):
                        for j in range(i + 1, len(uniq_list)):
                            k = tuple(sorted((uniq_list[i][0], uniq_list[j][0])))
                            pair_counter[k] += 1
            # FileItem text not processed here (could be extended)

    # Upsert nodes and relationships
    for (name_a, name_b), weight in tqdm(pair_counter.items(), desc="Writing graph"):
        node_a = upsert_node(graph, node_cache, name_a, "Entity")
        node_b = upsert_node(graph, node_cache, name_b, "Entity")
        rel = Relationship(node_a, DEFAULT_RELATION, node_b, weight=weight)
        graph.merge(rel)


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Build knowledge graph from MOSDAC crawl")
    parser.add_argument("--input", required=True, help="Path to .jl crawl file")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    # Connect to Neo4j
    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "password")
    graph = Graph(uri, auth=(user, password))

    build_graph(input_path, graph)
    print("[DONE] Knowledge graph build complete.")


if __name__ == "__main__":
    main()