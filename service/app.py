#!/usr/bin/env python3
"""Chatbot API service for MOSDAC.

Run with:
    uvicorn service.app:app --reload --port 8000

Environment variables:
    OPENAI_API_KEY           – Required if using OpenAI model
    QDRANT_HOST, QDRANT_PORT, QDRANT_COLLECTION – vector store
    NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD        – KG store
"""
from __future__ import annotations

import os
import logging
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, Filter, FieldCondition, MatchAny, Distance

from py2neo import Graph
import spacy
import openai

# Import geospatial service
import sys
sys.path.append('..')
from geospatial.geospatial_service import GeospatialService

# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mosdac-chatbot")

load_dotenv()

# Embedding model (same as used during ingest)
EMBED_MODEL_NAME = "all-MiniLM-L12-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

# Vector store (Qdrant)
qdrant = QdrantClient(
    host=os.getenv("QDRANT_HOST", "localhost"),
    port=int(os.getenv("QDRANT_PORT", 6333)),
)
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "mosdac_embeddings")

# Ensure collection exists (could raise later)
_coll_names = [c.name for c in qdrant.get_collections().collections]
if COLLECTION_NAME not in _coll_names:
    raise RuntimeError(f"Qdrant collection '{COLLECTION_NAME}' not found. Run processing pipeline first.")

# Knowledge graph (Neo4j)
try:
    graph = Graph(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(os.getenv("NEO4J_USER", "neo4j"), os.getenv("NEO4J_PASSWORD", "password")),
    )
except Exception as err:  # noqa: BLE001
    logger.warning("Failed to connect Neo4j: %s", err)
    graph = None  # allow service to still run

# spaCy for NER
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # Model not downloaded; attempt auto download
    import subprocess, sys

    logger.info("Downloading spaCy model…")
    subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
    nlp = spacy.load("en_core_web_sm")

# LLM (OpenAI)
openai.api_key = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo-0125")

# Chat memory (very simple in-memory store; for production use Redis)
session_memory: Dict[str, List[dict]] = {}
MEMORY_MAX_TURNS = 5

# Initialize geospatial service
geospatial_service = GeospatialService()

# ---------------------------------------------------------------------------
# FastAPI models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    session_id: str
    message: str
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    context: List[str]
    citations: List[dict]


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def retrieve_vectors(query: str, top_k: int = 5) -> List[dict]:
    """Return top_k vector results with payload and text."""
    query_vec = embedder.encode(query, normalize_embeddings=True).tolist()
    search_result = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vec,
        limit=top_k,
        with_payload=True,
    )
    docs = []
    for point in search_result:
        payload = point.payload or {}
        text = payload.get("text") or payload.get("body")  # embed pipeline could store
        docs.append({
            "id": point.id,
            "text": text,
            "score": point.score,
            "payload": payload,
        })
    return docs


def query_kg(query: str, limit: int = 5) -> List[str]:
    if graph is None:
        return []
    doc = nlp(query)
    ent_names = {ent.text.strip() for ent in doc.ents if ent.label_ in {"ORG", "GPE", "PERSON", "PRODUCT"}}
    if not ent_names:
        return []
    contexts = []
    for name in list(ent_names)[:3]:
        cypher = (
            "MATCH (e:Entity {name: $name})-[r]-(n) "
            "RETURN e.name AS source, type(r) AS rel, n.name AS target LIMIT $lim"
        )
        try:
            rows = graph.run(cypher, name=name, lim=limit).data()
            for row in rows:
                contexts.append(f"{row['source']} --{row['rel']}--> {row['target']}")
        except Exception as err:  # noqa: BLE001
            logger.warning("KG query failed: %s", err)
            continue
    return contexts


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------

def generate_answer(question: str, context: List[str], chat_history: List[dict]) -> str:
    if not openai.api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set.")

    system_prompt = (
        "You are MOSDAC AI assistant. Answer the user question using the given context. "
        "Cite sources by providing inline numbers like [1], [2]. If answer is unknown, say so politely."
    )
    context_block = "\n".join([f"[{i+1}] {c}" for i, c in enumerate(context)])

    messages = [{"role": "system", "content": system_prompt}]
    # Attach memory
    for m in chat_history[-MEMORY_MAX_TURNS:]:
        messages.append(m)
    # Append context and question
    messages.append({"role": "user", "content": f"Context:\n{context_block}\n\nQuestion: {question}"})

    response = openai.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# FastAPI endpoints
# ---------------------------------------------------------------------------

app = FastAPI(title="MOSDAC Chatbot RAG API")


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # Retrieve context from vector store and KG
    vect_docs = retrieve_vectors(req.message, req.top_k)
    kg_context = query_kg(req.message, limit=3)

    context_texts = []
    citations = []
    # Build combined context, store citations
    for i, doc in enumerate(vect_docs):
        if doc["text"]:
            context_texts.append(doc["text"])
            citations.append({"id": doc["id"], "source": doc["payload"].get("source")})
    context_texts.extend(kg_context)

    # Chat memory per session
    history = session_memory.setdefault(req.session_id, [])

    answer = generate_answer(req.message, context_texts, history)

    # Update memory
    history.append({"role": "user", "content": req.message})
    history.append({"role": "assistant", "content": answer})

    return ChatResponse(answer=answer, context=context_texts, citations=citations)

@app.get("/geospatial/coverage")
async def get_mission_coverage():
    """Get mission coverage as GeoJSON for map visualization."""
    return geospatial_service.get_mission_coverage_geojson()

@app.post("/geospatial/query_bbox")
async def query_by_bbox(bbox: List[float], mission: str = None, product_type: str = None):
    """Query satellite products by bounding box."""
    results = geospatial_service.query_by_bbox(bbox, mission=mission, product_type=product_type)
    return {"results": [result.__dict__ for result in results]}

@app.post("/geospatial/query_location")
async def query_by_location(lat: float, lon: float, mission: str = None):
    """Query products covering a specific location."""
    results = geospatial_service.query_by_location(lat, lon, mission=mission)
    return {"results": [result.__dict__ for result in results]}