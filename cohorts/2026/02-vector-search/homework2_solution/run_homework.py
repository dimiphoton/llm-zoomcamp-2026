"""Exécute toutes les questions du homework 2 et affiche les réponses."""

import subprocess
from pathlib import Path

import numpy as np
from embedder import Embedder
from gitsource import chunk_documents
from minsearch import Index, VectorSearch

COMMIT_ID = "8c1834d"
REPO_ROOT = Path(__file__).resolve().parents[4]  # racine llm-zoomcamp-2026


def load_documents(commit_id: str = COMMIT_ID) -> list[dict]:
    """Charge les leçons depuis le dépôt local (commit figé, sans appel réseau)."""
    result = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", commit_id],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    paths = [
        p
        for p in result.stdout.splitlines()
        if "/lessons/" in p and p.endswith(".md")
    ]
    documents = []
    for path in paths:
        raw = subprocess.run(
            ["git", "show", f"{commit_id}:{path}"],
            capture_output=True,
            check=True,
            cwd=REPO_ROOT,
        ).stdout
        documents.append({"filename": path, "content": raw.decode("utf-8")})
    return documents


def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


def main() -> None:
    query_q1 = "How does approximate nearest neighbor search work?"
    embedder = Embedder()

    # Q1
    v = embedder.encode(query_q1)
    print(f"Q1 v[0] = {v[0]:.4f}")

    # Chargement des documents (commit figé 8c1834d)
    documents = load_documents()
    print(f"Documents: {len(documents)}")

    # Q2
    target = "02-vector-search/lessons/07-sqlitesearch-vector.md"
    page = next(d for d in documents if d["filename"] == target)
    page_vec = embedder.encode(page["content"])
    sim = float(page_vec.dot(v))
    print(f"Q2 cosine similarity = {sim:.4f}")

    # Q3
    chunks = chunk_documents(documents, size=2000, step=1000)
    texts = [c["content"] for c in chunks]
    X = embedder.encode_batch(texts)
    scores = X.dot(v)
    best_idx = int(np.argmax(scores))
    print(f"Q3 best chunk filename = {chunks[best_idx]['filename']}")
    print(f"Q3 best score = {scores[best_idx]:.4f}")

    # Q4
    vindex = VectorSearch(keyword_fields=["filename"])
    vindex.fit(X, chunks)
    q4_query = "What metric do we use to evaluate a search engine?"
    q4_vec = embedder.encode(q4_query)
    q4_results = vindex.search(q4_vec, num_results=5)
    print(f"Q4 first result = {q4_results[0]['filename']}")

    # Q5
    text_index = Index(text_fields=["content"], keyword_fields=["filename"])
    text_index.fit(chunks)
    q5_query = "How do I store vectors in PostgreSQL?"
    q5_vec = embedder.encode(q5_query)
    vector_results = vindex.search(q5_vec, num_results=5)
    text_results = text_index.search(q5_query, num_results=5)

    vector_files = {r["filename"] for r in vector_results}
    text_files = {r["filename"] for r in text_results}
    only_vector = vector_files - text_files
    print(f"Q5 vector only files = {only_vector}")
    print("Q5 vector top 5:", [r["filename"] for r in vector_results])
    print("Q5 text top 5:", [r["filename"] for r in text_results])

    # Q6
    q6_query = "How do I give the model access to tools?"
    q6_vec = embedder.encode(q6_query)
    q6_vector = vindex.search(q6_vec, num_results=5)
    q6_text = text_index.search(q6_query, num_results=5)
    q6_hybrid = rrf([q6_vector, q6_text])
    print(f"Q6 RRF first = {q6_hybrid[0]['filename']}")
    print("Q6 vector top 5:", [r["filename"] for r in q6_vector])
    print("Q6 text top 5:", [r["filename"] for r in q6_text])
    print("Q6 hybrid top 5:", [r["filename"] for r in q6_hybrid])


if __name__ == "__main__":
    main()
