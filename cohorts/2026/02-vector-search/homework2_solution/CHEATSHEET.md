# Cheatsheet — Homework 2 : Vector Search

Mémo pour refaire le devoir du **Module 2**. Le notebook complet est dans
[`homework2.ipynb`](homework2.ipynb). Script rapide : [`run_homework.py`](run_homework.py).

## But

Mettre en pratique la **recherche vectorielle** sur les pages de leçons du cours :

1. Embeddings légers avec **ONNX** (`Embedder`, pas PyTorch).
2. Similarité cosinus à la main avec **numpy**.
3. **Chunking** + recherche vectorielle avec **minsearch**.
4. Comparer **recherche texte** vs **vectorielle**.
5. Combiner les deux avec **hybrid search (RRF)**.

Pas de LLM, pas de RAG — uniquement la partie recherche.

## Environnement

### Prérequis

- Python 3.12+
- **uv** (gestionnaire de paquets du cours)
- Connexion réseau pour télécharger le modèle ONNX et les leçons GitHub

### Installation

```powershell
# Depuis ce dossier
cd cohorts/2026/02-vector-search/homework2_solution
uv sync
```

| Package | Rôle |
|---------|------|
| `onnxruntime` | Inférence du modèle ONNX |
| `tokenizers` | Tokenisation pour l'embedder |
| `numpy` | Matrices de vecteurs, dot product |
| `minsearch` | `VectorSearch` + `Index` (texte) |
| `gitsource` | Télécharge les leçons + `chunk_documents` |
| `huggingface-hub` | Télécharge le modèle (dev) |

### Scripts embedder (depuis le repo du cours)

```powershell
Copy-Item ..\..\..\..\02-vector-search\embed\download.py .
Copy-Item ..\..\..\..\02-vector-search\embed\embedder.py .
uv run python download.py   # télécharge Xenova/all-MiniLM-L6-v2 (~90 Mo)
```

Le modèle est stocké dans `models/Xenova/all-MiniLM-L6-v2/` (ignoré par git).

### Lancer le notebook

```powershell
uv run jupyter notebook homework2.ipynb
```

Rejouer sans interface :

```powershell
uv run python run_homework.py
```

## Commandes Python clés

### Initialiser l'embedder

```python
from embedder import Embedder

embedder = Embedder()  # modèle all-MiniLM-L6-v2, vecteurs 384 dims, normalisés
v = embedder.encode("How does approximate nearest neighbor search work?")
v[0]  # Q1 : premier élément du vecteur
```

### Charger les leçons (commit figé `8c1834d`)

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [file.parse() for file in reader.read()]  # 72 pages
```

### Q2 — similarité cosinus (dot product)

Vecteurs normalisés → dot product = cosine similarity :

```python
page = next(d for d in documents if d["filename"] == "02-vector-search/lessons/07-sqlitesearch-vector.md")
page_vec = embedder.encode(page["content"])
sim = page_vec.dot(v)
```

### Q3 — chunking + recherche à la main

```python
from gitsource import chunk_documents
import numpy as np

chunks = chunk_documents(documents, size=2000, step=1000)
X = embedder.encode_batch([c["content"] for c in chunks])
scores = X.dot(v)
best = chunks[int(np.argmax(scores))]
best["filename"]
```

### Q4 — VectorSearch (minsearch)

```python
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["filename"])
vindex.fit(X, chunks)

q4_vec = embedder.encode("What metric do we use to evaluate a search engine?")
vindex.search(q4_vec, num_results=5)[0]["filename"]
```

### Q5 — texte vs vectoriel

```python
from minsearch import Index

text_index = Index(text_fields=["content"], keyword_fields=["filename"])
text_index.fit(chunks)

query = "How do I store vectors in PostgreSQL?"
q_vec = embedder.encode(query)

vector_top = vindex.search(q_vec, num_results=5)
text_top = text_index.search(query, num_results=5)

vector_files = {r["filename"] for r in vector_top}
text_files = {r["filename"] for r in text_top}
only_in_vector = vector_files - text_files
```

### Q6 — hybrid search (RRF)

```python
def rrf(result_lists, k=60, num_results=5):
    scores, docs = {}, {}
    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc
    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]

query = "How do I give the model access to tools?"
q_vec = embedder.encode(query)
hybrid = rrf([
    vindex.search(q_vec, num_results=5),
    text_index.search(query, num_results=5),
])
hybrid[0]["filename"]
```

Formule RRF : `score(d) = Σ 1 / (k + rank)` avec `k = 60`.

## Réponses obtenues

| Q | Réponse | Détail |
|---|---------|--------|
| Q1 | **-0.02** | `v[0] ≈ -0.0206` |
| Q2 | **0.37** | cosine similarity ≈ 0.361 |
| Q3 | **`02-vector-search/lessons/07-sqlitesearch-vector.md`** | meilleur chunk |
| Q4 | **`04-evaluation/lessons/05-search-metrics.md`** | 1er résultat vectoriel |
| Q5 | **`02-vector-search/lessons/08-pgvector.md`** | dans vector, pas dans texte |
| Q6 | **`01-agentic-rag/lessons/13-function-calling.md`** | 1er après RRF |

Soumission : <https://courses.datatalks.club/llm-zoomcamp-2026/homework/hw2>

## Dépannage TLS (Windows + Conda)

Si `download.py` ou `gitsource` échoue avec `CERTIFICATE_VERIFY_FAILED` :

1. Vérifier `%APPDATA%\uv\uv.toml` → `system-certs = true`
2. Retirer `SSL_CERT_FILE` après `conda activate` (Conda pointe vers un bundle incomplet)
3. Télécharger le modèle manuellement si besoin :

```powershell
$dest = "models/Xenova/all-MiniLM-L6-v2"
New-Item -ItemType Directory -Force -Path $dest
$base = "https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main"
Invoke-WebRequest "$base/tokenizer.json" -OutFile "$dest/tokenizer.json"
Invoke-WebRequest "$base/onnx/model.onnx" -OutFile "$dest/model.onnx"
```

4. Alternative locale (sans réseau) : lire les leçons via git au commit `8c1834d` — voir `load_documents()` dans `run_homework.py`.
