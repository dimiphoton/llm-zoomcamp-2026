# Cheatsheet — Homework 1 : Agentic RAG

Mémo pour refaire le devoir du Module 1. Le notebook complet et exécuté est dans
[`homework1.ipynb`](homework1.ipynb).

## But

Construire un système **RAG** à partir des **pages de leçons** du cours
(au lieu de la FAQ), puis le rendre **agentique** :

1. Télécharger les pages de leçons depuis GitHub (`gitsource`).
2. Indexer + chercher (`minsearch`).
3. Brancher un LLM dessus = RAG, et mesurer les tokens d'entrée.
4. Découper les pages en chunks pour réduire le contexte.
5. Donner un outil `search` au LLM = agent (`toyaikit`).

## Environnement et installation

### Prérequis

- **Python 3.10+** (3.14 recommandé par le cours)
- Un compte **OpenAI** avec une clé API (Q3, Q5 et Q6 appellent le LLM)
- **Git** pour cloner le fork et le synchroniser avec le dépôt officiel
- Familiarité basique avec le terminal et Jupyter

### Récupérer le code

Ce devoir vit dans le fork du cours. Après un **Sync fork** sur GitHub, mettre à jour le dépôt local :

```powershell
git pull origin main
```

Chemin du notebook : `cohorts/2026/01-agentic-rag/homework1_solution/homework1.ipynb`

### Environnement Python (uv + pyproject.toml)

Le cours utilise **[uv](https://docs.astral.sh/uv/)** comme gestionnaire de paquets.
Les dépendances sont déclarées dans [`pyproject.toml`](pyproject.toml) à côté de ce notebook.

```powershell
# Installer uv (Windows) — une seule fois
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Depuis le dossier homework1_solution/
cd cohorts/2026/01-agentic-rag/homework1_solution
uv sync
```

`uv sync` lit le `pyproject.toml`, crée le venv et installe les paquets listés.

Pour ajouter une dépendance plus tard :

```powershell
uv add nom-du-paquet
```

| Package | Rôle |
|---------|------|
| `gitsource` | Télécharge les pages de leçons depuis GitHub |
| `minsearch` | Indexation et recherche par mots-clés |
| `openai` | Client API pour le LLM (`gpt-5.4-mini`) |
| `toyaikit` | Agent avec function calling (Q6) |
| `python-dotenv` | Charge la clé API depuis `.env` |
| `jupyter` | Exécuter le notebook |

### Clé API (`.env`)

Créer un fichier `.env` **à la racine du dépôt** ou **dans ce dossier** (`homework1_solution/`) :

```
OPENAI_API_KEY=sk-...
```

Vérifier que `.env` est bien ignoré par Git (le `.gitignore` du cours le couvre déjà).

Chargé dans le notebook par :

```python
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))  # remonte l'arborescence pour trouver .env
```

Test rapide avant de lancer le devoir :

```python
from openai import OpenAI
client = OpenAI()
print("OK" if client.api_key else "Clé manquante")
```

### Lancer Jupyter

Depuis le dossier `homework1_solution/` :

```powershell
uv run jupyter notebook homework1.ipynb
```

`uv run` utilise automatiquement le venv créé par `uv sync`.

### Rejouer le notebook sans interface

```powershell
uv run jupyter nbconvert --to notebook --execute --inplace homework1.ipynb
```

Les questions **Q1, Q2 et Q4** ne consomment pas de crédits API (pas d'appel LLM).

### Dépannage TLS (Windows + Conda)

Si `uv sync` échoue avec `invalid peer certificate: UnknownIssuer`, c'est que **Conda**
a défini `SSL_CERT_FILE` vers un bundle incomplet. Correctifs appliqués sur ce poste :

1. **`%APPDATA%\uv\uv.toml`** avec `system-certs = true`
2. **Profil PowerShell** : retire `SSL_CERT_FILE` après `conda activate`

Vérification rapide (dans un **nouveau** terminal PowerShell) :

```powershell
uv sync   # doit télécharger sans erreur TLS
```

Pour **Git** (fetch upstream), si le même problème apparaît :

```powershell
git config --global http.sslBackend schannel
```

## Étapes clés (commandes Python)

### Préparation — télécharger les leçons (commit figé `8c1834d`)

```python
from gitsource import GithubRepositoryDataReader

reader = GithubRepositoryDataReader(
    repo_owner="DataTalksClub",
    repo_name="llm-zoomcamp",
    commit_id="8c1834d",
    allowed_extensions={"md"},
    filename_filter=lambda path: "/lessons/" in path,
)
documents = [f.parse() for f in reader.read()]  # -> dicts {filename, content}
```

### Q1 — nombre de pages

```python
len(documents)
```

### Q2 — indexation + recherche

```python
from minsearch import Index

index = Index(text_fields=["content"], keyword_fields=["filename"])
index.fit(documents)
results = index.search("How does the agentic loop keep calling the model until it stops?", num_results=5)
results[0]["filename"]
```

### Q3 — RAG + comptage des tokens

Point clé : adapter `RAGBase` (schéma FAQ) au schéma `filename`/`content`, et
faire en sorte que `llm()` renvoie la **réponse complète** pour lire le `usage` :

```python
response = client.responses.create(model="gpt-5.4-mini", input=messages)
answer = response.output_text
input_tokens = response.usage.input_tokens   # <- tokens d'entree
```

### Q4 — chunking (fenêtre glissante)

```python
from gitsource import chunk_documents

chunks = chunk_documents(documents, size=2000, step=1000)  # overlap = size - step
len(chunks)
```

### Q5 — RAG sur les chunks

Réindexer les `chunks` (même schéma), relancer la même requête, comparer
`usage.input_tokens` avec Q3.

### Q6 — agent avec un outil `search` (toyaikit)

Le type hint + la docstring suffisent : toyaikit génère le schéma de l'outil.

```python
from toyaikit.tools import Tools
from toyaikit.chat.runners import OpenAIResponsesRunner

def search(query: str) -> list[dict]:
    """Search the course lessons for chunks matching the given query."""
    return chunk_index.search(query, num_results=5)

tools = Tools()
tools.add_tool(search)
result = runner.loop(prompt="...", callback=callback)
```

Pour compter les appels : un compteur global incrémenté dans `search`.

## Réponses obtenues

| Q | Réponse | Mesure |
|---|---------|--------|
| Q1 | **72** | 72 pages |
| Q2 | **`01-agentic-rag/lessons/14-agentic-loop.md`** | 1er résultat |
| Q3 | **7000** | 7110 tokens |
| Q4 | **295** | 295 chunks |
| Q5 | **3× moins** | 7110 → 2293 (ratio 3,1×) |
| Q6 | **4** | 4 appels à `search` |

Soumission : <https://courses.datatalks.club/llm-zoomcamp-2026/homework/hw1>
