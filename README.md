# 📊 RAG OCP Financial Reports - Analyse des rapports financiers 2023

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/langchain-0.1.0-green.svg)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT4--o-purple.svg)](https://openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table des matières
- [Description du projet](#description-du-projet)
- [Architecture](#architecture)
- [Prérequis](#prérequis)
- [Installation](#installation)
- [Configuration](#configuration)
- [Utilisation](#utilisation)
- [Fonctionnalités](#fonctionnalités)
- [Structure du code](#structure-du-code)
- [Évaluation et métriques](#évaluation-et-métriques)
- [Exemples](#exemples)
- [Déploiement](#déploiement)
- [Contribution](#contribution)
- [License](#license)

## Dashboard

---

## 💡 L'idée derrière le projet

> *Marre de parcourir 200 pages pour trouver un seul chiffre ?*
> *J'ai construit l'outil que j'aurais voulu avoir.*

J'ai toujours été fasciné par une question simple : **et si on pouvait parler à ses documents comme à un humain ?**

C'est exactement ce que fait **DocMind**. Charge n'importe quel PDF, pose ta question en français (ou en anglais), et reçois une réponse précise — tirée directement du contenu de ton document. Pas d'invention, pas d'approximation. Juste ton document, qui te répond.

Sous le capot, c'est un pipeline **RAG (Retrieval Augmented Generation)** complet que j'ai conçu de A à Z : extraction du texte, découpage intelligent en chunks, vectorisation via OpenAI Embeddings, recherche par similarité cosinus dans ChromaDB, et génération de réponse avec GPT-4o. Le tout enveloppé dans une interface Streamlit en **mode nuit**, parce que les belles choses méritent un beau cadre.

---

## 📸 Aperçu

<div align="center">

<!-- SIDEBAR — chargement PDF -->
![image alt](https://github.com/Ramadiaw12/rag_project/blob/8f4a7d8e5f2f33b0add59876ccceaac6511cca21/imgpdf.png)
<!-- CHAT — réponse contextuelle -->
<img src="" width="30%" alt="Chat — réponse ancrée dans les sources"/>
&nbsp;
<!-- 🖼️ CONTEXTE — chunks retrouvés -->
<img src="assets/screenshot_context.png" width="30%" alt="Panneau contexte — chunks sources"/>

</div>

<div align="center">

<!-- 🖼️ DASHBOARD COMPLET — vue globale -->
<img src="assets/screenshot_full.png" width="92%" alt="Dashboard complet — vue night mode"/>

</div>

> 📁 **Pour ajouter tes captures** : crée un dossier `assets/` à la racine et dépose tes screenshots dedans, puis retire les commentaires `<!-- -->`.

---

1. Description du projet
markdown

## 🎯 Description du projet

Ce projet implémente un système de **Question-Réponse basé sur RAG (Retrieval Augmented Generation)** pour l'analyse des rapports financiers de l'OCP (Office Chérifien des Phosphates) pour l'année 2023.

### Objectifs
- 📈 Permettre l'interrogation en langage naturel des rapports financiers
- 🔍 Extraire des informations précises à partir de documents PDF
- 🤖 Générer des réponses contextualisées et fidèles aux documents sources
- ✅ Évaluer automatiquement la qualité des réponses (groundness)

### Problématique résolue
Les rapports financiers annuels sont des documents denses et complexes (souvent 200+ pages). Notre système permet aux analystes, investisseurs et équipes financières d'obtenir rapidement des réponses précises sans lecture exhaustive.

### Technologies utilisées
- **LangChain** - Orchestration du pipeline RAG
- **OpenAI GPT-4o / GPT-4o-mini** - Génération de réponses et évaluation
- **ChromaDB** - Base de données vectorielle
- **PyPDFLoader** - Extraction des PDFs
- **python-dotenv** - Gestion des variables d'environnement
2. Architecture
markdown
## 🏗️ Architecture du système
┌─────────────────────────────────────┐
│ Documents PDF (OCP) │
│ Rapport Financier Annuel 2023.pdf │
└────────────────┬────────────────────┘
↓
┌─────────────────────────────────────┐
│ PyPDFDirectoryLoader │
│ Chargement des documents │
└────────────────┬────────────────────┘
↓
┌─────────────────────────────────────┐
│ RecursiveCharacterTextSplitter │
│ Chunks: 300 tokens, overlap 20 │
└────────────────┬────────────────────┘
↓
┌─────────────────────────────────────┐
│ OpenAIEmbeddings (ada-002) │
│ Vectorisation des textes │
└────────────────┬────────────────────┘
↓
┌─────────────────────────────────────┐
│ ChromaDB Vector Store │
│ Collection: rapport_ocp_V2 │
└────────────────┬────────────────────┘
↓
┌─────────────────────────────────────────────────────────────────┐
│ PIPELINE RAG │
├─────────────────────────────────────────────────────────────────┤
│ Question → Retrieveur (k=5) → Contexte → Prompt → LLM → Réponse │
└─────────────────────────────────────────────────────────────────┘
↓
┌─────────────────────────────────────┐
│ Groundness Checker (GPT-4o) │
│ Évaluation de la fidélité │
└─────────────────────────────────────┘

text

### Flux de données
1. **Ingestion**: Chargement des PDFs et découpage en chunks (300 tokens)
2. **Indexation**: Vectorisation des chunks et stockage dans ChromaDB
3. **Recherche**: Similarité cosinus pour trouver les chunks pertinents
4. **Génération**: Construction d'un prompt avec contexte et question
5. **Évaluation**: Vérification automatique de la qualité des réponses
3. Prérequis
markdown
## 🔧 Prérequis

- **Python 3.9 ou supérieur**
- **Clé API OpenAI** (avec accès aux modèles gpt-4o et text-embedding-ada-002)
- **Environnement Unix/Linux/MacOS ou Windows** (avec WSL recommandé pour Windows)

### Modèles OpenAI utilisés
| Modèle | Usage | Coût approximatif |
|--------|-------|-------------------|
| `gpt-4o-mini` | Génération de réponses RAG | $0.15 / 1M tokens (input), $0.60 / 1M tokens (output) |
| `gpt-4o` | Évaluation (groundness checker) | $5.00 / 1M tokens (input), $15.00 / 1M tokens (output) |
| `text-embedding-ada-002` | Vectorisation des chunks | $0.13 / 1M tokens |

> **Note**: Les coûts sont indicatifs. Un rapport financier typique de 200 pages représente environ 1000-1500 chunks, soit ~$0.15 d'embeddings.
4. Installation
markdown
## 📦 Installation

### 1. Cloner le dépôt
```bash
git clone https://github.com/votre-username/rag-ocp-financial.git
cd rag-ocp-financial
2. Créer un environnement virtuel
bash
python -m venv venv
source venv/bin/activate  # Sur Windows: venv\Scripts\activate
3. Installer les dépendances
bash
pip install -r requirements.txt
4. Configurer les variables d'environnement
bash
cp .env.example .env
# Éditez .env avec votre clé API OpenAI
5. Placer les documents PDF
bash
mkdir -p pdfs
# Copiez vos rapports financiers OCP dans le dossier pdfs/
Fichier requirements.txt
text
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.1.0
chromadb>=0.4.0
python-dotenv>=1.0.0
pypdf>=3.0.0
ipython>=8.0.0  # Pour Jupyter/Colab
jupyter>=1.0.0  # Optionnel
tenacity>=8.0.0 # Pour les retry
termcolor>=2.0.0 # Pour couleur console
text

---

## **5. Configuration**

```markdown
## ⚙️ Configuration

### Variables d'environnement (.env)
```env
# Obligatoire
OPENAI_API_KEY=votre_clé_api_ici

# Optionnel (pour tracking)
LANGCHAIN_API_KEY=votre_clé_langsmith
LANGCHAIN_PROJECT=rag-ocp-financial

# Configuration des modèles (optionnel)
CHUNK_SIZE=300
CHUNK_OVERLAP=20
TOP_K_RESULTS=5
Structure des dossiers
text
rag-ocp-financial/
├── .env                        # Variables d'environnement
├── 📁 pdfs/                    # Dossier contenant les rapports PDF
│   └── Rapport Financier Annuel OCP 2023.pdf
├── 📁 store/                   # Base vectorielle Chroma (générée)
├── .env
├── .gitignore 
├── .python-version
├── main.py
├── pyproject.toml
├── rag.png    
├── 📁 rag.py                    # code source du dashboard 
├── 📁 RAGV2.ipnb                    # Code source du projet
├── README.md                                           
└── requirements.txt             
text

---

## **6. Utilisation**

```markdown
## 🚀 Utilisation

### 1. Initialisation de la base vectorielle (première utilisation)
```python
from rag_pipeline import initialize_vectorstore

# Charge les PDFs, crée les chunks et indexe dans Chroma
vectorstore = initialize_vectorstore("./pdfs", force_recreate=False)
2. Interrogation simple
python
from rag_pipeline import RAG

# Posez une question
response = RAG("Quel est le chiffre d'affaires de l'OCP en 2023?")
print(response)
3. Mode interactif
python
from rag_pipeline import interactive_qa

# Lancez une session interactive
interactive_qa()
4. Évaluation automatique
python
from evaluation import evaluate_with_metrics

# Évalue la qualité de la réponse
metrics = evaluate_with_metrics("Quelles sont les performances financières?")
print(f"Score groundness: {metrics['score']}/5")
5. Script complet d'exemple
python
# example.py
from rag_pipeline import RAG
from evaluation import evaluate
import os

# Configuration
os.makedirs("./pdfs", exist_ok=True)

# Question
question = "État du résultat global consolidé"
print(f"🔍 Question: {question}")

# Réponse
answer = RAG(question)
print(f"📝 Réponse: {answer[:200]}...")

# Évaluation (optionnel)
if os.getenv("OPENAI_API_KEY"):
    eval_result = evaluate(question)
    print(f"⚖️ Évaluation: {eval_result}")
text

---

## **7. Fonctionnalités**

```markdown
## ✨ Fonctionnalités

### Core Features
| Fonctionnalité | Description | Statut |
|----------------|-------------|--------|
| 🔍 Recherche sémantique | Trouve les passages pertinents via embeddings | ✅ |
| 🤖 Génération RAG | Réponses contextuelles avec GPT-4o-mini | ✅ |
| 📄 Support multi-PDF | Traite tous les PDFs d'un dossier | ✅ |
| ⚖️ Évaluation automatique | Vérification de groundness avec GPT-4o | ✅ |
| 💾 Persistance | Sauvegarde/recharge de la base vectorielle | ✅ |
| 🔄 Lazy loading | Traitement mémoire optimisé pour gros PDFs | ✅ |

### Advanced Features
| Fonctionnalité | Description | Statut |
|----------------|-------------|--------|
| 🎯 Filtrage par métadonnées | Recherche par source/page | 🚧 En développement |
| 📊 Visualisation des scores | Affichage des similarités | 🚧 En développement |
| 🔁 Mode conversationnel | Historique de questions | 🚧 En développement |
| 🌐 API REST | Exposition via FastAPI | 🚧 En développement |
| 📱 Interface Streamlit | UI utilisateur | 🚧 En développement |
8. Structure du code détaillée
markdown
## 📁 Structure du code

### `rag_pipeline.py` - Pipeline principal
```python
# Fonctions principales
- initialize_vectorstore()  # Crée ou charge la base vectorielle
- RAG()                     # Pipeline complet question → réponse
- interactive_qa()          # Mode interactif
- process_pdfs_lazily()     # Traitement optimisé mémoire
evaluation.py - Système d'évaluation
python
# Fonctions d'évaluation
- evaluate()                 # Évaluation simple
- evaluate_with_metrics()    # Évaluation avec métriques structurées
- batch_evaluate()           # Évaluation multiple
- GroundnessChecker class    # Évaluateur avec cache
utils.py - Utilitaires
python
# Fonctions utilitaires
- setup_logging()            # Configuration des logs
- validate_env()             # Validation des variables d'environnement
- estimate_tokens()          # Estimation du nombre de tokens
- save_interaction()         # Sauvegarde Q/R pour audit
Configuration des prompts
python
# prompts.py
RAG_TEMPLATE = """
Answer the following question based only on provided context
<context>
{context}
</context>
<question>
{question}
</question>
If the answer is not found, answer: JE NE SAIS PAS
"""

GROUNDNESS_TEMPLATE = """
Tu es un juge évaluant la fidélité au contexte.
Score (1-5): ...
"""
text

---

## **9. Évaluation et métriques**

```markdown
## 📊 Évaluation et métriques

### Métriques de groundness
Le système intègre un auto-évaluateur (LLM-as-a-judge) qui mesure:

| Métrique | Description | Échelle |
|----------|-------------|---------|
| **Score de groundness** | Fidélité de la réponse au contexte | 1-5 |
| **Hallucinations** | Informations non présentes dans les docs | Liste |
| **Couverture** | Proportion de la question répondue | 0-100% |
| **Pertinence** | Adéquation de la réponse à la question | 1-5 |

### Exemple de rapport d'évaluation
```json
{
  "question": "Quel est le chiffre d'affaires 2023?",
  "score": 5,
  "hallucinations": [],
  "faithfulness": true,
  "explanation": "La réponse cite exactement les 87,4 Mds MAD présents dans le contexte",
  "context_length": 1250,
  "answer_length": 187,
  "processing_time": 2.3
}
Tests de robustesse
bash
# Lancer les tests unitaires
pytest tests/

# Tester avec différentes questions
python -m src.test_suite
text

---

## **10. Exemples d'utilisation**

```markdown
## 💡 Exemples

### Requêtes financières typiques
```python
questions = [
    "Quel est le chiffre d'affaires de l'OCP en 2023?",
    "Comment a évolué l'EBITDA par rapport à 2022?",
    "État du résultat global consolidé",
    "Quels sont les principaux risques mentionnés?",
    "Quelle est la politique de dividendes?"
]

for q in questions:
    print(f"\nQ: {q}")
    print(f"R: {RAG(q)[:200]}...")
Résultats attendus
text
Q: Quel est le chiffre d'affaires de l'OCP en 2023?
R: Le chiffre d'affaires consolidé s'établit à 87,4 milliards de dirhams en 2023...

Q: État du résultat global consolidé
R: Le résultat net part du groupe s'élève à 28,1 milliards de dirhams...

Q: Je veux dormir (hors contexte)
R: JE NE SAIS PAS
text

---

## **11. Déploiement**

```markdown
## 🚢 Déploiement

### Option 1: API FastAPI
```python
# api.py
from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline import RAG

app = FastAPI()

class Query(BaseModel):
    question: str
    user_id: str = None

@app.post("/ask")
async def ask(query: Query):
    answer = RAG(query.question)
    return {"question": query.question, "answer": answer}

# uvicorn api:app --reload
Option 2: Interface Streamlit
python
# app.py
import streamlit as st
from rag_pipeline import RAG

st.title("📊 RAG OCP Financial Assistant")
question = st.text_input("Posez votre question:")
if question:
    with st.spinner("Recherche en cours..."):
        answer = RAG(question)
    st.markdown(answer)
Option 3: Docker
dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "src.interactive"]
text

---

## **12. Bonnes pratiques et optimisation**

```markdown
## 🎯 Bonnes pratiques

### Optimisation des coûts
1. **Cache des embeddings**: Évite de re-vectoriser les mêmes textes
2. **Chunks optimisés**: 300 tokens équilibre précision/contexte
3. **Reuse de la base**: Chargez Chroma une seule fois
4. **Lazy loading**: Pour les très gros documents

### Gestion des erreurs
```python
try:
    response = RAG(question)
except Exception as e:
    logger.error(f"Erreur: {e}")
    response = "Service temporairement indisponible"
Monitoring
Logs structurés (JSON)

Métriques de performance

Alertes sur coûts API

text

---

## **13. Contribution**

```markdown
## 🤝 Contribution

Les contributions sont les bienvenues !

### Comment contribuer
1. Fork le projet
2. Créez votre branche (`git checkout -b feature/amazing-feature`)
3. Committez vos changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing-feature`)
5. Ouvrez une Pull Request

### Roadmap
- [ ] Support d'autres formats (Excel, Word)
- [ ] Interface graphique Streamlit
- [ ] Support multilingue
- [ ] Fine-tuning sur données financières
- [ ] Intégration avec bases de données externes
14. License
markdown
## 📄 License

Distribué sous la licence MIT. Voir `LICENSE` pour plus d'informations.

## 📧 Contact

Votre Nom - [@votre_twitter](https://twitter.com/...) - email@example.com

Lien du projet: [https://github.com/votre-username/rag-ocp-financial](https://github.com/votre-username/rag-ocp-financial)

## 🙏 Remerciements
- [LangChain](https://python.langchain.com/)
- [OpenAI](https://openai.com/)
- [ChromaDB](https://www.trychroma.com/)
- [OCP Group](https://www.ocpgroup.ma/) pour les rapports publics
Fichier .env.example
env
# OpenAI API Key - Obligatoire
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Optionnel - LangSmith pour le tracing
LANGCHAIN_API_KEY=ls_xxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=rag-ocp-financial
LANGCHAIN_TRACING_V2=true

# Configuration du système
CHUNK_SIZE=300
CHUNK_OVERLAP=20
TOP_K_RESULTS=5
MAX_CONTEXT_TOKENS=4000

# Mode debug
DEBUG=false
LOG_LEVEL=INFO
Badges et visuels
Vous pouvez ajouter ces badges en haut du README:

markdown
[![Python](https://img.shields.io/badge/python-3.9+-blue?logo=python)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green?logo=langchain)](https://langchain.com)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT4--o-purple?logo=openai)](https://openai.com)
[![Chroma](https://img.shields.io/badge/ChromaDB-0.4.0-orange)](https://trychroma.com)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
Résumé final
Ce README professionnel couvre:

Présentation claire du projet et de ses objectifs

Architecture détaillée avec diagramme ASCII

Installation pas à pas avec toutes les dépendances

Configuration complète (variables d'environnement)

Exemples d'utilisation concrets

Documentation technique des fonctions principales

Métriques et évaluation du système

Options de déploiement (API, Streamlit, Docker)

Bonnes pratiques d'optimisation

Guide de contribution et roadmap

Le README est conçu pour être:

Complet mais pas trop long

Structuré avec des sections claires

Professionnel avec des badges et du formatage

Pratique avec des exemples utilisables directement
