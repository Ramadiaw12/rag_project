<div align="center">

<!-- ANIMATED HEADER -->
<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:0f2027,50:203a43,100:2c5364&height=200&section=header&text=RAG%20OCP%20Financial&fontSize=52&fontColor=ffffff&fontAlignY=38&desc=Intelligence%20Artificielle%20au%20service%20de%20l'analyse%20financière&descAlignY=60&descColor=a8d8ea&animation=fadeIn"/>

<!-- ANIMATED BADGES -->
<p align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&size=22&duration=3000&pause=500&color=2C9EC2&center=true&vCenter=true&width=600&lines=🤖+RAG+Pipeline+avec+LangChain+%26+GPT-4o;📄+Analyse+de+rapports+PDF+intelligente;💬+Interrogez+vos+documents+en+langage+naturel;⚡+Réponses+précises+et+vérifiables" alt="Typing SVG" />
</p>

<br/>

[![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge&logo=chainlink&logoColor=white)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6B35?style=for-the-badge)](https://trychroma.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

<br/>

> **✨ Et si lire un rapport financier devenait une conversation ?**
>
> *Transformez des centaines de pages PDF en interface conversationnelle intelligente.*

</div>

---

## 🌟 Aperçu

<table>
<tr>
<td width="50%">

### 📄 Chargement des documents
<img src="docchargement.png" width="100%" alt="Upload interface"/>

</td>
<td width="50%">

### 💬 Chat contextuel
<img src="https://github.com/Ramadiaw12/rag_project/blob/205cc06108806e162c3c26a2d4191b03d472eade/imgres.png" width="100%" alt="Chat interface"/>

</td>
</tr>
</table>

<div align="center">

### 🖥️ Dashboard complet
<img src="https://github.com/Ramadiaw12/rag_project/blob/19958e8c57a6c52010a03af7ec6af2221b6ff10b/Capture%20d%E2%80%99%C3%A9cran%20du%202026-03-19%2014-30-15.png" width="85%" alt="Dashboard"/>

</div>

---

## 🧠 Comment ça marche ?

```mermaid
graph TD
    A["📄 Rapports PDF — OCP 2023"] --> B["📖 PyPDFDirectoryLoader\nChargement des documents"]
    B --> C["✂️ RecursiveCharacterTextSplitter\nChunks: 300 tokens · overlap 20"]
    C --> D["🧮 OpenAIEmbeddings ada-002\nVectorisation des textes"]
    D --> E[("🗄️ ChromaDB Vector Store\nCollection: rapport_ocp_V2")]
    E --> F["🔍 Retrieveur sémantique\nTop-K = 5 résultats"]
    F --> G["🔗 Prompt Builder\nContexte + Question"]
    G --> H["🤖 GPT-4o-mini\nGénération de la réponse"]
    H --> I["⚖️ Groundness Checker GPT-4o\nÉvaluation de fidélité 1–5"]

    style A fill:#1e3a5f,color:#a8d8ea
    style B fill:#2c5364,color:#e0f2fe
    style C fill:#203a43,color:#e0f2fe
    style D fill:#1e3a5f,color:#a8d8ea
    style E fill:#0f2027,color:#a8d8ea
    style F fill:#2c5364,color:#e0f2fe
    style G fill:#203a43,color:#e0f2fe
    style H fill:#1e3a5f,color:#a8d8ea
    style I fill:#22c55e,color:#fff
```

---

## 🚀 Stack Technologique

<div align="center">

| Composant | Technologie | Rôle |
|-----------|-------------|------|
| 🔗 **Orchestration** | LangChain | Pipeline RAG complet |
| 🤖 **LLM** | GPT-4o / GPT-4o-mini | Génération & Évaluation |
| 🧮 **Embeddings** | text-embedding-ada-002 | Vectorisation des chunks |
| 🗄️ **Vector Store** | ChromaDB | Stockage & recherche |
| 📄 **Extraction PDF** | PyPDFLoader | Chargement des documents |
| ⚖️ **Évaluation** | LLM-as-a-Judge | Score de groundness |

</div>

---

## 📦 Installation

### 1️⃣ Cloner le dépôt

```bash
git clone https://github.com/Ramadiaw12/rag_project.git
cd rag_project
```

### 2️⃣ Créer l'environnement virtuel

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3️⃣ Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4️⃣ Configurer les variables d'environnement

```bash
cp .env.example .env
# Ouvrez .env et ajoutez votre clé OpenAI
```

### 5️⃣ Placer vos PDFs

```bash
mkdir -p pdfs
# Copiez vos rapports OCP dans pdfs/
```

---

## ⚙️ Configuration

```env
# .env — Variables d'environnement

# 🔑 Obligatoire
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# 📡 Optionnel — LangSmith tracing
LANGCHAIN_API_KEY=ls_xxxxxxxxxxxxxxxxxxxx
LANGCHAIN_PROJECT=rag-ocp-financial
LANGCHAIN_TRACING_V2=true

# ⚙️ Paramètres RAG
CHUNK_SIZE=300
CHUNK_OVERLAP=20
TOP_K_RESULTS=5
MAX_CONTEXT_TOKENS=4000

# 🐛 Debug
DEBUG=false
LOG_LEVEL=INFO
```

---

## 💡 Utilisation

### Initialisation de la base vectorielle

```python
from rag_pipeline import initialize_vectorstore

vectorstore = initialize_vectorstore("./pdfs", force_recreate=False)
```

### Interrogation simple

```python
from rag_pipeline import RAG

response = RAG("Quel est le chiffre d'affaires de l'OCP en 2023 ?")
print(response)
```

### Mode interactif

```python
from rag_pipeline import interactive_qa

interactive_qa()
```

### Évaluation automatique

```python
from evaluation import evaluate_with_metrics

metrics = evaluate_with_metrics("Quelles sont les performances financières ?")
print(f"Score groundness : {metrics['score']}/5")
```

---

## 📊 Exemples de résultats

```
🔍 Q : Quel est le chiffre d'affaires de l'OCP en 2023 ?
📝 R : Le chiffre d'affaires consolidé s'établit à 87,4 milliards de dirhams...

🔍 Q : Comment a évolué l'EBITDA par rapport à 2022 ?
📝 R : Le résultat net part du groupe s'élève à 28,1 milliards de dirhams...

🔍 Q : Je veux dormir (hors contexte)
📝 R : JE NE SAIS PAS
```

---

## ✨ Fonctionnalités

### ✅ Disponibles

| # | Fonctionnalité | Description |
|---|----------------|-------------|
| 🔍 | **Recherche sémantique** | Similarité cosinus via embeddings |
| 🤖 | **Génération RAG** | Réponses contextuelles avec GPT-4o-mini |
| 📄 | **Multi-PDF** | Traitement de tous les PDFs d'un dossier |
| ⚖️ | **Auto-évaluation** | Groundness checker avec GPT-4o |
| 💾 | **Persistance** | Sauvegarde / rechargement Chroma |
| 🔄 | **Lazy loading** | Traitement mémoire optimisé |

### 🚧 En développement

| # | Fonctionnalité | Description |
|---|----------------|-------------|
| 🎯 | Filtrage par métadonnées | Recherche par source/page |
| 📊 | Scores de similarité | Visualisation des scores de retrieval |
| 🔁 | Mode conversationnel | Historique de questions multi-tour |
| 🌐 | API REST FastAPI | Exposition du pipeline via API |
| 📱 | Interface Streamlit | UI utilisateur interactive |

---

## 📊 Métriques d'évaluation

<div align="center">

| Métrique | Échelle | Description |
|----------|---------|-------------|
| 🏆 **Groundness** | 1 → 5 | Fidélité de la réponse au contexte |
| 🚨 **Hallucinations** | Liste | Infos non présentes dans les docs |
| 📐 **Couverture** | 0 → 100% | Proportion de la question répondue |
| 🎯 **Pertinence** | 1 → 5 | Adéquation réponse / question |

</div>

**Exemple de rapport JSON :**

```json
{
  "question": "Quel est le chiffre d'affaires 2023 ?",
  "score": 5,
  "hallucinations": [],
  "faithfulness": true,
  "explanation": "La réponse cite exactement les 87,4 Mds MAD du contexte",
  "context_length": 1250,
  "answer_length": 187,
  "processing_time": 2.3
}
```

---

## 💰 Estimation des coûts OpenAI

| Modèle | Usage | Coût indicatif |
|--------|-------|----------------|
| `gpt-4o-mini` | Génération RAG | $0.15 / 1M tokens (input) |
| `gpt-4o` | Groundness checker | $5.00 / 1M tokens (input) |
| `text-embedding-ada-002` | Vectorisation | $0.13 / 1M tokens |

> 💡 Un rapport de ~200 pages ≈ 1000–1500 chunks ≈ **~$0.15 d'embeddings**

---

## 🗂️ Structure du projet

```
rag_project/
├── 📄 .env                    # Variables d'environnement
├── 📄 .env.example            # Template de configuration
├── 📄 .gitignore
├── 📄 requirements.txt
├── 📄 main.py                 # Point d'entrée principal
├── 📄 rag.py                  # Dashboard Streamlit
├── 📓 RAGV2.ipynb             # Notebook de développement
├── 📁 pdfs/                   # Rapports PDF sources
│   └── Rapport Financier OCP 2023.pdf
├── 📁 store/                  # Base vectorielle Chroma (générée)
└── 📄 README.md
```

---

## 🤝 Contribuer

Les contributions sont les bienvenues ! 🎉

```bash
# 1. Forkez le projet
# 2. Créez votre branche
git checkout -b feature/nouvelle-fonctionnalite

# 3. Commitez vos changements
git commit -m "feat: ajout de la fonctionnalité X"

# 4. Pushez vers votre fork
git push origin feature/nouvelle-fonctionnalite

# 5. Ouvrez une Pull Request 🚀
```

### 🗺️ Roadmap

- [ ] 📊 Interface Streamlit complète
- [ ] 🌐 API REST avec FastAPI
- [ ] 🔁 Mode conversationnel multi-tour
- [ ] 📂 Support Excel & Word
- [ ] 🌍 Support multilingue (FR / EN / AR)
- [ ] 🎯 Fine-tuning sur données financières
- [ ] 🐋 Dockerisation complète

---

## 📄 Licence

Distribué sous la licence **MIT**. Voir [`LICENSE`](LICENSE) pour plus d'informations.

---

<div align="center">

### 🙏 Remerciements

[![LangChain](https://img.shields.io/badge/LangChain-Docs-1C3C3C?style=flat-square&logo=chainlink)](https://python.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-Platform-412991?style=flat-square&logo=openai)](https://openai.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-FF6B35?style=flat-square)](https://trychroma.com)
[![OCP Group](https://img.shields.io/badge/OCP_Group-Rapports_Publics-0066CC?style=flat-square)](https://www.ocpgroup.ma/)

<br/>

---

**Conçu avec ❤️ par [DIAWANE Ramatoulaye](https://github.com/Ramadiaw12)**

*"L'IA ne remplace pas l'analyste — elle lui donne des super-pouvoirs."*

<img width="100%" src="https://capsule-render.vercel.app/api?type=waving&color=0:2c5364,50:203a43,100:0f2027&height=120&section=footer"/>

</div>