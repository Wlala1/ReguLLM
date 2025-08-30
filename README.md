# ReguLLM — Regulatory Knowledge Vector Search

Build a local vector database from regulation/legislation text files and run semantic search over them. ReguLLM uses Google Generative AI for text embeddings and ChromaDB for vector storage. It supports common Chinese encodings (UTF-8 / GBK / GB2312).

-----

## ✨ Features

  * **Document ingestion & chunking**: Automatically loads and segments files under `knowledge/`.
  * **Embeddings**: Generates text vectors via Google Generative AI (API key required).
  * **Local vector store**: Persists to `vector_db/` with ChromaDB.
  * **Semantic search**: Natural-language queries return the most relevant passages.
  * **Encoding friendly**: Detects and handles UTF-8 / GBK / GB2312.

-----

## 📋 Requirements

  * Python 3.8+ (3.10/3.11 recommended)
  * Internet connection (needed when building embeddings)

-----

## 🚀 Installation

1.  Clone the repository:

    ```bash
    git clone https://github.com/Wlala1/ReguLLM.git
    ```

2.  Navigate to the project directory:

    ```bash
    cd ReguLLM
    ```

3.  Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

    The repository includes a `requirements.txt` at the project root.

-----

## 🔧 Setup

### 1\) Get a Google AI API key

Create a key in [Google AI Studio](https://aistudio.google.com/), then set it as an environment variable:

```bash
export GOOGLE_API_KEY="your_api_key_here"
```

(Alternatively, you can edit the `GOOGLE_API_KEY` directly in `trans_embedding.py`.)

### 2\) Prepare your regulation texts

Place your `.txt` files into the `knowledge/` folder. For example:

```
knowledge/
├── regulation_1.txt
├── regulation_2.txt
├── regulation_3.txt
├── regulation_4.txt
└── regulation_5.txt
```

Files encoded as UTF-8 / GBK / GB2312 are supported.

-----

## ⚡ Quickstart

### Option A — Build the knowledge base

This command reads files from `knowledge/`, creates embeddings, and saves them to `vector_db/`.

```bash
python trans_embedding.py
```

### Option B — Run the test/interactive script

This script allows you to run automated tests or enter an interactive search mode to try queries.

```bash
python test_knowledge_base.py
```

-----

## ⚙️ Programmatic Usage (API)

Use the `LegalDocumentVectorStore` class from `trans_embedding.py` to build or load the vector store and perform similarity searches.

```python
from trans_embedding import LegalDocumentVectorStore

# 1) Initialize the store
builder = LegalDocumentVectorStore(
    google_api_key="your_api_key",
    knowledge_dir="./knowledge",
    vector_db_dir="./vector_db",
    chunk_size=1000,     # optional, default 1000
    chunk_overlap=200    # optional, default 200
)

# 2) Build or load the knowledge base
# This will create embeddings if the vector store doesn't exist
vectorstore = builder.build_knowledge_base()

# 3) Search for relevant documents
results = builder.search_similar_documents(
    vectorstore,
    query="Contract breach liabilities",
    k=5
)

# Print the results
for doc in results:
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Content: {doc.page_content[:200]} ...")
    print("-" * 20)

```

**Default parameters you can tune**: `chunk_size=1000`, `chunk_overlap=200`, `knowledge_dir="./knowledge"`, `vector_db_dir="./vector_db"`.

-----

## 📁 Repository Layout

```
.
├── trans_embedding.py         # Main script to build/query the vector store
├── test_knowledge_base.py     # Test script & interactive demo
├── config.py                  # Configuration settings
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── knowledge/                 # Place your .txt files here
└── vector_db/                 # Created automatically to store vectors
```

**Additional files** (for experiments/extensions):

  * `app.py`
  * `main.py`
  * `build_knowledge_base.py`
  * `confidence_agent.py`
  * `interface.html`

*The top-level repo also includes directories like `legal_compliance_db1/`. See the GitHub file list for the current state.*

-----

## 💡 Troubleshooting

1.  **API key errors / quota**

      * Ensure the environment variable is set correctly, the key is valid, and you have not exceeded your API quota.

2.  **Encoding issues**

      * Although UTF-8 / GBK / GB2312 are handled automatically, converting files to UTF-8 beforehand may help in stubborn cases.

3.  **ChromaDB write failures**

      * Check directory permissions for `vector_db/` and ensure you have sufficient available disk space.

4.  **Dependency problems**

      * Ensure you are using Python 3.8 or newer. Consider installing dependencies inside a virtual environment (`venv`) to avoid conflicts.

5.  **Updating your corpus**

      * After adding, removing, or modifying documents in `knowledge/`, delete the `vector_db/` directory and re-run `python trans_embedding.py` to rebuild a clean index.

-----

## 🗺️ Roadmap (Suggested)

  * Simple one-command launcher for a web UI (if using `app.py`/`interface.html`).
  * Basic evaluation scripts for retrieval quality.
  * Support for additional formats (PDF/DOCX/HTML) and batch cleaning.
  * Pluggable embedding backends (open-source models/services).
  * Dockerfile and minimal deployment guide.
  * Jupyter notebook examples.

-----

## ⚖️ License

For learning and research purposes only. **Do not commit your API keys to version control.**