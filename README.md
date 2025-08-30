# ReguLLM — Regulatory Knowledge Vector Search

Build a local vector database from regulation/legislation text files and run semantic search over them. ReguLLM uses Google Generative AI for text embeddings and pickle for vector storage. 

-----

## ✨ Features

  * **Document ingestion & chunking**: Automatically loads and segments files under `knowledge/`.
  * **Embeddings**: Generates text vectors via Qwen Generative AI (API key required).
  * **Semantic search**: Natural-language queries return the most relevant passages.


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

Create a key in Qwen, then set it as an environment variable:

```bash
export DASHSCOPE_API_KEY="your_api_key_here"
```

(Alternatively, you can edit the `Qwen` directly in `trans_embedding.py`.)

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

### Step 1 — Build the knowledge base

Reads files from `knowledge/`, creates embeddings, builds the (graph) index, and saves .pkl artifacts to `graph_store/` (configurable).

```bash
python build_knowledge_base.py
```

### Step 2 — Start the service

Launch the lightweight web app for GraphRAG search/Q&A.

```bash
python app.py
```

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

## ⚖️ License

For learning and research purposes only. **Do not commit your API keys to version control.**

我这里用的是pikle 存储graphRAG， 改一下