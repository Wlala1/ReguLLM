ReguLLM — Regulatory Knowledge Vector Search

Build a local vector database from regulation/legislation text files and run semantic search over them. ReguLLM uses Google Generative AI for text embeddings and ChromaDB for vector storage. It supports common Chinese encodings (UTF-8 / GBK / GB2312).  ￼

⸻

Features
	•	Document ingestion & chunking: Automatically loads and segments files under knowledge/.
	•	Embeddings: Generates text vectors via Google Generative AI (API key required).
	•	Local vector store: Persists to vector_db/ with ChromaDB.
	•	Semantic search: Natural-language queries return the most relevant passages.
	•	Encoding friendly: Detects and handles UTF-8 / GBK / GB2312.  ￼

⸻

Requirements
	•	Python 3.8+ (3.10/3.11 recommended)
	•	Internet connection (needed when building embeddings)  ￼

⸻

Installation

git clone https://github.com/Wlala1/ReguLLM.git
cd ReguLLM
pip install -r requirements.txt

The repository includes a requirements.txt at the project root.  ￼

⸻

Setup

1) Get a Google AI API key

Create a key in Google AI Studio, then set it as an environment variable:

export GOOGLE_API_KEY="your_api_key_here"

(Alternatively, you can edit GOOGLE_API_KEY directly in trans_embedding.py.)  ￼

2) Prepare your regulation texts

Place your .txt files into the knowledge/ folder, for example:

knowledge/
├── regulation_1.txt
├── regulation_2.txt
├── regulation_3.txt
├── regulation_4.txt
└── regulation_5.txt

Files encoded as UTF-8 / GBK / GB2312 are supported.  ￼

⸻

Quickstart

Option A — build the knowledge base

python trans_embedding.py

This reads knowledge/, creates embeddings, and persists them to vector_db/.  ￼

Option B — run the test/interactive script

python test_knowledge_base.py

You can run automatic tests or enter interactive search mode to try queries.  ￼

⸻

Programmatic usage (API)

Use the LegalDocumentVectorStore class from trans_embedding.py to build or load the vector store and perform similarity search:

from trans_embedding import LegalDocumentVectorStore

# 1) Initialize
builder = LegalDocumentVectorStore(
    google_api_key="your_api_key",
    knowledge_dir="./knowledge",
    vector_db_dir="./vector_db",
    chunk_size=1000,     # optional, default 1000
    chunk_overlap=200    # optional, default 200
)

# 2) Build or load the knowledge base
vectorstore = builder.build_knowledge_base()

# 3) Search
results = builder.search_similar_documents(
    vectorstore,
    query="Contract breach liabilities",
    k=5
)

for doc in results:
    print(f"Source: {doc.metadata.get('source')}")
    print(f"Content: {doc.page_content[:200]} ...")

Default parameters you can tune: chunk_size=1000, chunk_overlap=200, knowledge_dir="./knowledge", vector_db_dir="./vector_db".  ￼

⸻

Repository layout

.
├── trans_embedding.py         # build/query the vector store
├── test_knowledge_base.py     # test & interactive demo
├── config.py                  # configuration
├── requirements.txt           # dependencies
├── README.md                  # this file
├── knowledge/                 # place your .txt files here
└── vector_db/                 # created automatically to store vectors

# Additional files present in the repo (for experiments/extensions)
├── app.py
├── main.py
├── build_knowledge_base.py
├── confidence_agent.py
└── interface.html

The top-level repo also includes directories like legal_compliance_db1/. See the GitHub file list for the current state.  ￼

⸻

Troubleshooting
	1.	API key errors / quota
Ensure the environment variable is set, the key is valid, and you have available quota.  ￼
	2.	Encoding issues
Although UTF-8 / GBK / GB2312 are handled automatically, converting to UTF-8 may help in stubborn cases.  ￼
	3.	ChromaDB write failures
Check permissions for vector_db/ and available disk space.  ￼
	4.	Dependency problems
Use Python 3.8+ and consider installing inside a virtual environment.  ￼
	5.	Updating your corpus
After adding new documents, delete the vector_db/ directory and rebuild to ensure a clean index.  ￼

⸻

Roadmap (suggested)
	•	Simple one-command launcher for a web UI (if using app.py/interface.html)
	•	Basic evaluation scripts for retrieval quality
	•	Support for additional formats (PDF/DOCX/HTML) and batch cleaning
	•	Pluggable embedding backends (open-source models/services)
	•	Dockerfile and minimal deployment guide
	•	Jupyter notebook examples

(The items above are suggestions for future improvements.)

⸻

License

For learning and research purposes only. Do not commit your API keys to version control.  ￼

⸻

This README reflects the repository’s current contents and usage patterns visible on GitHub.  ￼