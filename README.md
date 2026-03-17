# WebToRAG

Turn any website into a local, searchable knowledge base you can query in plain English.

WebToRAG crawls web content, converts it into vector embeddings, and lets you interact with the result through a command-line interface, all running locally with no data leaving your machine.

---

## How it works

1. **Crawl** — scrapes structured content from your target URLs
2. **Embed** — converts the content into vector embeddings stored locally
3. **Query** — ask questions via the CLI; the system retrieves relevant content and passes it to a local LLM for a contextual response

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com) running locally with your chosen model pulled

```bash
pip install requests beautifulsoup4 tqdm sentence-transformers numpy pandas scikit-learn ollama
```

---

## Usage

### 1. Configure your target

Edit `pipeline/config/crawl_and_extract_sections.json` to set the URLs and domains you want to crawl.

### 2. Crawl and extract

```bash
python pipeline/crawl_and_extract_sections.py
```

### 3. Generate embeddings

```bash
python pipeline/generate_vectors.py
```

### 4. Query

```bash
python user_interfaces/ask_rag_cli.py
```

Or run the full pipeline in one step:

```bash
python pipeline/run_pipeline.py
```

---

## Configuration

| File | Purpose |
|---|---|
| `crawl_and_extract_sections.json` | Target URLs, domains, and crawl depth |
| `generate_vectors.json` | Chunking strategy, embedding model, vector storage settings |
| `llm_config.json` | Local LLM model and parameters |

---

## Project structure

```
WebToRAG/
├── pipeline/
│   ├── run_pipeline.py
│   ├── crawl_and_extract_sections.py
│   ├── generate_vectors.py
│   └── config/
│       ├── crawl_and_extract_sections.json
│       └── generate_vectors.json
└── user_interfaces/
    ├── ask_rag_cli.py
    └── config/
        └── llm_config.json
```
