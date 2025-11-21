# 📚 Clean RAG System

A production-ready Retrieval Augmented Generation (RAG) system with smart table handling, strict citation requirements, and comprehensive evaluation.

## 🌟 Features

- **Smart PDF Ingestion**: Detects and separately processes tables vs text
- **Table Summarization**: Generates LLM-friendly summaries of tables
- **Vector Storage**: ChromaDB (local) with easy swap to Pinecone
- **Strict Citations**: System prompt enforces context-only answers
- **No Hallucination**: Built-in guardrails against making up information
- **RAGAS Evaluation**: Measure faithfulness and other quality metrics
- **Streamlit UI**: Simple chat interface for document Q&A

## 📁 Project Structure

```
clean-rag/
├── .env                    # API keys (create from .env.template)
├── requirements.txt        # Python dependencies
├── app.py                  # Streamlit chat interface
├── README.md               # This file
├── src/
│   ├── __init__.py         # Package initialization
│   ├── ingestion.py        # PDF loading, table detection, chunking
│   ├── retrieval.py        # Vector store setup and retrieval
│   ├── chain.py            # RAG chain with citation prompts
│   └── evaluation.py       # RAGAS evaluation scripts
└── chroma_db/              # Local vector database (auto-created)
```

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Create project directory
mkdir clean-rag && cd clean-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy template
cp .env.template .env

# Edit .env and add your OpenAI API key
# OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the Application

```bash
# Start the Streamlit UI
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## 💻 Usage Examples

### Programmatic Usage

```python
from src.ingestion import ingest_pdf
from src.chain import create_rag_chain

# 1. Ingest a PDF
chunks = ingest_pdf("my_document.pdf")
print(f"Created {len(chunks)} chunks")

# 2. Create RAG chain and add documents
rag = create_rag_chain()
rag.add_documents(chunks)

# 3. Ask questions
response = rag.invoke("What are the key findings?")
print(response.answer)

# 4. View sources
for doc in response.sources:
    print(f"- {doc.metadata['source']}: {doc.page_content[:100]}...")
```

### Command Line Ingestion

```bash
# Ingest a PDF from command line
python -m src.ingestion path/to/document.pdf
```

### Run Evaluation

```bash
# Full evaluation with all RAGAS metrics
python -m src.evaluation --full

# Quick faithfulness-only test
python -m src.evaluation --quick
```

## ⚙️ Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | Required | Your OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Model for generation |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Model for embeddings |
| `CHUNK_SIZE` | `1000` | Text chunk size |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | ChromaDB storage path |
| `CHROMA_COLLECTION_NAME` | `clean_rag_docs` | Collection name |

### Switching to Pinecone

1. Install Pinecone: `pip install pinecone-client langchain-pinecone`

2. Add to `.env`:
```bash
PINECONE_API_KEY=your-key
PINECONE_INDEX_NAME=clean-rag-index
```

3. Change provider in code:
```python
from src.retrieval import get_vector_store_provider

# Use Pinecone instead of ChromaDB
provider = get_vector_store_provider("pinecone")
```

## 📊 Evaluation Metrics

The system uses RAGAS for evaluation:

| Metric | Description | Target |
|--------|-------------|--------|
| **Faithfulness** | Answer only uses context info | > 0.9 |
| **Answer Relevancy** | Answer addresses the question | > 0.8 |
| **Context Precision** | Retrieved docs are relevant | > 0.8 |
| **Context Recall** | All needed info was retrieved | > 0.8 |

## 🔧 Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   PDF Upload    │────▶│   Ingestion     │────▶│   ChromaDB      │
│                 │     │  (unstructured) │     │  (embeddings)   │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│    Answer       │◀────│   RAG Chain     │◀────│   Retriever     │
│  (with cites)   │     │  (LangChain)    │     │   (top-k)       │
└─────────────────┘     └─────────────────┘     └─────────────────┘
        │
        ▼
┌─────────────────┐
│    RAGAS        │
│   Evaluation    │
└─────────────────┘
```

## 🛡️ Anti-Hallucination Design

The system uses multiple layers to prevent hallucination:

1. **System Prompt**: Explicitly commands "Answer only based on provided context"
2. **Temperature 0**: Deterministic outputs reduce creativity/hallucination
3. **Source Citations**: Forces attribution to specific sources
4. **"I don't know"**: Trained to admit when context lacks answer
5. **RAGAS Evaluation**: Measures faithfulness to detect issues

## 📝 License

MIT License - feel free to use and modify.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run evaluation to ensure quality
5. Submit a pull request
