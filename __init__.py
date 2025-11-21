"""
Clean RAG System - Source Package
==================================

A production-ready Retrieval Augmented Generation system with:
- Smart PDF ingestion with table detection
- Vector storage (ChromaDB/Pinecone)
- Citation-based answers
- RAGAS evaluation

Modules:
- ingestion: PDF processing and chunking
- retrieval: Vector store and retrieval
- chain: RAG logic and LLM integration
- evaluation: RAGAS-based evaluation
"""

from src.ingestion import PDFIngestionPipeline, ingest_pdf
from src.retrieval import RAGRetriever, get_vector_store_provider
from src.chain import RAGChain, create_rag_chain

__all__ = [
    "PDFIngestionPipeline",
    "ingest_pdf",
    "RAGRetriever",
    "get_vector_store_provider",
    "RAGChain",
    "create_rag_chain",
]

__version__ = "1.0.0"