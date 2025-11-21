"""
retrieval.py - Vector Store and Retrieval Module
=================================================
This module handles vector storage and retrieval:
1. Setting up ChromaDB (with easy swap to Pinecone)
2. Creating embeddings using OpenAI
3. Storing and retrieving documents
4. Configurable retrieval strategies

Author: Senior AI Engineer
"""

import os
from typing import List, Optional, Protocol
from abc import ABC, abstractmethod
from dotenv import load_dotenv

# LangChain components
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# For type hints
from langchain.vectorstores.base import VectorStore
from langchain.schema import BaseRetriever

# Load environment variables
load_dotenv()


# ============================================
# Abstract Base Class for Vector Store
# This allows easy swapping between providers
# ============================================

class VectorStoreProvider(ABC):
    """
    Abstract base class for vector store providers.
    
    Implement this interface to add new vector store backends
    (e.g., Pinecone, Weaviate, Qdrant, etc.)
    """
    
    @abstractmethod
    def get_vectorstore(self) -> VectorStore:
        """Return the underlying vector store instance."""
        pass
    
    @abstractmethod
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def get_retriever(self, **kwargs) -> BaseRetriever:
        """Get a retriever instance for the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search."""
        pass


# ============================================
# ChromaDB Implementation
# ============================================

class ChromaDBProvider(VectorStoreProvider):
    """
    ChromaDB vector store provider.
    
    ChromaDB is an open-source embedding database that runs locally.
    It's perfect for development and smaller deployments.
    
    Features:
    - Persistent local storage
    - No external dependencies
    - Fast similarity search
    """
    
    def __init__(
        self,
        collection_name: str = None,
        persist_directory: str = None,
        embedding_model: str = None
    ):
        """
        Initialize ChromaDB provider.
        
        Args:
            collection_name: Name of the collection (like a table name)
            persist_directory: Where to store the database files
            embedding_model: OpenAI embedding model to use
        """
        # Load settings from environment with fallbacks
        self.collection_name = collection_name or os.getenv(
            "CHROMA_COLLECTION_NAME", 
            "clean_rag_docs"
        )
        self.persist_directory = persist_directory or os.getenv(
            "CHROMA_PERSIST_DIR", 
            "./chroma_db"
        )
        embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", 
            "text-embedding-3-small"
        )
        
        # Initialize OpenAI embeddings
        # This converts text into numerical vectors for similarity search
        self.embeddings = OpenAIEmbeddings(
            model=embedding_model,
            # Optionally reduce dimensions for faster search
            # dimensions=1536  # Uncomment to specify dimensions
        )
        
        # Initialize or load existing ChromaDB
        self._vectorstore = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        
        print(f"✅ ChromaDB initialized")
        print(f"   Collection: {self.collection_name}")
        print(f"   Directory: {self.persist_directory}")
    
    def get_vectorstore(self) -> VectorStore:
        """Return the Chroma vector store instance."""
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to ChromaDB.
        
        Documents are automatically embedded and stored.
        The embeddings are persisted to disk.
        
        Args:
            documents: List of Document objects to store
        """
        if not documents:
            print("⚠️ No documents to add")
            return
        
        print(f"📥 Adding {len(documents)} documents to ChromaDB...")
        
        # Add documents (they get embedded automatically)
        self._vectorstore.add_documents(documents)
        
        print(f"✅ Successfully added {len(documents)} documents")
    
    def get_retriever(
        self, 
        search_type: str = "similarity",
        k: int = 4,
        **kwargs
    ) -> BaseRetriever:
        """
        Get a retriever for the vector store.
        
        The retriever is used by LangChain chains to fetch
        relevant documents for a query.
        
        Args:
            search_type: "similarity", "mmr", or "similarity_score_threshold"
            k: Number of documents to retrieve
            **kwargs: Additional arguments for the retriever
            
        Returns:
            A LangChain retriever instance
        """
        return self._vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs={"k": k, **kwargs}
        )
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 4
    ) -> List[Document]:
        """
        Perform direct similarity search.
        
        This is useful for debugging or when you need
        more control than the retriever provides.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of most similar documents
        """
        return self._vectorstore.similarity_search(query, k=k)
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 4
    ) -> List[tuple]:
        """
        Search with relevance scores.
        
        Returns documents along with their similarity scores,
        useful for filtering by relevance threshold.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of (document, score) tuples
        """
        return self._vectorstore.similarity_search_with_score(query, k=k)
    
    def delete_collection(self) -> None:
        """
        Delete the entire collection.
        
        Use with caution - this removes all stored documents!
        """
        print(f"🗑️ Deleting collection: {self.collection_name}")
        self._vectorstore.delete_collection()
        print("✅ Collection deleted")
    
    def get_collection_stats(self) -> dict:
        """
        Get statistics about the collection.
        
        Returns:
            Dictionary with collection statistics
        """
        collection = self._vectorstore._collection
        return {
            "name": self.collection_name,
            "count": collection.count(),
            "persist_directory": self.persist_directory
        }


# ============================================
# Pinecone Implementation (Swappable)
# ============================================

class PineconeProvider(VectorStoreProvider):
    """
    Pinecone vector store provider.
    
    Pinecone is a managed vector database service.
    Use this for production deployments requiring:
    - Scalability
    - High availability
    - Managed infrastructure
    
    To use: Uncomment pinecone imports in requirements.txt
    """
    
    def __init__(
        self,
        index_name: str = None,
        embedding_model: str = None
    ):
        """
        Initialize Pinecone provider.
        
        Args:
            index_name: Name of the Pinecone index
            embedding_model: OpenAI embedding model to use
        """
        # Import Pinecone (only when needed)
        try:
            from langchain_pinecone import PineconeVectorStore
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "Pinecone not installed. Run: pip install pinecone-client langchain-pinecone"
            )
        
        # Load settings
        self.index_name = index_name or os.getenv(
            "PINECONE_INDEX_NAME", 
            "clean-rag-index"
        )
        api_key = os.getenv("PINECONE_API_KEY")
        
        if not api_key:
            raise ValueError("PINECONE_API_KEY not found in environment")
        
        embedding_model = embedding_model or os.getenv(
            "EMBEDDING_MODEL", 
            "text-embedding-3-small"
        )
        
        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_model)
        
        # Initialize Pinecone client
        pc = Pinecone(api_key=api_key)
        
        # Get or create index
        self.index = pc.Index(self.index_name)
        
        # Initialize vector store
        self._vectorstore = PineconeVectorStore(
            index=self.index,
            embedding=self.embeddings,
            text_key="text"
        )
        
        print(f"✅ Pinecone initialized: {self.index_name}")
    
    def get_vectorstore(self) -> VectorStore:
        """Return the Pinecone vector store instance."""
        return self._vectorstore
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to Pinecone."""
        if not documents:
            return
        
        print(f"📥 Adding {len(documents)} documents to Pinecone...")
        self._vectorstore.add_documents(documents)
        print(f"✅ Successfully added {len(documents)} documents")
    
    def get_retriever(self, k: int = 4, **kwargs) -> BaseRetriever:
        """Get a retriever for Pinecone."""
        return self._vectorstore.as_retriever(
            search_kwargs={"k": k, **kwargs}
        )
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Perform similarity search in Pinecone."""
        return self._vectorstore.similarity_search(query, k=k)


# ============================================
# Factory Function for Provider Selection
# ============================================

def get_vector_store_provider(
    provider: str = "chroma",
    **kwargs
) -> VectorStoreProvider:
    """
    Factory function to get the appropriate vector store provider.
    
    This makes it easy to swap between providers by changing
    a single parameter.
    
    Args:
        provider: "chroma" or "pinecone"
        **kwargs: Provider-specific configuration
        
    Returns:
        Initialized vector store provider
        
    Usage:
        # Use ChromaDB (default)
        store = get_vector_store_provider("chroma")
        
        # Switch to Pinecone
        store = get_vector_store_provider("pinecone")
    """
    providers = {
        "chroma": ChromaDBProvider,
        "pinecone": PineconeProvider,
    }
    
    if provider not in providers:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(providers.keys())}")
    
    return providers[provider](**kwargs)


# ============================================
# Convenience Class for RAG Retrieval
# ============================================

class RAGRetriever:
    """
    High-level retriever class for RAG applications.
    
    This wraps the vector store provider and adds
    RAG-specific functionality like:
    - Automatic reranking
    - Score thresholding
    - Context formatting
    """
    
    def __init__(
        self,
        provider: str = "chroma",
        top_k: int = 4,
        score_threshold: float = None,
        **provider_kwargs
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            provider: Vector store provider ("chroma" or "pinecone")
            top_k: Number of documents to retrieve
            score_threshold: Minimum similarity score (optional)
            **provider_kwargs: Arguments passed to the provider
        """
        self.provider = get_vector_store_provider(provider, **provider_kwargs)
        self.top_k = top_k
        self.score_threshold = score_threshold
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store."""
        self.provider.add_documents(documents)
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: User's question
            
        Returns:
            List of relevant documents
        """
        if self.score_threshold:
            # Use score-based filtering
            results = self.provider.similarity_search_with_score(
                query, 
                k=self.top_k
            )
            # Filter by score threshold
            documents = [
                doc for doc, score in results 
                if score >= self.score_threshold
            ]
        else:
            documents = self.provider.similarity_search(
                query, 
                k=self.top_k
            )
        
        return documents
    
    def get_retriever(self) -> BaseRetriever:
        """Get a LangChain retriever instance."""
        return self.provider.get_retriever(k=self.top_k)
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents as context string.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string with citations
        """
        if not documents:
            return "No relevant context found."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "Unknown")
            chunk_type = doc.metadata.get("chunk_type", "text")
            
            context_parts.append(
                f"[Source {i}: {source} ({chunk_type})]\n{doc.page_content}"
            )
        
        return "\n\n---\n\n".join(context_parts)


# ============================================
# Main block for testing
# ============================================

if __name__ == "__main__":
    # Test the retrieval system
    print("Testing ChromaDB Provider...")
    
    # Create provider
    provider = ChromaDBProvider()
    
    # Create test documents
    test_docs = [
        Document(
            page_content="Python is a programming language known for its simplicity.",
            metadata={"source": "test.pdf", "chunk_type": "text"}
        ),
        Document(
            page_content="Machine learning is a subset of artificial intelligence.",
            metadata={"source": "test.pdf", "chunk_type": "text"}
        ),
    ]
    
    # Add documents
    provider.add_documents(test_docs)
    
    # Test search
    results = provider.similarity_search("What is Python?", k=1)
    print(f"\nSearch results for 'What is Python?':")
    for doc in results:
        print(f"  - {doc.page_content[:100]}...")
    
    # Get stats
    stats = provider.get_collection_stats()
    print(f"\nCollection stats: {stats}")