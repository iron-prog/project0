"""
chain.py - RAG Chain Module
============================
This module implements the RAG logic:
1. Takes user question + retrieved context
2. Generates answer with strict citation requirements
3. Prevents hallucination through careful prompting

Author: Senior AI Engineer
"""

import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# LangChain components
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.schema.output_parser import StrOutputParser

# Import our retrieval module
from src.retrieval import RAGRetriever, get_vector_store_provider

# Load environment variables
load_dotenv()


# ============================================
# System Prompts
# ============================================

# This prompt is CRITICAL for preventing hallucination
# It strictly instructs the LLM to only use provided context
SYSTEM_PROMPT = """You are a helpful assistant that answers questions based ONLY on the provided context.

STRICT RULES YOU MUST FOLLOW:
1. Answer ONLY based on the provided context. Do not use any external knowledge.
2. If the answer is not in the context, say "I do not know based on the provided documents."
3. Do NOT hallucinate or make up information.
4. Always cite your sources by referencing [Source X] when using information from the context.
5. If multiple sources contain relevant information, cite all of them.
6. Be concise but complete in your answers.
7. If the context contains tables, reference the table data specifically.

Remember: It's better to say "I don't know" than to provide incorrect information."""

# Alternative prompt for more conversational responses
CONVERSATIONAL_PROMPT = """You are a knowledgeable assistant helping users understand documents.

Guidelines:
- Base your answers STRICTLY on the provided context
- If information isn't in the context, clearly state: "I do not know based on the provided documents."
- Cite sources as [Source X] when referencing specific information
- Never invent or assume information not present in the context
- For tables, describe the relevant data points
- Be helpful and conversational while maintaining accuracy"""


# ============================================
# Data Classes for Structured Output
# ============================================

@dataclass
class RAGResponse:
    """
    Structured response from the RAG chain.
    
    Contains the answer along with metadata for
    evaluation and debugging.
    """
    answer: str  # The generated answer
    sources: List[Document]  # Retrieved source documents
    query: str  # Original user query
    context: str  # Formatted context used


# ============================================
# RAG Chain Implementation
# ============================================

class RAGChain:
    """
    Main RAG chain class that orchestrates:
    1. Document retrieval
    2. Context formatting
    3. Answer generation with citations
    
    This is the core logic of the RAG system.
    """
    
    def __init__(
        self,
        retriever: RAGRetriever = None,
        llm_model: str = None,
        temperature: float = 0,
        top_k: int = 4,
        system_prompt: str = None,
        vector_store_provider: str = "chroma"
    ):
        """
        Initialize the RAG chain.
        
        Args:
            retriever: Pre-configured RAGRetriever (optional)
            llm_model: OpenAI model name (default: gpt-4o-mini)
            temperature: LLM temperature (0 for deterministic)
            top_k: Number of documents to retrieve
            system_prompt: Custom system prompt (optional)
            vector_store_provider: "chroma" or "pinecone"
        """
        # Load model from environment or use default
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        self.temperature = temperature
        self.top_k = top_k
        
        # Initialize LLM
        # Using temperature=0 for more deterministic, faithful answers
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=self.temperature
        )
        
        # Initialize retriever if not provided
        if retriever:
            self.retriever = retriever
        else:
            self.retriever = RAGRetriever(
                provider=vector_store_provider,
                top_k=top_k
            )
        
        # Set system prompt
        self.system_prompt = system_prompt or SYSTEM_PROMPT
        
        # Build the prompt template
        self.prompt = self._build_prompt()
        
        # Build the chain
        self.chain = self._build_chain()
        
        print(f"✅ RAG Chain initialized")
        print(f"   Model: {self.llm_model}")
        print(f"   Temperature: {self.temperature}")
        print(f"   Top-K: {self.top_k}")
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """
        Build the prompt template for the RAG chain.
        
        The template includes:
        - System message with strict instructions
        - Context from retrieved documents
        - User's question
        """
        template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt),
            ("human", """Context from documents:
{context}

---

User Question: {question}

Please answer the question based ONLY on the context above. Remember to cite your sources.""")
        ])
        
        return template
    
    def _format_documents(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a context string.
        
        Each document is labeled with a source number for citation.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        if not documents:
            return "No relevant documents found."
        
        formatted_parts = []
        
        for i, doc in enumerate(documents, 1):
            # Extract metadata
            source = doc.metadata.get("source", "Unknown")
            chunk_type = doc.metadata.get("chunk_type", "text")
            page = doc.metadata.get("page_number", "N/A")
            
            # Format the document with clear labeling
            header = f"[Source {i}] (File: {source}, Type: {chunk_type}, Page: {page})"
            content = doc.page_content
            
            formatted_parts.append(f"{header}\n{content}")
        
        return "\n\n" + "="*50 + "\n\n".join(formatted_parts)
    
    def _build_chain(self):
        """
        Build the LangChain LCEL chain.
        
        This creates a pipeline that:
        1. Takes a question
        2. Retrieves relevant documents
        3. Formats context
        4. Generates answer
        """
        # The chain uses LCEL (LangChain Expression Language)
        chain = (
            self.prompt 
            | self.llm 
            | StrOutputParser()
        )
        
        return chain
    
    def invoke(self, question: str) -> RAGResponse:
        """
        Process a question and generate an answer.
        
        This is the main entry point for asking questions.
        
        Args:
            question: User's question
            
        Returns:
            RAGResponse with answer, sources, and metadata
        """
        # Step 1: Retrieve relevant documents
        print(f"🔍 Retrieving documents for: {question[:50]}...")
        documents = self.retriever.retrieve(question)
        print(f"📚 Retrieved {len(documents)} documents")
        
        # Step 2: Format context
        context = self._format_documents(documents)
        
        # Step 3: Generate answer
        print("🤖 Generating answer...")
        answer = self.chain.invoke({
            "context": context,
            "question": question
        })
        
        # Step 4: Return structured response
        return RAGResponse(
            answer=answer,
            sources=documents,
            query=question,
            context=context
        )
    
    def stream(self, question: str):
        """
        Stream the answer token by token.
        
        Useful for real-time UI updates.
        
        Args:
            question: User's question
            
        Yields:
            Answer tokens as they're generated
        """
        # Retrieve documents first
        documents = self.retriever.retrieve(question)
        context = self._format_documents(documents)
        
        # Stream the response
        for chunk in self.chain.stream({
            "context": context,
            "question": question
        }):
            yield chunk
    
    def add_documents(self, documents: List[Document]) -> None:
        """
        Add documents to the underlying vector store.
        
        Args:
            documents: List of Document objects to add
        """
        self.retriever.add_documents(documents)


# ============================================
# Conversation Chain with Memory
# ============================================

class ConversationalRAGChain(RAGChain):
    """
    RAG chain with conversation history support.
    
    This extends the basic RAG chain to maintain context
    across multiple turns of conversation.
    """
    
    def __init__(self, **kwargs):
        """Initialize with conversation history tracking."""
        super().__init__(**kwargs)
        self.conversation_history: List[Dict[str, str]] = []
    
    def _build_prompt(self) -> ChatPromptTemplate:
        """
        Build prompt with conversation history support.
        """
        template = ChatPromptTemplate.from_messages([
            ("system", self.system_prompt + "\n\nYou may also reference the conversation history for context about what the user is asking."),
            MessagesPlaceholder(variable_name="history"),
            ("human", """Context from documents:
{context}

---

Current Question: {question}

Answer based on the context and conversation history.""")
        ])
        
        return template
    
    def invoke(self, question: str) -> RAGResponse:
        """
        Process question with conversation history.
        """
        # Retrieve documents
        documents = self.retriever.retrieve(question)
        context = self._format_documents(documents)
        
        # Build history messages
        history_messages = []
        for msg in self.conversation_history[-6:]:  # Last 3 turns
            if msg["role"] == "user":
                history_messages.append(HumanMessage(content=msg["content"]))
            else:
                history_messages.append(AIMessage(content=msg["content"]))
        
        # Generate answer
        answer = self.chain.invoke({
            "context": context,
            "question": question,
            "history": history_messages
        })
        
        # Update history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})
        
        return RAGResponse(
            answer=answer,
            sources=documents,
            query=question,
            context=context
        )
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        print("🗑️ Conversation history cleared")


# ============================================
# Factory function
# ============================================

def create_rag_chain(
    conversational: bool = False,
    **kwargs
) -> RAGChain:
    """
    Factory function to create RAG chains.
    
    Args:
        conversational: Whether to use conversation history
        **kwargs: Arguments passed to the chain constructor
        
    Returns:
        Configured RAG chain instance
    """
    if conversational:
        return ConversationalRAGChain(**kwargs)
    return RAGChain(**kwargs)


# ============================================
# Main block for testing
# ============================================

if __name__ == "__main__":
    print("Testing RAG Chain...")
    
    # Create chain
    chain = create_rag_chain()
    
    # Test with a sample question (assumes documents are already indexed)
    test_question = "What is this document about?"
    
    try:
        response = chain.invoke(test_question)
        print(f"\nQuestion: {test_question}")
        print(f"\nAnswer: {response.answer}")
        print(f"\nSources used: {len(response.sources)}")
    except Exception as e:
        print(f"Error (expected if no documents indexed): {e}")