"""
app.py - Streamlit Chat Interface
==================================
A simple but functional web UI for chatting with your documents.

Features:
- PDF upload and processing
- Real-time chat interface
- Source citation display
- Conversation history

Run with: streamlit run app.py

Author: Senior AI Engineer
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Import our modules
from src.ingestion import PDFIngestionPipeline, ingest_pdf
from src.retrieval import RAGRetriever
from src.chain import RAGChain, ConversationalRAGChain, create_rag_chain

# Load environment variables
load_dotenv()


# ============================================
# Page Configuration
# ============================================

st.set_page_config(
    page_title="Clean RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================
# Custom CSS for better UI
# ============================================

st.markdown("""
<style>
    .stAlert {
        margin-top: 1rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 0.5rem 0;
        font-size: 0.9rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #e3f2fd;
    }
    .assistant-message {
        background-color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)


# ============================================
# Session State Initialization
# ============================================

def init_session_state():
    """
    Initialize Streamlit session state variables.
    
    Session state persists across reruns within a session,
    allowing us to maintain conversation history and loaded documents.
    """
    # Conversation history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # RAG chain instance
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    
    # Document processing status
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    
    # Retriever instance
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    # Processing status
    if "processing" not in st.session_state:
        st.session_state.processing = False


# ============================================
# Sidebar: Document Upload & Settings
# ============================================

def render_sidebar():
    """
    Render the sidebar with upload and settings options.
    """
    with st.sidebar:
        st.header("📁 Document Upload")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help="Upload a PDF to chat with its contents"
        )
        
        # Settings expander
        with st.expander("⚙️ Settings", expanded=False):
            # Number of documents to retrieve
            top_k = st.slider(
                "Documents to retrieve (top-k)",
                min_value=1,
                max_value=10,
                value=4,
                help="Number of relevant chunks to retrieve for each question"
            )
            
            # Temperature setting
            temperature = st.slider(
                "LLM Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.1,
                help="Lower = more deterministic, Higher = more creative"
            )
            
            # Conversational mode toggle
            conversational = st.checkbox(
                "Enable conversation history",
                value=True,
                help="Keep context from previous questions"
            )
            
            st.session_state.settings = {
                "top_k": top_k,
                "temperature": temperature,
                "conversational": conversational
            }
        
        # Process button
        if uploaded_file is not None:
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                process_document(uploaded_file)
        
        # Status indicator
        st.divider()
        if st.session_state.documents_loaded:
            st.success("✅ Documents loaded and ready!")
            
            # Clear chat button
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if hasattr(st.session_state.rag_chain, 'clear_history'):
                    st.session_state.rag_chain.clear_history()
                st.rerun()
            
            # Reset documents button
            if st.button("📄 Load New Document", use_container_width=True):
                st.session_state.documents_loaded = False
                st.session_state.messages = []
                st.session_state.rag_chain = None
                st.rerun()
        else:
            st.info("👆 Upload a PDF to get started")
        
        # Info section
        st.divider()
        st.caption("Built with LangChain + ChromaDB")
        st.caption("Model: GPT-4o-mini")


# ============================================
# Document Processing
# ============================================

def process_document(uploaded_file):
    """
    Process the uploaded PDF document.
    
    This function:
    1. Saves the uploaded file temporarily
    2. Runs the ingestion pipeline
    3. Stores chunks in the vector database
    4. Initializes the RAG chain
    """
    st.session_state.processing = True
    
    # Create a progress container
    progress_container = st.sidebar.empty()
    
    with progress_container.container():
        st.info("Processing document...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Save uploaded file to temp location
            status_text.text("📄 Saving file...")
            progress_bar.progress(10)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
            
            # Step 2: Run ingestion pipeline
            status_text.text("🔍 Extracting content (this may take a while)...")
            progress_bar.progress(30)
            
            pipeline = PDFIngestionPipeline()
            result = pipeline.ingest(tmp_path)
            
            # Step 3: Combine all chunks
            status_text.text("✂️ Processing chunks...")
            progress_bar.progress(60)
            
            all_chunks = result.text_chunks + result.table_chunks
            
            # Step 4: Initialize retriever and add documents
            status_text.text("💾 Storing in vector database...")
            progress_bar.progress(80)
            
            # Get settings
            settings = st.session_state.get("settings", {})
            top_k = settings.get("top_k", 4)
            temperature = settings.get("temperature", 0.0)
            conversational = settings.get("conversational", True)
            
            # Create retriever
            retriever = RAGRetriever(
                provider="chroma",
                top_k=top_k
            )
            retriever.add_documents(all_chunks)
            st.session_state.retriever = retriever
            
            # Step 5: Initialize RAG chain
            status_text.text("🤖 Initializing RAG chain...")
            progress_bar.progress(90)
            
            st.session_state.rag_chain = create_rag_chain(
                retriever=retriever,
                conversational=conversational,
                temperature=temperature,
                top_k=top_k
            )
            
            # Done!
            progress_bar.progress(100)
            status_text.text("✅ Complete!")
            
            st.session_state.documents_loaded = True
            st.session_state.processing = False
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Show success message
            st.sidebar.success(
                f"Processed {result.metadata['total_chunks']} chunks "
                f"({result.metadata['total_tables']} tables)"
            )
            
            # Rerun to update UI
            st.rerun()
            
        except Exception as e:
            st.session_state.processing = False
            st.sidebar.error(f"Error processing document: {str(e)}")
            raise e


# ============================================
# Chat Interface
# ============================================

def render_chat():
    """
    Render the main chat interface.
    """
    st.header("💬 Chat with your Document")
    
    # Check if documents are loaded
    if not st.session_state.documents_loaded:
        st.info(
            "👈 Upload a PDF document in the sidebar to start chatting. "
            "The system will extract text and tables, then allow you to "
            "ask questions about the content."
        )
        
        # Show example questions
        st.subheader("Example Questions")
        st.markdown("""
        Once you upload a document, you can ask questions like:
        - "What is this document about?"
        - "Summarize the key points"
        - "What data is shown in the tables?"
        - "Explain section X in detail"
        """)
        return
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show sources for assistant messages
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 View Sources", expanded=False):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}** ({source['type']})")
                        st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                        st.divider()
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Invoke RAG chain
                    response = st.session_state.rag_chain.invoke(prompt)
                    
                    # Display answer
                    st.markdown(response.answer)
                    
                    # Prepare sources for storage
                    sources = [
                        {
                            "content": doc.page_content,
                            "type": doc.metadata.get("chunk_type", "text"),
                            "source": doc.metadata.get("source", "Unknown")
                        }
                        for doc in response.sources
                    ]
                    
                    # Show sources
                    with st.expander("📚 View Sources", expanded=False):
                        for i, source in enumerate(sources, 1):
                            st.markdown(f"**Source {i}** ({source['type']})")
                            st.text(source["content"][:500] + "..." if len(source["content"]) > 500 else source["content"])
                            st.divider()
                    
                    # Add assistant message to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response.answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")


# ============================================
# Main App
# ============================================

def main():
    """
    Main application entry point.
    """
    # Initialize session state
    init_session_state()
    
    # App title
    st.title("📚 Clean RAG Document Chat")
    st.caption("Upload PDFs, extract tables & text, and chat with your documents")
    
    # Render sidebar
    render_sidebar()
    
    # Render chat interface
    render_chat()


# ============================================
# Entry Point
# ============================================

if __name__ == "__main__":
    main()