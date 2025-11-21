"""
ingestion.py - Document Ingestion Module
=========================================
This module handles the heavy lifting of PDF processing:
1. Loading PDFs using the 'unstructured' library
2. Detecting and separating tables from regular text
3. Generating text summaries of tables for LLM understanding
4. Chunking text using RecursiveCharacterTextSplitter

Author: Senior AI Engineer
"""

import os
from typing import List, Dict, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# Unstructured library for PDF parsing
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Table, Text, Title, NarrativeText

# LangChain for text splitting
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# For table formatting
from tabulate import tabulate
import pandas as pd

# OpenAI for table summarization
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()


@dataclass
class ProcessedDocument:
    """
    Data class to hold processed document information.
    This makes it easy to pass around document data between functions.
    """
    text_chunks: List[Document]  # Regular text split into chunks
    table_chunks: List[Document]  # Table summaries as chunks
    metadata: Dict  # Document-level metadata


class PDFIngestionPipeline:
    """
    Main class for ingesting and processing PDF documents.
    
    This pipeline:
    1. Extracts all elements from a PDF (text, tables, titles, etc.)
    2. Identifies tables and generates LLM-friendly summaries
    3. Chunks all content for vector storage
    """
    
    def __init__(
        self,
        chunk_size: int = None,
        chunk_overlap: int = None,
        llm_model: str = None
    ):
        """
        Initialize the ingestion pipeline.
        
        Args:
            chunk_size: Size of text chunks (default from env or 1000)
            chunk_overlap: Overlap between chunks (default from env or 200)
            llm_model: OpenAI model for table summarization
        """
        # Load settings from environment with fallbacks
        self.chunk_size = chunk_size or int(os.getenv("CHUNK_SIZE", 1000))
        self.chunk_overlap = chunk_overlap or int(os.getenv("CHUNK_OVERLAP", 200))
        self.llm_model = llm_model or os.getenv("LLM_MODEL", "gpt-4o-mini")
        
        # Initialize the text splitter with RecursiveCharacterTextSplitter
        # This splitter tries to keep semantically related text together
        # by splitting on paragraphs, then sentences, then words
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
        )
        
        # Initialize LLM for table summarization
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0  # Use 0 for deterministic summaries
        )
    
    def load_pdf(self, pdf_path: str) -> List:
        """
        Load a PDF and extract all elements using unstructured.
        
        The 'unstructured' library automatically detects different
        element types: tables, text, titles, lists, etc.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of extracted elements
        """
        print(f"📄 Loading PDF: {pdf_path}")
        
        # partition_pdf extracts structured elements from the PDF
        # strategy="hi_res" provides better table detection
        # infer_table_structure=True extracts table structure as HTML
        elements = partition_pdf(
            filename=pdf_path,
            strategy="hi_res",  # High-resolution extraction
            infer_table_structure=True,  # Extract table structure
            include_metadata=True  # Include page numbers, etc.
        )
        
        print(f"✅ Extracted {len(elements)} elements from PDF")
        return elements
    
    def separate_tables_and_text(
        self, 
        elements: List
    ) -> Tuple[List[Table], List[str]]:
        """
        Separate table elements from text elements.
        
        This is crucial because tables need special handling:
        - Tables are converted to summaries for LLM understanding
        - Text is chunked normally
        
        Args:
            elements: List of elements from partition_pdf
            
        Returns:
            Tuple of (table_elements, text_strings)
        """
        tables = []
        texts = []
        
        for element in elements:
            # Check if this element is a Table
            if isinstance(element, Table):
                tables.append(element)
                print(f"📊 Found table on page {element.metadata.page_number if hasattr(element.metadata, 'page_number') else 'unknown'}")
            else:
                # All other elements (Title, NarrativeText, etc.) are treated as text
                text_content = str(element)
                if text_content.strip():  # Only add non-empty text
                    texts.append(text_content)
        
        print(f"📊 Total tables found: {len(tables)}")
        print(f"📝 Total text blocks found: {len(texts)}")
        
        return tables, texts
    
    def generate_table_summary(self, table: Table) -> str:
        """
        Generate a text summary of a table for LLM understanding.
        
        Tables in their raw form are hard for LLMs to understand.
        This function converts them to a readable text summary
        that captures the key information.
        
        Args:
            table: Table element from unstructured
            
        Returns:
            Text summary of the table
        """
        # Get the table content (usually HTML or text representation)
        table_text = str(table)
        
        # Get metadata for context
        page_num = getattr(table.metadata, 'page_number', 'unknown')
        
        # Create a prompt for table summarization
        summary_prompt = f"""You are a data analyst. Analyze the following table and provide a clear, 
comprehensive text summary that captures ALL the information in the table.

Include:
1. What the table is about (the topic/title if apparent)
2. The column headers and what they represent
3. Key data points and values
4. Any notable patterns, totals, or important figures
5. The relationship between different columns/rows

Table content:
{table_text}

Provide a detailed summary that would allow someone to understand all the data 
without seeing the original table. Be specific with numbers and values."""

        try:
            # Use LLM to generate summary
            response = self.llm.invoke(summary_prompt)
            summary = response.content
            
            # Add metadata to the summary
            final_summary = f"[TABLE SUMMARY - Page {page_num}]\n{summary}\n[END TABLE SUMMARY]"
            
            return final_summary
            
        except Exception as e:
            # Fallback: return raw table with metadata
            print(f"⚠️ Table summarization failed: {e}")
            return f"[TABLE - Page {page_num}]\n{table_text}\n[END TABLE]"
    
    def chunk_text(
        self, 
        texts: List[str], 
        source: str
    ) -> List[Document]:
        """
        Chunk text content using RecursiveCharacterTextSplitter.
        
        This creates overlapping chunks that maintain context
        while fitting within token limits for embedding.
        
        Args:
            texts: List of text strings to chunk
            source: Source identifier for metadata
            
        Returns:
            List of Document objects with chunks
        """
        # Combine all texts with double newlines for clear separation
        combined_text = "\n\n".join(texts)
        
        # Create chunks using the splitter
        chunks = self.text_splitter.split_text(combined_text)
        
        # Convert to Document objects with metadata
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_id": i,
                    "chunk_type": "text",
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)
        
        print(f"✂️ Created {len(documents)} text chunks")
        return documents
    
    def process_tables(
        self, 
        tables: List[Table], 
        source: str
    ) -> List[Document]:
        """
        Process tables into summarized Document chunks.
        
        Each table gets its own Document with a text summary
        that the LLM can understand and reference.
        
        Args:
            tables: List of Table elements
            source: Source identifier for metadata
            
        Returns:
            List of Document objects with table summaries
        """
        documents = []
        
        for i, table in enumerate(tables):
            print(f"📊 Summarizing table {i+1}/{len(tables)}...")
            
            # Generate text summary of the table
            summary = self.generate_table_summary(table)
            
            # Get page number if available
            page_num = getattr(table.metadata, 'page_number', 'unknown')
            
            # Create Document with table summary
            doc = Document(
                page_content=summary,
                metadata={
                    "source": source,
                    "chunk_id": f"table_{i}",
                    "chunk_type": "table",
                    "page_number": page_num,
                    "table_index": i
                }
            )
            documents.append(doc)
        
        print(f"✅ Processed {len(documents)} tables")
        return documents
    
    def ingest(self, pdf_path: str) -> ProcessedDocument:
        """
        Main entry point: Process a PDF and return chunked documents.
        
        This orchestrates the entire ingestion pipeline:
        1. Load PDF
        2. Separate tables and text
        3. Summarize tables
        4. Chunk all content
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ProcessedDocument with all chunks and metadata
        """
        # Extract filename for source tracking
        source = os.path.basename(pdf_path)
        
        # Step 1: Load PDF and extract elements
        elements = self.load_pdf(pdf_path)
        
        # Step 2: Separate tables from text
        tables, texts = self.separate_tables_and_text(elements)
        
        # Step 3: Process tables into summaries
        table_chunks = self.process_tables(tables, source)
        
        # Step 4: Chunk text content
        text_chunks = self.chunk_text(texts, source)
        
        # Combine all chunks
        all_chunks = text_chunks + table_chunks
        
        print(f"\n📦 Ingestion complete!")
        print(f"   - Text chunks: {len(text_chunks)}")
        print(f"   - Table chunks: {len(table_chunks)}")
        print(f"   - Total chunks: {len(all_chunks)}")
        
        return ProcessedDocument(
            text_chunks=text_chunks,
            table_chunks=table_chunks,
            metadata={
                "source": source,
                "total_elements": len(elements),
                "total_tables": len(tables),
                "total_chunks": len(all_chunks)
            }
        )


# ============================================
# Convenience function for quick ingestion
# ============================================

def ingest_pdf(pdf_path: str) -> List[Document]:
    """
    Quick function to ingest a PDF and return all chunks.
    
    Usage:
        chunks = ingest_pdf("my_document.pdf")
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        List of all Document chunks (text + tables)
    """
    pipeline = PDFIngestionPipeline()
    result = pipeline.ingest(pdf_path)
    return result.text_chunks + result.table_chunks


# ============================================
# Main block for testing
# ============================================

if __name__ == "__main__":
    # Test the ingestion pipeline
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python ingestion.py <path_to_pdf>")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    chunks = ingest_pdf(pdf_path)
    
    print("\n" + "="*50)
    print("Sample chunks:")
    print("="*50)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i} ({chunk.metadata.get('chunk_type', 'unknown')}) ---")
        print(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)