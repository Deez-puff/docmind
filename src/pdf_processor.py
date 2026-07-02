import fitz  # PyMuPDF
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
def extract_text_from_pdf(pdf_path):
    """
    Extracts raw text from a PDF file using PyMuPDF.
    
    Why PyMuPDF?
    - Handles scanned PDFs, complex layouts, tables
    - Much faster and more accurate than other PDF libraries
    - Returns text page by page
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        full_text += page.get_text()
    
    doc.close()
    return full_text


def chunk_text(text, chunk_size=1000, chunk_overlap=200):
    """
    Splits large text into smaller overlapping chunks.
    
    Why chunk?
    - LLMs have a token limit (can't read entire PDFs at once)
    - Smaller chunks = more precise retrieval
    
    Why overlap?
    - Overlap of 200 chars ensures no important info
      is cut off at chunk boundaries
    
    RecursiveCharacterTextSplitter tries to split at:
    1. Paragraphs first
    2. Then sentences
    3. Then words
    4. Then characters (last resort)
    This keeps chunks semantically meaningful.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    
    chunks = splitter.split_text(text)
    return chunks


def process_pdf(pdf_path):
    """
    Full pipeline: PDF → extracted text → chunks
    Returns list of text chunks ready for embedding.
    """
    print(f"📄 Processing: {os.path.basename(pdf_path)}")
    
    # ── Extract text ──────────────────────────────────────
    text = extract_text_from_pdf(pdf_path)
    print(f"✅ Extracted {len(text)} characters")
    
    # ── Chunk text ────────────────────────────────────────
    chunks = chunk_text(text)
    print(f"✅ Split into {len(chunks)} chunks")
    
    return chunks, text