from embedder import load_vectorstore, get_embeddings

def get_retriever(pdf_name, k=4, persist_dir="data/vectorstore"):
    """
    Creates a retriever from an existing vector store.
    
    What is a retriever?
    - It's a search engine over your PDF chunks
    - Given a question, it finds the k most relevant chunks
    - Uses cosine similarity between question vector and chunk vectors
    
    k=4 means return top 4 most relevant chunks
    More chunks = more context but slower/more expensive LLM call
    """
    vectorstore = load_vectorstore(pdf_name, persist_dir)
    retriever   = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    return retriever


def retrieve_relevant_chunks(question, pdf_name, k=4):
    """
    Main retrieval function:
    1. Loads vector store for the given PDF
    2. Converts question to vector
    3. Finds k most similar chunks
    4. Returns them as a list of strings
    
    Why return strings?
    - LangChain Documents have metadata we don't need
    - LLM just needs the raw text content
    """
    retriever = get_retriever(pdf_name, k)
    docs      = retriever.invoke(question)

    # ── Extract just the text content ─────────────────────
    chunks = [doc.page_content for doc in docs]

    return chunks


def format_context(chunks):
    """
    Joins retrieved chunks into a single context string
    to pass to the LLM.
    
    Each chunk is separated by a divider for clarity.
    """
    context = "\n\n---\n\n".join(chunks)
    return context