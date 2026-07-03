# DocMind — AI Study Companion

![Python](https://img.shields.io/badge/Python-3.14-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-RAG-orange?style=for-the-badge)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-green?style=for-the-badge)
![Groq](https://img.shields.io/badge/Groq-Llama3.3-purple?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-red?style=for-the-badge&logo=streamlit)
![GitHub](https://img.shields.io/badge/GitHub-Public-black?style=for-the-badge&logo=github)

> An AI-powered study companion that lets you upload any PDF and instantly get answers, summaries, quizzes, and topic explanations powered by RAG, ChromaDB, and Groq's Llama 3.3 70B.

---
## Table of Contents
- [About the Project](#about-the-project)
- [How It Works](#how-it-works)
- [Key Features](#key-features)
- [Important Limitation](#important-limitation)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [What is RAG](#what-is-rag)
- [Future Improvements](#future-improvements)

---

## About the Project

Students and researchers often struggle with large PDFs — textbooks, research papers, and lecture notes. Reading everything manually is slow and inefficient. DocMind solves this by building an AI system that:

- Reads and understands any text-based PDF document
- Answers specific questions accurately using only the document's content
- Generates structured summaries for quick revision
- Creates multiple choice quizzes to test understanding
- Explains difficult topics in simple, beginner-friendly language at adjustable difficulty levels

This project demonstrates a real-world RAG (Retrieval Augmented Generation) pipeline — the same architecture used in enterprise AI tools, ChatGPT plugins, and document intelligence platforms.

---

## How It Works
PDF Upload
|
v
Extract raw text using PyMuPDF
|
v
Split text into overlapping chunks (LangChain TextSplitter)
|
v
Convert each chunk into a vector (sentence-transformers)
|
v
Store all vectors in ChromaDB (local vector database)
|
v
User asks a question or requests a feature
|
v
Convert question to vector, find most similar chunks
|
v
Send relevant chunks + question to Groq (Llama 3.3 70B)
|
v
Return accurate, context-grounded answer to user

---

## Key Features

| Feature | Description |
|---|---|
| PDF Upload | Upload any text-based PDF directly in the browser |
| Q&A Chat | Ask questions and get accurate answers grounded in the PDF |
| Summarization | Generate structured summaries with main topic, key points, and conclusions |
| Quiz Generator | Create multiple choice questions with answers and explanations |
| Topic Explainer | Get difficult topics explained at Elementary, High School, Undergraduate, or Expert level |
| Download Results | Download summaries, quizzes, and explanations as text files |
| Persistent Vector Store | ChromaDB saves embeddings to disk — no re-processing on repeat visits |
| Chat History | Q&A tab maintains full conversation history within the session |

---

## Important Limitation

DocMind can only process PDFs that contain actual selectable text. It cannot read or extract content from:

- Scanned PDFs (image-based documents where text is a photograph)
- PDFs where text appears as embedded images
- Handwritten documents converted to PDF

To check if your PDF is compatible, open it and try selecting and copying text. If you can highlight and copy the text normally, DocMind will work with it. If the text cannot be selected, the PDF is image-based and will not be processed correctly.

This is a known limitation of PyMuPDF's text extraction approach. Future versions may include OCR (Optical Character Recognition) support to handle scanned documents.

---

## Project Structure
docmind/
|
|-- data/
|   |-- uploads/          <- uploaded PDFs stored here (gitignored)
|   |-- vectorstore/      <- ChromaDB vector database (gitignored)
|
|-- src/
|   |-- pdf_processor.py  <- extract and chunk PDF text
|   |-- embedder.py       <- convert text to vectors, manage ChromaDB
|   |-- retriever.py      <- search relevant chunks for a query
|   |-- llm_handler.py    <- Groq API calls for all 4 features
|
|-- app.py                <- Streamlit web application
|-- main.py               <- terminal-based runner for testing
|-- requirements.txt      <- Python dependencies
|-- .env                  <- API key (never pushed to GitHub)
|-- .gitignore            <- excludes .env, uploads, vectorstore
|-- README.md             <- project documentation

---

## Tech Stack

| Category | Tools Used |
|---|---|
| Language | Python 3.14 |
| RAG Framework | LangChain, LangChain Community, LangChain Core |
| LLM | Groq API (Llama 3.3 70B) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2, runs locally) |
| Vector Database | ChromaDB (local, persistent) |
| PDF Processing | PyMuPDF (fitz) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Web App | Streamlit |
| Environment | python-dotenv |

---

## Setup and Installation

### 1 - Clone the Repository
git clone https://github.com/Deez-puff/docmind.git
cd docmind

### 2 - Install Dependencies
pip install -r requirements.txt

### 3 - Get a Free Groq API Key
1. Go to https://console.groq.com
2. Sign up with Google or email (no credit card required)
3. Click API Keys in the left sidebar
4. Click Create API Key
5. Copy the key (starts with gsk_)

### 4 - Create a .env File
Create a file named .env in the root folder and add:
GROQ_API_KEY=gsk_your_key_here

### 5 - Create Required Folders
mkdir data/uploads
mkdir data/vectorstore

---

## How to Run

### Web App (Recommended)
python -m streamlit run app.py
Opens automatically in your browser at http://localhost:8501

### Terminal (for testing)
python main.py

---

## What is RAG

RAG stands for Retrieval Augmented Generation. It is the core architecture powering DocMind.

The problem with sending an entire PDF to an LLM is that PDFs can be hundreds of pages long, and LLMs have a limit on how much text they can process at once. Even if they could process the whole document, it would be slow and expensive.

RAG solves this by:

1. Splitting the document into small chunks
2. Converting each chunk into a numerical vector representing its meaning
3. Storing all vectors in a vector database (ChromaDB)
4. When a question is asked, converting the question into a vector
5. Finding the chunks with the most similar vectors (most relevant to the question)
6. Sending only those relevant chunks to the LLM along with the question
7. The LLM reads only the relevant parts and gives a precise answer

This approach is fast, accurate, cost-efficient, and prevents the LLM from making up information (hallucination) since it is grounded in the actual document content.

RAG is used in Perplexity AI, ChatGPT plugins, enterprise document tools, legal research platforms, and most modern AI assistants that work with private documents.

---

## Future Improvements

- [ ] Add OCR support for scanned and image-based PDFs
- [ ] Support multiple PDF uploads simultaneously
- [ ] Add support for Word documents and PowerPoint files
- [ ] Implement conversation memory for multi-turn Q&A
- [ ] Add a flashcard generator feature
- [ ] Support for highlighting and annotating source chunks in answers
- [ ] Deploy on Streamlit Cloud for public access
- [ ] Add multilingual support for non-English PDFs
- [ ] Implement a study progress tracker

---

## Author

Deepak Rajesh

Built as part of a personal AI engineering project exploring RAG, vector databases, and LLM integration.

---

## License

This project is open source and available under the MIT License.
