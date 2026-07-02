import sys
sys.path.append('src')
from pdf_processor import process_pdf
from embedder import create_vectorstore, vectorstore_exists
from retriever import retrieve_relevant_chunks, format_context
from llm_handler import answer_question, summarize_text, generate_quiz, explain_topic

# ── Setup ─────────────────────────────────────────────────
print("=" * 50)
print("   AI STUDY COMPANION — STEP 6 TEST")
print("=" * 50)

pdf_path = "data/uploads/test.pdf"
pdf_name = "test.pdf"

# ── Process & Embed PDF if not already done ───────────────
if not vectorstore_exists(pdf_name):
    chunks, _ = process_pdf(pdf_path)
    create_vectorstore(chunks, pdf_name)
else:
    print("✅ Vector store already exists!")

# ── Test Q&A ──────────────────────────────────────────────
print("\n── Test 1: Q&A ──")
question = "What is this document about?"
chunks   = retrieve_relevant_chunks(question, pdf_name, k=2)
context  = format_context(chunks)
answer   = answer_question(question, context)
print(f"❓ Question: {question}")
print(f"💡 Answer: {answer}")

# ── Test Summary ──────────────────────────────────────────
print("\n── Test 2: Summary ──")
all_chunks = retrieve_relevant_chunks("summarize everything", pdf_name, k=4)
all_context = format_context(all_chunks)
summary = summarize_text(all_context)
print(f"📝 Summary:\n{summary}")

print("\n🎉 LLM Handler Working!")
print("=" * 50)