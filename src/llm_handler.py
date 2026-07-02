import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

# ── Load API key ──────────────────────────────────────────
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("❌ GROQ_API_KEY not found in .env file!")

print("✅ Groq API configured!")

# ── Model to use ──────────────────────────────────────────
# llama-3.3-70b-versatile = powerful, free, fast
MODEL_NAME = "llama-3.3-70b-versatile"


def get_llm():
    """
    Returns a configured Groq LLM instance.

    Why Groq + Llama 3.3 70B?
    - Completely free tier
    - Very fast (Groq has custom AI chips)
    - Llama 3.3 70B is extremely capable
    - Perfect for RAG applications
    """
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=MODEL_NAME,
        temperature=0.3    # low temperature = more factual answers
    )
    return llm


def answer_question(question, context):
    """
    Answers a question using retrieved PDF context.
    Prevents hallucination by instructing the model
    to only use the provided context.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content="""You are a helpful AI study assistant.
Answer questions based ONLY on the provided context from a PDF document.
If the answer is not in the context, say "I couldn't find this in the document."
Do NOT make up information that is not in the context.
Be clear, concise and helpful."""),
        HumanMessage(content=f"""Context from PDF:
{context}

Question: {question}

Answer:""")
    ]

    response = llm.invoke(messages)
    return response.content


def summarize_text(text):
    """
    Summarizes PDF content in a structured format.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content="You are a helpful AI study assistant. Provide clear, structured summaries."),
        HumanMessage(content=f"""Please provide a clear, concise summary of the following text.
Structure your summary with:
- Main topic
- Key points (bullet points)
- Important conclusions

Text to summarize:
{text}

Summary:""")
    ]

    response = llm.invoke(messages)
    return response.content


def generate_quiz(context, num_questions=5):
    """
    Generates multiple choice quiz questions from PDF content.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content="You are a helpful AI study assistant and quiz generator."),
        HumanMessage(content=f"""Based on the following text, generate {num_questions} multiple choice questions
to test understanding of the material.

Format each question exactly like this:
Q1: [Question text]
A) [Option A]
B) [Option B]
C) [Option C]
D) [Option D]
Answer: [Correct letter]
Explanation: [Brief explanation]

Text:
{context}

Quiz:""")
    ]

    response = llm.invoke(messages)
    return response.content


def explain_topic(topic, context):
    """
    Explains a difficult topic from the PDF in simple terms.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content="""You are a helpful AI study assistant and expert teacher.
Explain topics in simple, easy-to-understand terms."""),
        HumanMessage(content=f"""Using the context provided from a PDF document,
explain the following topic in simple terms as if teaching a student.

Use:
- Simple language
- Real-world analogies where helpful
- Step-by-step breakdown if needed
- Examples to clarify concepts

Context from PDF:
{context}

Topic to explain: {topic}

Explanation:""")
    ]

    response = llm.invoke(messages)
    return response.content