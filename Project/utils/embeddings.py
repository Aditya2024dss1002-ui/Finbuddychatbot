import os
import sys
import faiss
import numpy as np
import re
from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from langchain_core.messages import HumanMessage, SystemMessage

# Fix import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.llm import get_chatgroq_model


# Load embedding model
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------- TEXT EXTRACTION --------------------
def extract_text_from_pdf(file):
    """Extract and clean text from uploaded PDF."""
    text = ""
    try:
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                cleaned = re.sub(r'\s+', ' ', page_text)
                text += cleaned + " "
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text.strip()


# -------------------- SMART CHUNKING --------------------
def chunk_text_smart(text, max_chunk_size=600):
    """Split text into sentence-based chunks for better retrieval."""
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


# -------------------- VECTOR STORE --------------------
def create_vector_store(text_chunks):
    """Convert text chunks to vectors and create FAISS index."""
    try:
        text_chunks = [t for t in text_chunks if t.strip()]
        if not text_chunks:
            print("No valid text chunks to embed.")
            return None, None

        vectors = embedding_model.encode(text_chunks)
        if len(vectors) == 0:
            print("No vectors created from text.")
            return None, None

        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(np.array(vectors))
        return index, vectors
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None, None


# -------------------- RETRIEVAL --------------------
def retrieve_top_chunks(query, text_chunks, index, vectors, top_k=3):
    """Retrieve top relevant text chunks for a given query."""
    try:
        if index is None or vectors is None:
            print("Index or vectors not initialized.")
            return []
        query_vector = embedding_model.encode([query])
        _, indices = index.search(query_vector, top_k)
        return [text_chunks[i] for i in indices[0] if i < len(text_chunks)]
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return []


# -------------------- RAG ANSWER GENERATION --------------------
def answer_with_rag(query, text_chunks, index, vectors):
    """
    Generate a summarized and traceable answer using document chunks.
    Includes context, reasoning, and document source reference.
    """
    try:
        if not text_chunks or index is None:
            return "Document could not be processed. Please upload a readable file."

        retrieved_chunks = retrieve_top_chunks(query, text_chunks, index, vectors)
        if not retrieved_chunks:
            return "No relevant text found in the uploaded document."

        # Combine retrieved chunks and mark them for traceability
        context = "\n\n".join([f"[Section {i+1}]: {chunk}" for i, chunk in enumerate(retrieved_chunks)])

        # Initialize model
        chat_model = get_chatgroq_model()

        # Improved system instruction for contextual summarization
        prompt = [
            SystemMessage(content=(
                "You are FinBuddy, a professional financial analyst. "
                "You must read the provided document sections and answer the question clearly, "
                "summarizing and contextualizing the relevant parts. "
                "Be concise, factual, and explain as if presenting to an investor or executive."
            )),
            HumanMessage(content=f"Context from uploaded document:\n{context}\n\nQuestion: {query}")
        ]

        response = chat_model.invoke(prompt)

        # Append transparency footer
        final_answer = (
            f"{response.content.strip()}\n\n"
            f"_Response generated using the uploaded document context (Sections {', '.join([str(i+1) for i in range(len(retrieved_chunks))])})._"
        )

        return final_answer

    except Exception as e:
        return f"Error generating RAG answer: {str(e)}"
