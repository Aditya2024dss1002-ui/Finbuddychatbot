import streamlit as st
import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Load environment variables
load_dotenv()

# Project imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from models.llm import get_chatgroq_model
from utils.news_fetcher import fetch_yahoo_finance_news
from utils.summarizer import summarize_text
from utils.rag_handler import extract_text_from_pdf, create_vector_store, answer_with_rag

# ------------- Page Layout -------------
st.set_page_config(page_title="FinBuddy AI", layout="wide")

def styled_title(title):
    st.markdown(f"<h2 style='color:#004aad; font-weight:600;'>{title}</h2>", unsafe_allow_html=True)

def get_chat_response(chat_model, messages, system_prompt):
    """LLM chat handler"""
    try:
        formatted_messages = [SystemMessage(content=system_prompt)]
        for msg in messages:
            formatted_messages.append(
                HumanMessage(content=msg["content"]) if msg["role"] == "user" else AIMessage(content=msg["content"])
            )
        response = chat_model.invoke(formatted_messages)
        return response.content
    except Exception as e:
        return f"Error generating response: {str(e)}"

# --------------------------------------
def main():
    st.sidebar.title("FinBuddy Navigation")
    page = st.sidebar.radio("Go to:", ["Finance Chat", "Document Q&A (RAG)", "Instructions"], index=0)
    st.sidebar.markdown("---")
    if st.sidebar.button("Clear Chat"):
        st.session_state.clear()
        st.rerun()

    if page == "Finance Chat":
        finance_chat_page()
    elif page == "Document Q&A (RAG)":
        rag_page()
    else:
        instructions_page()

# --------------------------------------
def finance_chat_page():
    styled_title("FinBuddy AI - Financial Assistant")

    # Columns for layout
    col1, col2 = st.columns([2, 1])

    # --- News Section ---
    with col1:
        styled_title("Latest Financial News")
        category = st.selectbox("Select Category", ["Top News", "Markets", "Crypto", "Technology", "Economy"])
        df = fetch_yahoo_finance_news(category)

        if not df.empty:
            for _, row in df.iterrows():
                st.markdown(f"**[{row['title']}]({row['link']})**")
                st.caption(f"{row['published']} | {row['source']}")
                st.write(row["summary"])
                st.divider()
        else:
            st.warning("No news articles found.")

    # --- Summarizer Section ---
    with col2:
        styled_title("Summarize Top News")
        concise_mode = st.radio("Response Style", ["Concise", "Detailed"])
        if st.button("Summarize"):
            if not df.empty:
                combined_text = " ".join(df["summary"].tolist())
                summary = summarize_text(combined_text, concise=(concise_mode == "Concise"))
                st.success(summary)
            else:
                st.warning("No articles to summarize.")

    # --- Chatbot Section ---
    styled_title("Ask FinBuddy Anything")
    system_prompt = "You are FinBuddy, a professional AI financial advisor."

    chat_model = get_chatgroq_model()
    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about finance, investing, or markets..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = get_chat_response(chat_model, st.session_state.messages, system_prompt)
                st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# --------------------------------------
def rag_page():
    styled_title("Document Q&A - Retrieval Augmented Generation (RAG)")
    st.info("Upload your financial documents (PDF, TXT, CSV) and ask FinBuddy questions about them.")

    uploaded_file = st.file_uploader("Upload Document", type=["pdf", "txt", "csv"])
    if uploaded_file:
        with st.spinner("Processing document..."):
            if uploaded_file.type == "application/pdf":
                text = extract_text_from_pdf(uploaded_file)
            else:
                text = uploaded_file.read().decode("utf-8")
            chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
            index, vectors = create_vector_store(chunks)
            st.success("Document processed successfully! You can now ask questions.")

        query = st.text_input("Ask a question about your document:")
        if query:
            with st.spinner("Generating answer..."):
                answer = answer_with_rag(query, chunks, index, vectors)
                st.write(answer)
    else:
        st.warning("Please upload a document to start.")

# --------------------------------------
def instructions_page():
    styled_title("Setup Instructions")
    st.markdown("""
    **FinBuddy AI** combines financial news, summarization, and document-based Q&A.  
    Use the sidebar to navigate between features.

    **Quick Setup**
    1. Install dependencies with:
       ```
       pip install -r requirements.txt
       ```
    2. Create a `.env` file with:
       ```
       GROQ_API_KEY=your_key_here
       ```
    3. Run:
       ```
       streamlit run app.py
       ```
    """)

# --------------------------------------
if __name__ == "__main__":
    main()
