FinBuddy AI 

An AI-powered financial assistant built with Streamlit and Groq’s Llama 3.1 model.
It brings together live finance news, document-based Q&A, AI summarization, and sentiment analysis — all in one interactive app.
 Features
 Live News Feed — Fetches Yahoo Finance updates and summarizes them.
 Chatbot — Ask finance-related questions to FinBuddy.
 Document Q&A (RAG) — Upload PDFs or text files and get contextual answers
 Summarization — Choose between concise or detailed responses.

⚙️ Tech Stack
Component	Technology
Frontend	Streamlit
LLM	Groq API (Llama 3.1)
Embeddings	SentenceTransformers
Vector Store	FAISS
Data	Yahoo Finance RSS
NLP	NLTK, TextBlob
Env	python-dotenv

🧩 System Flow
User → Streamlit App → Groq API (LLM)
        ├── News Fetcher → Summarizer
        ├── RAG Module → Document Q&A

🧰 Setup
git clone https://github.com/<your-username>/finbuddychatbot.git
cd finbuddychatbot/Project
pip install -r requirements.txt


Create a .env file:

GROQ_API_KEY=your_groq_key_here


Run:

streamlit run app.py

⚠️ Deployment Notes

On Streamlit Cloud, use Python 3.10–3.12 (Torch conflict with 3.13).

Use faiss-cpu instead of GPU builds.

Keep your .env key valid.

📈 Summary

FinBuddy AI:

Demonstrates LLM + RAG integration

Is deployable and modular

Provides real-time financial insights

Shows explainable answers using document context
