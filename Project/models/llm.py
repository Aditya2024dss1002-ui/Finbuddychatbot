import os
import sys
from langchain_groq import ChatGroq
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

def get_chatgroq_model():
    """Initialize and return the Groq chat model"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("Groq API key not found. Set GROQ_API_KEY in .env or environment variables.")

        # Updated model name
        groq_model = ChatGroq(
            api_key=api_key,
            model="llama-3.1-8b-instant"   # or "llama-3.1-70b-versatile"
        )
        return groq_model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Groq model: {str(e)}")
