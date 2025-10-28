from models.llm import get_chatgroq_model
from langchain_core.messages import HumanMessage, SystemMessage

def summarize_text(text, concise=True):
    """Summarize text using LLM"""
    try:
        model = get_chatgroq_model()
        if not model:
            return "⚠️ Model not initialized."

        style = "Give a short one-sentence summary." if concise else "Provide a detailed summary with insights."
        prompt = [
            SystemMessage(content="You are a financial news analyst."),
            HumanMessage(content=f"{style}\n\n{text}")
        ]
        response = model.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Error summarizing: {str(e)}"
