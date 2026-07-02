"""
LLM Factory returns the correct LLM cline based on config.
The rest of the codebase never imports Groq or Ollama directly.
They always call get_llm() from here.

Why?
- Swap LLM provider by changing one .env value
- Test the agent with a mock LLM without touching agent code
- Single place to add logging, retries, of fallback logic later
"""

from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from config import settings

def get_llm():
    """
    Returns the configured LLM instance.
    Fails loudly if provider is misconfigured - never fails silently.
    """
    if settings.LLM_PROVIDER == "groq":
        if not settings.GROQ_API_KEY:
            raise ValueError(
                "LLM_PROVIDER is set to 'groq' but GROQ_API_KEY is missing from .env"
            )
        return ChatGroq(
            model=settings.GROQ_MODEL,
            api_key=settings.GROQ_API_KEY,
            temperature=0,
            max_retries=2,
        )
    elif settings.LLM_PROVIDER == "ollama":
        return ChatOllama(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
        )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER '{settings.LLM_PROVIDER}' in .env. "
            f"Must be 'groq' or 'ollama'."
        )