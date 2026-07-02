import pytest
from unittest.mock import patch

def test_get_llm_returns_groq_when_configured():
    from agents.llm_factory import get_llm
    from langchain_groq import ChatGroq

    llm=get_llm()
    assert isinstance(llm, ChatGroq)

def test_get_llm_raises_if_groq_key_missing():
    from config import settings

    original_key = settings.GROQ_API_KEY
    settings.GROQ_API_KEY=""

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        from agents.llm_factory import get_llm
        get_llm()
    
    settings.GROQ_API_KEY=original_key  #restore

def test_get_llm_raises_on_unknown_provider():
    from config import settings

    original = settings.LLM_PROVIDER
    settings.LLM_PROVIDER = "openai"

    with pytest.raises(ValueError, match="Unknown LLM_PROVIDER"):
        from agents.llm_factory import get_llm
        get_llm()
    
    settings.LLM_PROVIDER=original      #restore

def test_llm_responds_to_simple_prompt():
    from agents.llm_factory import get_llm
    from langchain_core.messages import HumanMessage

    llm=get_llm()
    response = llm.invoke([HumanMessage(content="Reply with the word PONG only.")])

    assert response.content.strip().upper() == "PONG"