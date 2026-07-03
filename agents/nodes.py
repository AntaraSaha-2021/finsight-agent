from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from agents.state import AgentState
from agents.llm_factory import get_llm

#System prompt: the agent's identity and rules
SYSTEM_PROMPT="""You are FinSight, an expert financial research assistant.

Your job is to help users research companies, analyze financial documents,
and synthesize information from multiple sources.

Rules you always follow:
- Only make claims supported by retrieved documents or search results
- Clearly distinguish between document-based facts and general knowledge
- If you are uncertain, say so explicitly
- Never fabricate financial figures, date, or company data
- Always cite your sources

You have access to two tools:
1. rag_tool: searches uploaded financial documents (annual reports, filings)
2. web_search: searches the web for lastest news and information
"""

def supervisor_node(state: AgentState) -> dict:
    llm = get_llm()

    decision_prompt = f"""Given this financial research question, decide which tool to use first.
    
    Question: {state['question']}

    Available tools:
    - rag: Use when question asks about uploaded documents, annual reports, specific filings, or document-based financial data
    - web_search: Use when question asks about recent news, current prices, latest developments, or information not in documents
    - answer: Use ONLY when you already have enough context to answer well

    Respond with exactly one word: rag, web_search, or answer    
    """
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=decision_prompt)
    ]

    response = llm.invoke(messages)

    #Parse the decision clean and lowercase
    decision = response.content.strip().lower()

    #Guard: if LLM returns something unexpected, default to web_search
    if decision not in ["rag", "web_search", "answer"]:
        decision = "web_search"

    return {
        "next_action": decision,
        "messages": [AIMessage(content=f"[Supervisor] Routing to: {decision}")]
    }

def rag_node(state: AgentState) -> dict:
    from tools.rag_tool import retrieve

    query = state["question"]
    retrieved = retrieve(query)

    if not retrieved:
        return {
            "retrieved_docs": [],
            "next_action": "web_search",
            "messages": [AIMessage(content="[RAG] No documents in index. Routing to web search.")]
        }
    
    return {
        "retrieved_docs": retrieved,
        "sources": ["document"],
        "next_action": "answer",
        "messages": [AIMessage(content=f"[RAG] Retrieved {len(retrieved)} relevant chunks.")]
    }

def web_node(state: AgentState) -> dict:
    #TODO: Currently a placeholder. Full implementation later
    return {
        "web_results": ["[Web search not yet implemented, coming soon]"],
        "sources": ["web"],
        "next_action": "answer",
        "message": [AIMessage(content="[Web] Search results placeholder.")]
    }

def answer_node(state: AgentState) -> dict:
    llm = get_llm()

    #Build context from whatever was gathered
    context_parts = []

    if state.get("retrieved_docs"):
        docs_text = "\n".join(state["retrieved_docs"])
        context_parts.append(f"FROM DOCUMENTS:\n{docs_text}")
    
    if state.get("web_results"):
        web_text = "\n".join(state["web_results"])
        context_parts.append(f"FROM WEB SEARCH:\n{web_text}")

    context = "\n\n".join(context_parts) if context_parts else "No context gathered."

    answer_prompt = f"""Using the context below, answer the user's question.
    
    Question: {state['question']}
    
    Context:
    {context}

    Instructions:
    - Answer based on the context provided
    - If context is insufficient, say what information is missing
    - Be concise but complete
    - Cite which source (document or web) each fact comes from"""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=answer_prompt),
    ]

    response = llm.invoke(messages)

    return {
        "final_answer": response.content,
        "messages": [AIMessage(content=response.content)]
    }