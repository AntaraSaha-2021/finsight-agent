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
"""

def supervisor_node(state: AgentState) -> dict:
    llm = get_llm()

    has_docs=bool(state.get("retrieved_docs"))
    has_web=bool(state.get("web_results"))
    previous_action=state.get("next_action", "none")

    context_summary =[]

    if has_docs:
        context_summary.append(
            f"RAG retrieval: found {len(state['retrieved_docs'])} document chunks"
        )
    if has_web:
        context_summary.append(
            f"Web Search: found {len(state['web_results'])} results"
        )
    if not context_summary:
        context_summary.append("No tools have been used yet.")
    
    #Add conversation history context to supervisor decision
    history = state.get("conversation_history", "")
    history_context = f"\nConversation so far:\n{history}" if history else ""

    decision_prompt = f"""You are managing a financial research task.
    
    Question: {state['question']}
    {history_context}

    Current context gathere:
    {chr(10).join(context_summary)}

    Previous action: {previous_action}

    Decide what to do next. Rules:
    - If no tools used yet: choose the most appropriate tool
    - If RAG was used but returned nothing: try web_search
    - If web_search was used but returned nothing: go to answer and explain limitation
    - If sufficient context exists from either tool: go to answer
    - Do not repeat a tool that already ran successfully
    - Never run the same tool twice

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
        print(f"[Supervisor] Unexpected decision '{decision}' - defaulting to answer")
        decision = "answer"

    return {
        "next_action": decision,
        "messages": [AIMessage(content=f"[Supervisor] Routing to: {decision}")]
    }

def rag_node(state: AgentState) -> dict:
    from tools.rag_tool import retrieve

    query = state["question"]
    retrieved = retrieve(query)

    existing = state.get("sources", [])
    new_sources = ["document"] if retrieved and "document" not in existing else []


    # if not retrieved:
    #     return {
    #         "retrieved_docs": [],
    #         "next_action": "web_search",
    #         "messages": [AIMessage(content="[RAG] No documents in index. Routing to web search.")]
    #     }
    
    return {
        "retrieved_docs": retrieved,
        "sources": existing + new_sources,
        # "next_action": "answer",
        "messages": [AIMessage(content=f"[RAG] Retrieved {len(retrieved)} relevant chunks." if retrieved else "[RAG] No documents found in index.")]
    }

def web_node(state: AgentState) -> dict:
    from tools.web_search_tool import search

    query = state['question']
    results = search(query)

    existing = state.get("sources", [])
    new_sources = ["web"] if results and "web" not in existing else []

    return {
        "web_results": results,
        "sources": existing + new_sources,
        # "next_action": "answer",
        "messages": f"[WebSearch] found {len(results)} results." if results else "[WebSearch] No results found."
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

    history = state.get("conversation_history","")
    history_context = f"\nPrevious conversation:\n{history}\n" if history else ""

    answer_prompt = f"""Using the context below, answer the user's question.
    
    Question: {state['question']}
    {history_context}
    
    Context:
    {context}

    Instructions:
    - If the question refers to something from previous conversation, use that context
    - Answer based strictly on retrieved context and conversation histroy
    - If context is insufficient, clearly say what information is missing
    - Be concise but complete
    - Never fabricate financial figures or dates
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