from langgraph.graph import StateGraph, START, END
from agents.state import AgentState
from agents.nodes import supervisor_node, rag_node, web_node, answer_node

def route_after_supervisor(state: AgentState) -> str:
    return state["next_action"]

def build_graph():
    graph = StateGraph(AgentState)

    #-----------Register nodes--------------------
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web_search", web_node)
    graph.add_node("answer", answer_node)

    #------------Register edges-------------------
    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "rag": "rag",
            "web_search": "web_search",
            "answer": "answer",
        }
    )

    graph.add_edge("rag", "answer")
    graph.add_edge("web_search", "answer")

    graph.add_edge("answer", END)

    #----------Compile------------------------------
    return graph.compile()

agent = build_graph()