"""
What regression could this introduce?
- Adding a new node without a corresponding edge = dead end, agent hangs forever
- Renaming next_action values without updating route function = KeyError at runtime, silent in dev, catastrophic in prod
- These tests catch both immediately
"""
import pytest
from agents.state import AgentState
from agents.graph import build_graph, agent

def test_graph_builds_without_error():
    graph = build_graph()
    assert graph is not None

def test_graph_has_correct_nodes():
    graph = build_graph()
    node_names = set(graph.nodes.keys())
    expected = {"supervisor", "rag", "web_search", "answer"}
    assert expected.issubset(node_names)

def test_supervisor_routes_to_valid_node():
    #Supervisor must always return a valid next_action
    from agents.nodes import supervisor_node

    state = AgentState(
        question="What is the revenue of Infosys?",
        messages=[],
        next_action="",
        retrieved_docs=[],
        web_results=[],
        final_answer="",
        sources=[],
    )

    result = supervisor_node(state)
    assert result["next_action"] in ["rag", "web_search", "answer"]

def test_full_agent_run_returns_answer():
    #Agent MUST return a non-empty final answer
    initial_state ={
        "question": "What is return on equity?",
        "messages": [],
        "retrieved_docs": [],
        "web_results": [],
        "sources": [],
        "final_answer": "",
        "next_action": ""
    }

    result = agent.invoke(initial_state)

    assert result["final_answer"] != ""
    assert len(result["messages"]) > 0

def test_answer_node_handles_empty_context():
    from agents.nodes import answer_node

    state = AgentState(
        question="What is EBITDA?",
        messages=[],
        next_action="answer",
        retrieved_docs=[],
        web_results=[],
        final_answer="",
        sources=[],
    )

    result = answer_node(state)
    assert result["final_answer"] != ""

def test_react_loop_does_not_exceed_max_iterations():
    from agents.graph import MAX_ITERATIONS
    from agents.nodes import supervisor_node
    from langchain_core.messages import AIMessage

    fake_supervisor_messages = [
        AIMessage(content="[Supervisor] Decision: rag")
        for _ in range(MAX_ITERATIONS)
    ]

    from agents.graph import route_after_supervisor
    from agents.state import AgentState

    state = AgentState(
        question="test",
        messages=fake_supervisor_messages,
        next_action="rag",
        retrieved_docs=[],
        web_results=[],
        final_answer="",
        sources=[],
    )

    result = route_after_supervisor(state)
    assert result=="answer"