from typing import Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from typing_extensions import TypedDict

class AgentState(TypedDict):
    """
    The complete state of one agent run.
    Every field is optional except 'message'. nodes only populate what they contribute.
    """

    #Full conversation history
    messages: Annotated[list[BaseMessage], add_messages]

    #The current user question, extracted for tool use
    question: str

    #Which tool the supervisor decided to use. values: "rag", "web_search", "answer"
    next_action: str

    #Documents retrieved by RAG tool
    retrieved_docs: list[str]

    #Web search results
    web_results: list[str]

    #final systhesized answer
    final_answer: str

    #Source attribution: what was used to answer
    sources: list[str]
