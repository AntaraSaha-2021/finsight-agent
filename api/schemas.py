"""
Pydantic schemas for all API inputs and outputs.
Every endpoint has a typed request and response.

Why?
- Rejects malformed input before it reaches the agent
- Documents the API contract explicitly
- Prevents prompt injection via field-level validation
"""

from pydantic import BaseModel, Field, field_validator
import re

class ChatRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The user's financial research question"
    )
    session_id: str = Field(
        default="default",
        max_length=50,
        description="Session ID for conversation memory"
    )

    @field_validator("question")
    @classmethod
    def sanitize_question(cls, v:str) -> str:
        """
        Basic prompt injection guard.
        Blocks common injection patterns.
        This is the first line of defense; not the only one.
        """
        injection_patterns = [
            r"ignore previous instructions",
            r"ignore all instructions",
            r"disregard your"
            r"you are now"
            r"act as"
            r"jailbreak"
        ]
        lowered = v.lower()
        for pattern in injection_patterns:
            if re.search(pattern, lowered):
                raise ValueError(
                    "Question contains disallowed content."
                )
        return v.strip()
    

class ChaResponse(BaseModel):
    answer: str
    session_id: str
    sources: list[str] = []

class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    model: str
    index_ready: bool = False
    active_sessions: int=0
