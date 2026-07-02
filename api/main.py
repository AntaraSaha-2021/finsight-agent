"""
FinSight Agent - FastAPI application
Exposes the agent as a REST API.

Endpoints:
    GET     /health: liveness check, no auth required
    POST    /chat:   main agent endpoint, auth required
    POST    /upload: PDF upload enpoint, auth required
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, ChaResponse, HealthResponse
from config import settings

from langchain_core.messages import HumanMessage
from agents.graph import agent

app = FastAPI(
    title="FinSight Agent",
    description="Agentic AI system for finacial research",
    version="0.1.0",
    #TODO: Disable docs in prod (for security)
)

#CORS - allows the HTML frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    #TODO: tighten this before production
    allow_methods=["GET","POST"],
    allow_headers=["*"],
)

################Authentication#########
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """
    Validates Bearer token against API_SECRET_KEY in .env
    """
    if credentials.credentials != settings.API_SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return credentials.credentials

#####################Routes################
@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(
        status="Ok",
        llm_provider=settings.LLM_PROVIDER,
        model=(
            settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq" else settings.OLLAMA_MODEL
        ),
    )

@app.post("/chat", response_model=ChaResponse)
def chat(request: ChatRequest, token: str = Depends(verify_token)):
    #Main agent endpoint
    initial_state = {
        "question": request.question,
        "messages": [HumanMessage(content=request.question)],
        "retrieved_docs": [],
        "web_results": [],
        "sources": [],
        "final_answer": "",
        "next_action": "",
    }

    result = agent.invoke(initial_state)

    return ChaResponse(
        answer=result["final_answer"],
        session_id=request.session_id,
        sources=result.get("sources", []),
    )
