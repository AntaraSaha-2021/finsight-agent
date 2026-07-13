"""
FinSight Agent - FastAPI application
Exposes the agent as a REST API.

Endpoints:
    GET     /health: liveness check, no auth required
    POST    /chat:   main agent endpoint, auth required
    POST    /upload: PDF upload enpoint, auth required
    DELETE  /session/session_id: clears the active sessions
"""

from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import ChatRequest, ChatResponse, HealthResponse
from config import settings

from langchain_core.messages import HumanMessage
from agents.graph import agent

import shutil
from pathlib import Path

app = FastAPI(
    title="FinSight Agent",
    description="Agentic AI system for financial research",
    version="0.1.0",
    #TODO: Disable docs in prod (for security)
)

#CORS - allows the HTML frontend to call this API
app.add_middleware(
    CORSMiddleware,
    # allow_origins=["*"],    
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_methods=["GET","POST","DELETE"],
    allow_headers=["Authorization", "Content-Type"],
    # allow_headers=["*"],
    allow_credentials=False,
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
    from tools.rag_tool import get_index_status
    from memory.session_memory import get_session_count

    index_status = get_index_status()

    return HealthResponse(
        status="OK",
        llm_provider=settings.LLM_PROVIDER,
        model=(
            settings.GROQ_MODEL if settings.LLM_PROVIDER == "groq" else settings.OLLAMA_MODEL
        ),
        index_ready = index_status["index_exists"],
        active_sessions=get_session_count(),
    )

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest, token: str = Depends(verify_token)):
    #Main agent endpoint
    from memory.session_memory import (format_history_for_prompt, add_turn)

    history = format_history_for_prompt(request.session_id)

    initial_state = {
        "question": request.question,
        "messages": [HumanMessage(content=request.question)],
        "retrieved_docs": [],
        "web_results": [],
        "sources": [],
        "final_answer": "",
        "next_action": "",
        "session_id": request.session_id,
        "conversation_history": history,
    }

    result = agent.invoke(initial_state)

    #Store this turn in memory for next call
    add_turn(session_id=request.session_id,
             question=request.question,
             answer=result["final_answer"])

    return ChatResponse(
        answer=result["final_answer"],
        session_id=request.session_id,
        sources=list(set(result.get("sources", []))),
    )

@app.post("/upload")
def upload_pdf(file: UploadFile=File(...), token: str=Depends(verify_token)):
    from tools.rag_tool import ingest_pdf

    safe_filename = Path(file.filename).name
    if not safe_filename.endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are accepted"
        )
    
    data_dir = Path(settings.DATA_DIR)
    data_dir.mkdir(exist_ok=True)
    save_path = data_dir / safe_filename

    try:
        with save_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    result = ingest_pdf(str(save_path))

    if not result["success"]:
        raise HTTPException(
            status_code=422,
            detail=result["error"]
        )
    
    return {
        "message": "PDF ingested successfully",
        "filename": safe_filename,
        "pages": result["pages"],
        "chunks": result["chunks"],
    }

@app.delete("/session/{session_id}")
def clear_session(session_id: str, token: str = Depends(verify_token)):
    #Useful for starting fresh without restarting server
    from memory.session_memory import clear_session as _clear
    _clear(session_id)
    return {"message": f"Session {session_id} cleared"}