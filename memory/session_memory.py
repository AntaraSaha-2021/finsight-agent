import re
from datetime import datetime

MAX_HISTORY_LENGTH =10      #max turns per session (1 turn = 1 user + 1 agent)
MAX_MESSAGE_LENGTH = 2000   #max chars per message stored
SESSION_ID_PATTERN = re.compile(r'[a-zA-Z0-9_-]{1,50}$')


#---------------In-memory store-----------------------------
#{session_id: [{role, content, timestamp}, ...]}
_store: dict[str, list[dict]] ={}


#----------------------Validation---------------------------------
def _validate_session_id(session_id: str) -> bool:
    return bool(SESSION_ID_PATTERN.match(session_id))

#----------------------------Public interface-------------------------
def get_history(session_id: str) -> list[dict]:
    #returns conversation history for a session
    #returns empty if session does not exist
    if not _validate_session_id(session_id):
        return []
    return _store.get(session_id, [])

def add_turn(session_id: str, question: str, answer: str) -> None:
    if not _validate_session_id(session_id):
        return
    
    if session_id not in _store:
        _store[session_id] =[]

    timestamp = datetime.utcnow().isoformat()

    #Add use turn
    _store[session_id].append(
        {
            "role": "user",
            "content": question[:MAX_MESSAGE_LENGTH],
            "timestamp": timestamp,
        }
    )

    #Add agent turn
    _store[session_id].append(
        {
            "role": "assistant",
            "content": answer[:MAX_MESSAGE_LENGTH],
            "timestamp": timestamp,
        }
    )

    #Enforce max history - drop oldest turns (2 messages = 1 turn)
    max_messages = MAX_HISTORY_LENGTH * 2
    if len(_store[session_id]) > max_messages:
        _store[session_id] = _store[session_id][-max_messages:]


def format_history_for_prompt(session_id: str) -> str:
    history = get_history(session_id)
    if not history:
        return ""
    
    lines =[]
    for msg in history:
        role = "User" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}: {msg['content']}")
    
    return "\n".join(lines)


def clear_session(session_id: str) -> None:
    if session_id in _store:
        del _store[session_id]

def get_session_count() -> int:
    return len(_store)