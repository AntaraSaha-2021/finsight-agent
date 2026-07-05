from memory import session_memory

def set_function():
    #Clear session memory before each test
    session_memory._store.clear()

def test_get_history_returns_empty_for_new_session():
    result = session_memory.get_history("new-session-123")
    assert result ==[]

def test_add_turn_stores_both_messages():
    #One turn must store exactly 2 messages (user+assistant)
    session_memory.add_turn("test-001", "What is revenuw?", "It is $30B")
    history = session_memory.get_history("test-001")
    assert len(history)==2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

def test_history_enforces_max_length():
    for i in range(session_memory.MAX_HISTORY_LENGTH+5):
        session_memory.add_turn("test-002", f"Question {i}", f"Answer {i}")

    history = session_memory.get_history("test-002")
    max_messages = session_memory.MAX_HISTORY_LENGTH*2
    assert len(history)<=max_messages

def test_invalid_session_id_rejected():
    session_memory.add_turn("../../../etc/passwd", "hack", "attempt")
    result = session_memory.get_history("../../../etc/passwd")
    assert result == []

def test_format_history_empty_for_new_session():
    result = session_memory.format_history_for_prompt("no-history")
    assert result == ""

def test_format_history_returns_readable_string():
    session_memory.add_turn("test-003", "What is TCS revenue?", "It is $30B")
    result = session_memory.format_history_for_prompt("test-003")
    assert "User:" in result
    assert "Assistant:" in result
    assert "TCS revenue" in result

def test_clear_session_remove_history():
    session_memory.add_turn("test-004", "Question", "Answer")
    session_memory.clear_session("test-004")
    assert session_memory.get_history("test-004") == []

def test_get_session_count_reflects_active_sessions():
    initial = session_memory.get_session_count()
    session_memory.add_turn("count-test-001", "Q", "A")
    session_memory.add_turn("count-test-002", "Q", "A")
    assert session_memory.get_session_count() == initial+2

def test_message_content_truncated_at_max_length():
    long_message = "x" * 5000
    session_memory.add_turn("test-005", long_message, "short answer")
    history = session_memory.get_history("test-005")
    assert len(history[0]["content"]) <= session_memory.MAX_MESSAGE_LENGTH