from unittest.mock import patch, MagicMock

def test_search_returns_list():
    from tools.web_search_tool import search
    result = search("TCS annual revenue 2024")
    assert isinstance(result, list)

def test_search_returns_empty_on_failure():
    from tools.web_search_tool import search
    with patch("tools.web_search_tool._attempt_search", return_value=None):
        result = search("anything")
        assert result==[]

def test_sanitize_query_caps_length():
    from tools.web_search_tool import _sanitize_query
    long_query = "a"*500
    result = _sanitize_query(long_query)
    assert len(result) <=200

def test_sanitize_query_strips_withespace():
    from tools.web_search_tool import _sanitize_query
    result = _sanitize_query("   TCS revenue    ")
    assert result == "TCS revenue"

def test_format_results_handles_missing_fields():
    from tools.web_search_tool import _format_results
    incomplete = [{"title": "Test"}]
    result = _format_results(incomplete)
    assert len(result) ==1
    assert "Test" in result[0]

def test_search_empty_query_returns_empty():
    from tools.web_search_tool import search
    result = search("")
    assert result==[]