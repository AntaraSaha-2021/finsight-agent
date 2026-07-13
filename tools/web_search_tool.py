import time
from ddgs import DDGS

MAX_QUERY_LENGTH=200
MAX_RESULTS=5
RETRY_DELAY_SECONDS=2

def _sanitize_query(query: str) -> str:
    #Sanitizes search query before sending it to DDG. Caps length, strips whitespace.
    return query.strip()[:MAX_QUERY_LENGTH]

def _attempt_search(query: str, max_results: int) -> list[dict] | None:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(
                query,
                max_results=max_results
            ))
            return results if results else None
    except Exception as e:
        print(f"[WebSearch] Attampt failed: {e}")
        return None
    
def _format_results(results: list[dict]) -> list[str]:
    formatted =[]
    for r in results:
        title = r.get("title", "No title")
        body = r.get("body", "No content")
        href = r.get("href", "No URL")
        formatted.append(f"[{title}] {body} (Source: {href})")
    return formatted

def search(query: str, max_results: int=MAX_RESULTS) -> list[str]:
    #Each returned string is formatted as: [Title] snippet text (url)
    #Returns empty list on any failure. 
    #Caller (web_node) handles empty results.
    sanitized = _sanitize_query(query)

    if not sanitized:
        return []
    
    results = _attempt_search(sanitized, max_results)

    #Retry once if first attempt failed
    if results is None:
        time.sleep(RETRY_DELAY_SECONDS)
        results = _attempt_search(sanitized, max_results)
    
    if not results:
        return []
    
    return _format_results(results)