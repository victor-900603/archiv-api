import requests

API_BASE_URL = "http://localhost:8000"

def query_rag(query: str, history: list[dict] | None = None) -> dict:
    response = requests.post(f"{API_BASE_URL}/rag/ask", json={"query": query, "history": history})
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code} - {response.text}")