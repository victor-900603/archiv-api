import os

import requests


API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")


def query_rag(query: str, history: list[dict] | None = None) -> dict:
    try:
        response = requests.post(
            f"{API_BASE_URL}/rag/ask",
            json={"query": query, "history": history},
            timeout=30,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as exc:
        raise Exception(f"Cannot connect to backend API at {API_BASE_URL}: {exc}") from exc