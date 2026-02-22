from __future__ import annotations

import httpx

from app.core.config import settings

'''
    send the grounded prompt to LLM and get the response . low temp=more factual output
'''
def generate(prompt: str, temperature: float = 0.2) -> str:

    payload = {
        "model":  settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        resp = httpx.post(
            f"{settings.OLLAMA_BASE_URL}/api/generate",
            json=payload,
            timeout=60.0,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()
    except httpx.HTTPError as exc:
        return f"[LLM Error] Could not reach Ollama: {exc}"