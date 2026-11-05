"""Thin client for Ollama's /api/chat, non-streaming."""

import os
import time

import requests

RETRIES = 3
RETRY_WAIT = 5  # seconds


def host():
    # the same variable the ollama command reads, e.g. OLLAMA_HOST=127.0.0.1:11435
    h = os.environ.get("OLLAMA_HOST", "localhost:11434").rstrip("/")
    return h if "://" in h else f"http://{h}"


def chat(model, messages, tools=None, num_ctx=8192, num_predict=512, temperature=0.0, seed=0, format=None) -> dict:
    body = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature, "seed": seed},
    }
    if tools:
        body["tools"] = tools
    if format:
        body["format"] = format
    for attempt in range(RETRIES + 1):
        try:
            r = requests.post(f"{host()}/api/chat", json=body, timeout=600)
            break
        except requests.ConnectionError:
            if attempt == RETRIES:
                raise
            time.sleep(RETRY_WAIT)
    if r.status_code != 200:
        # Ollama puts the useful part ("model not found", out of memory) in the body
        raise RuntimeError(f"ollama returned {r.status_code}: {r.text[:300]}")
    out = r.json()
    return {
        "message": out["message"],
        "prompt_tokens": out.get("prompt_eval_count", 0),
        # the part of the prompt Ollama reused from the previous request instead of
        # processing again; prompt_s only covers the rest
        "cached_tokens": out.get("prompt_eval_cached_count", 0),
        "output_tokens": out.get("eval_count", 0),
        "prompt_s": out.get("prompt_eval_duration", 0) / 1e9,
        "gen_s": out.get("eval_duration", 0) / 1e9,
        "total_s": out.get("total_duration", 0) / 1e9,
    }
