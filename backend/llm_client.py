import logging
import os
from typing import Optional

import requests

LOG = logging.getLogger(__name__)


class GeminiClient:
    """Minimal Gemini REST client.

    Environment variables:
      - GEMINI_API_KEY: Google Gemini API key
      - GEMINI_MODEL: model name, defaults to gemini-1.5-flash
      - GEMINI_API_URL: optional base URL, defaults to generativelanguage.googleapis.com
    """

    def __init__(self) -> None:
        self.key = os.environ.get("GEMINI_API_KEY")
        self.model = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
        self.base_url = os.environ.get("GEMINI_API_URL", "https://generativelanguage.googleapis.com")

    def enabled(self) -> bool:
        return bool(self.key)

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        return f"{base}/v1beta/models/{self.model}:generateContent?key={self.key}"

    def generate(self, prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> Optional[str]:
        if not self.enabled():
            LOG.debug("GeminiClient not configured (missing GEMINI_API_KEY)")
            return None

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": prompt}],
                }
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        try:
            resp = requests.post(self._endpoint(), json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            candidates = data.get("candidates") if isinstance(data, dict) else None
            if candidates:
                first = candidates[0]
                parts = first.get("content", {}).get("parts", []) if isinstance(first, dict) else []
                if parts:
                    text = parts[0].get("text") if isinstance(parts[0], dict) else None
                    if isinstance(text, str) and text.strip():
                        return text.strip()
            LOG.warning("GeminiClient: unexpected response format")
        except Exception as exc:
            LOG.exception("GeminiClient request failed: %s", exc)
        return None
