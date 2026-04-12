import requests
from openai import OpenAI

import config


class LLMBackend:
    def __init__(self, backend: str = config.DEFAULT_BACKEND):
        self.backend = backend

    def call(self, prompt: str, use_strong: bool = False) -> str:
        if self.backend == "openrouter":
            model = config.STRONG_MODEL if use_strong else config.FAST_MODEL
            print(f"[backend] {self.backend} | model={model}", flush=True)
            client = OpenAI(base_url=config.OPENROUTER_BASE_URL, api_key=config.OPENROUTER_API_KEY)
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.choices[0].message.content or ""

        if self.backend == "ollama":
            model = config.OLLAMA_FAST_MODEL
            print(f"[backend] {self.backend} | model={model}", flush=True)
            resp = requests.post(
                config.OLLAMA_URL,
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["response"]

        raise ValueError(f"unknown backend '{self.backend}' — expected 'openrouter' or 'ollama'")

    def name(self) -> str:
        return self.backend
