from langchain.memory import ConversationSummaryMemory
from langchain_ollama import OllamaLLM

import config


def get_memory() -> ConversationSummaryMemory:
    """Return a conversation memory that summarises history using the local LLM."""
    llm = OllamaLLM(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL)
    return ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
