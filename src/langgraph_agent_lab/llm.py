"""Provider-aware LLM factory used by classifier and answer nodes."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv

load_dotenv()


def _provider() -> str:
    """Resolve the selected provider, preferring explicit configuration."""
    selected = os.getenv("LLM_PROVIDER", "").strip().lower()
    if selected:
        if selected not in {"openai", "gemini", "anthropic"}:
            raise RuntimeError("LLM_PROVIDER must be one of: openai, gemini, anthropic")
        return selected

    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("ANTHROPIC_API_KEY"):
        return "anthropic"
    raise RuntimeError(
        "No LLM API key found. Set LLM_PROVIDER and its API key, or configure exactly one key."
    )


def get_llm(model: str | None = None, temperature: float = 0.0) -> Any:
    """Create an LLM client from explicit or auto-detected provider configuration."""
    provider = _provider()

    if provider == "openai":
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("LLM_PROVIDER=openai requires OPENAI_API_KEY")
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-openai") from exc
        model_name = model or os.getenv("LLM_MODEL") or "gpt-4o-mini"
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
        )

    if provider == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            raise RuntimeError("LLM_PROVIDER=gemini requires GEMINI_API_KEY")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as exc:
            raise RuntimeError("Install: pip install langchain-google-genai") from exc
        model_name = model or os.getenv("LLM_MODEL") or "gemini-2.5-flash"
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=temperature,
        )

    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("LLM_PROVIDER=anthropic requires ANTHROPIC_API_KEY")
    try:
        from langchain_anthropic import ChatAnthropic
    except ImportError as exc:
        raise RuntimeError("Install: pip install langchain-anthropic") from exc
    model_name = model or os.getenv("LLM_MODEL") or "claude-sonnet-4-20250514"
    return ChatAnthropic(
        model=model_name,
        temperature=temperature,
    )
