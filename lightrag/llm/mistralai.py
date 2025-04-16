import os
import sys
import logging
from typing import Any, Union
from collections.abc import AsyncIterator

import pipmaster as pm
if not pm.is_installed("mistralai"):
    pm.install("mistralai")

from dotenv import load_dotenv
load_dotenv(dotenv_path=".env", override=False)

from mistralai import Mistral
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from httpx import HTTPStatusError, RequestError, TimeoutException

# Preserve your debug helper if you need it
from ..utils import verbose_debug, VERBOSE_DEBUG, logger

class InvalidResponseError(Exception):
    """Raised when the Mistral API returns an empty or malformed response."""
    pass

def create_mistral_async_client(
    api_key: str | None = None,
    endpoint: str | None = None,
    client_configs: dict[str, Any] | None = None,
) -> Mistral:
    """
    Instantiate a Mistral async client.

    :param api_key: Mistral API key (falls back to MISTRAL_API_KEY env var)
    :param endpoint: Base URL for the Mistral API (falls back to MISTRAL_API_BASE env var)
    :param client_configs: Additional kwargs to pass to Mistral(...)
    """
    if client_configs is None:
        client_configs = {}

    if not api_key:
        api_key = os.environ["MISTRAL_API_KEY"]

    # Mistral SDK expects `api_key` and optionally `endpoint`
    merged = {**client_configs, "api_key": api_key}
    if endpoint is not None:
        merged["endpoint"] = endpoint
    else:
        # allow override via env var
        base = os.environ.get("MISTRAL_API_BASE")
        if base:
            merged["endpoint"] = base

    return Mistral(**merged)

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type(
        (HTTPStatusError, RequestError, TimeoutException, InvalidResponseError)
    ),
)
async def mistral_complete_if_cache(
    model: str,
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    endpoint: str | None = None,
    api_key: str | None = None,
    token_tracker: Any | None = None,
    **kwargs: Any,
) -> str:
    """
    Async chat‐completion using Mistral with retry & simple empty‐response checking.
    Mimics your OpenAI wrapper’s signature as closely as possible.
    """
    if history_messages is None:
        history_messages = []

    # drop any kwargs your wrapper used only for OpenAI
    kwargs.pop("hashing_kv", None)
    kwargs.pop("keyword_extraction", None)

    # build the combined message list
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if not VERBOSE_DEBUG and logger.level == logging.DEBUG:
        logging.getLogger("mistralai").setLevel(logging.INFO)

    logger.debug("Sending Mistral chat.completions...")
    verbose_debug(f"Model: {model}, Messages: {messages}, Kwargs: {kwargs}")

    client = create_mistral_async_client(api_key=api_key, endpoint=endpoint)
    # use async context to ensure clean HTTP‐client teardown
    async with client:
        response = await client.chat.complete_async(
            model=model,
            messages=messages,
            **kwargs,
        )

    # extract
    try:
        content = response.choices[0].message.content
    except Exception as e:
        raise InvalidResponseError(f"Malformed response: {e}")

    if not content or not content.strip():
        raise InvalidResponseError("Received empty content from Mistral API")

    # optional token tracking if Mistral returns usage info
    if token_tracker and hasattr(response, "usage"):
        token_tracker.add_usage({
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        })

    return content

async def mistral_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, Any]] | None = None,
    **kwargs: Any,
) -> Union[str, AsyncIterator[str]]:
    """
    Convenience wrapper mirroring your openai_complete() — 
    just calls mistral_complete_if_cache under the hood.
    """
    return await mistral_complete_if_cache(
        kwargs.pop("model", "mistral-large-latest"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    retry=retry_if_exception_type((HTTPStatusError, RequestError, TimeoutException)),
)
async def mistral_embed(
    texts: list[str],
    model: str = "mistral-embed",
    endpoint: str | None = None,
    api_key: str | None = None,
    client_configs: dict[str, Any] | None = None,
) -> np.ndarray:
    """
    Async embedding generator using Mistral’s embeddings API.
    Returns a numpy array of shape (len(texts), embedding_dim).
    """
    client = create_mistral_async_client(api_key=api_key, endpoint=endpoint, client_configs=client_configs)
    async with client:
        response = await client.embeddings.create_async(
            model=model,
            inputs=texts,
        )

    # Mistral returns `response.data` → list of { embedding: [...] }
    try:
        vectors = [item.embedding for item in response.data]
    except Exception as e:
        raise InvalidResponseError(f"Malformed embedding response: {e}")

    return np.array(vectors)
