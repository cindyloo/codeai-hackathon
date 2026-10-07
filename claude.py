"""Claude client for the prompt endpoint, via the Anthropic Messages API.

The API key comes from the environment (`ANTHROPIC_API_KEY`) and is only ever
handed to the SDK, so nothing here echoes a secret.
"""

from __future__ import annotations

import threading

import config
from logs import log

_client = None
_client_lock = threading.Lock()


class ClaudeError(Exception):
    """Failure to report to the caller. ``status`` is the HTTP code to send."""

    def __init__(self, message: str, status: int = 502, detail: str = ""):
        super().__init__(detail or message)
        self.message = message
        self.status = status
        self.detail = detail


def _get_client():
    """Build the Anthropic client once, lazily.

    Lazily because the SDK is only needed when the endpoint is actually used —
    the app has to start and serve the avatar experience on a machine with no
    Anthropic key at all.
    """
    global _client
    with _client_lock:
        if _client is None:
            try:
                import anthropic
            except ImportError as exc:
                raise ClaudeError(
                    "The model service isn't installed on this server.",
                    status=503,
                    detail="anthropic is missing; pip install -r requirements.txt",
                ) from exc

            if not config.ANTHROPIC_API_KEY:
                raise ClaudeError(
                    "The model service isn't configured.",
                    status=503,
                    detail="Set ANTHROPIC_API_KEY",
                )

            _client = anthropic.Anthropic(
                api_key=config.ANTHROPIC_API_KEY,
                timeout=config.CLAUDE_TIMEOUT,
                # Let the SDK back off on 429/5xx rather than surfacing a
                # transient throttle as a failure on the first attempt.
                max_retries=3,
            )
    return _client


def allowed_models() -> list[str]:
    return list(config.CLAUDE_ALLOWED_MODELS)


def check_model(model_id: str) -> None:
    """Reject anything outside the allowlist.

    An empty allowlist refuses everything. That is the intended default: an
    endpoint that will run whatever model string it is handed is an open
    invitation to invoke the most expensive model on the account.
    """
    if not config.CLAUDE_ALLOWED_MODELS:
        raise ClaudeError(
            "No models are enabled on this server.",
            status=503,
            detail="CLAUDE_ALLOWED_MODELS is empty; nothing can be invoked",
        )
    if model_id not in config.CLAUDE_ALLOWED_MODELS:
        raise ClaudeError(
            f"Model '{model_id}' isn't on the allowed list.",
            status=400,
            detail=f"allowed: {', '.join(config.CLAUDE_ALLOWED_MODELS)}",
        )


def converse(model_id: str, prompt: str, *, system: str | None = None,
             max_tokens: int = 1024, temperature: float | None = None
             ) -> dict:
    """Run one prompt against one Claude model and return the reply.

    ``temperature`` is omitted from the request unless explicitly supplied —
    current Claude models reject sampling parameters outright, so sending a
    default would break exactly the models most people will reach for.
    """
    check_model(model_id)

    request: dict[str, object] = {
        "model": model_id,
        "max_tokens": max(1, min(int(max_tokens), config.CLAUDE_MAX_TOKENS)),
        "messages": [{"role": "user", "content": prompt}],
    }
    if system:
        request["system"] = system
    if temperature is not None:
        request["temperature"] = max(0.0, min(float(temperature), 1.0))

    client = _get_client()
    try:
        response = client.messages.create(**request)
    except Exception as exc:  # noqa: BLE001 - translated below
        raise _translate(exc) from exc

    text = "".join(block.text for block in response.content
                   if block.type == "text")
    usage = response.usage

    return {
        "model_id": model_id,
        "text": text,
        # "refusal" here means a safety classifier declined the prompt.
        "stop_reason": response.stop_reason,
        "usage": {
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "total_tokens": usage.input_tokens + usage.output_tokens,
        },
    }


def _translate(exc: Exception) -> ClaudeError:
    """Map an Anthropic SDK exception to something safe to return.

    The caller sees a message written here; the original goes to the log.
    """
    import anthropic

    # Most specific first: several of these subclass APIStatusError.
    known: list[tuple[type[Exception], str, int]] = [
        (anthropic.AuthenticationError,
         "The server's Anthropic API key is invalid.", 503),
        (anthropic.PermissionDeniedError,
         "This server isn't allowed to use that model.", 403),
        (anthropic.NotFoundError,
         "That model isn't available.", 404),
        (anthropic.RateLimitError,
         "The model service is busy. Try again in a moment.", 429),
        (anthropic.BadRequestError,
         "That request wasn't valid for this model.", 400),
        (anthropic.APITimeoutError,
         "The model took too long to reply.", 504),
        (anthropic.InternalServerError,
         "The model service is having trouble. Try again?", 502),
        (anthropic.APIConnectionError,
         "The server couldn't reach the model service.", 502),
    ]
    message, status = "The model service failed. Try again?", 502
    for cls, known_message, known_status in known:
        if isinstance(exc, cls):
            message, status = known_message, known_status
            break

    name = type(exc).__name__
    log(f"[claude] {name}: {exc}")
    return ClaudeError(message, status=status, detail=f"{name}: {exc}")
