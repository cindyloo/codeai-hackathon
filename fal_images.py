"""fal.ai image client: sketch rendering and the T-pose transform.

Rendering and the T-pose redraw both go to an instruction-driven image editor
(FLUX.1 Kontext by default): the drawing goes in as the reference image and
the prompt says what to do with it. Background removal is a separate
segmentation model (BiRefNet), since the editor can't output an alpha channel.

The API key comes from the environment (`FAL_KEY`) and is only ever handed to
the SDK, so nothing here echoes a secret.
"""

from __future__ import annotations

import base64
import threading

import config
from logs import log

_client = None
_client_lock = threading.Lock()


class FalError(Exception):
    """Failure to report to the caller. ``status`` is the HTTP code to send."""

    def __init__(self, message: str, status: int = 502, detail: str = ""):
        super().__init__(detail or message)
        self.message = message
        self.status = status
        self.detail = detail


def _get_client():
    """Build the fal client once, lazily.

    Lazily because the SDK is only needed when an image feature is actually
    used — the app has to start and serve the avatar experience on a machine
    with no fal key at all.
    """
    global _client
    with _client_lock:
        if _client is None:
            try:
                import fal_client
            except ImportError as exc:
                raise FalError(
                    "The image service isn't installed on this server.",
                    status=503,
                    detail="fal-client is missing; pip install -r requirements.txt",
                ) from exc

            if not config.FAL_KEY:
                raise FalError(
                    "The image service isn't configured.",
                    status=503,
                    detail="Set FAL_KEY",
                )

            _client = fal_client.SyncClient(
                key=config.FAL_KEY, default_timeout=config.FAL_TIMEOUT)
    return _client


def _sniff_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _data_uri(image_bytes: bytes) -> str:
    mime = _sniff_mime(image_bytes)
    return f"data:{mime};base64,{base64.b64encode(image_bytes).decode('ascii')}"


def _call(model_id: str, arguments: dict) -> dict:
    client = _get_client()
    try:
        return client.subscribe(model_id, arguments,
                                client_timeout=config.FAL_TIMEOUT)
    except Exception as exc:  # noqa: BLE001 - translated below
        raise _translate(exc) from exc


def _fetch(image: dict | None) -> bytes:
    """Bytes of a fal output image: inline data URI, or a CDN URL."""
    url = (image or {}).get("url") or ""
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    if not url:
        raise FalError("The model didn't return an image.", status=502,
                       detail="empty image url")

    import httpx
    try:
        response = httpx.get(url, timeout=config.FAL_TIMEOUT,
                             follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        log(f"[fal] fetching output failed: {exc}")
        raise FalError("The finished image couldn't be downloaded.",
                       status=502, detail=str(exc)) from exc
    return response.content


def _edit_image(image_bytes: bytes, instruction: str, *, model_id: str,
                seed: int = 0) -> dict:
    """Send one image plus an instruction; return the first image back."""
    arguments: dict[str, object] = {
        "prompt": instruction,
        "image_url": _data_uri(image_bytes),
        "output_format": "png",
        "num_images": 1,
        # Return the image inline rather than as a CDN URL: one less request,
        # and nothing left sitting on fal's storage.
        "sync_mode": True,
    }
    if seed:
        arguments["seed"] = int(seed)

    result = _call(model_id, arguments)

    # fal blanks flagged outputs (a black image) rather than erroring.
    if any(result.get("has_nsfw_concepts") or []):
        raise FalError(
            "That couldn't be rendered — try a different drawing or prompt.",
            status=422, detail="has_nsfw_concepts")

    images = result.get("images") or []
    if not images:
        raise FalError("The model didn't return an image.", status=502,
                       detail=f"no images from {model_id}")

    image = images[0]
    mime = image.get("content_type") or "image/png"
    return {
        "image_bytes": _fetch(image),
        "output_format": mime.split("/")[-1],
        "seed": result.get("seed"),
    }


def render_sketch(image_bytes: bytes, prompt: str, *,
                  negative_prompt: str | None = None,
                  seed: int = 0,
                  model_id: str | None = None) -> dict:
    """Render a line drawing into a finished image.

    There is no caller-selectable ``model_id`` from the browser's point of
    view: it defaults to ``config.FAL_RENDER_MODEL``, and the only other
    caller (``pose_to_tshape``) passes a different fixed model of its own
    rather than letting one bubble up from a request.

    The editor has no negative-prompt field, so ``negative_prompt`` is folded
    into the instruction as an explicit list of things to avoid.
    """
    instruction = (
        "Turn this drawing into a finished, colored illustration of the same "
        "subject. Keep exactly what was drawn, with only the features the "
        "drawing already has. Keep the drawing's shape, proportions and "
        "composition. "
        f"Description: {prompt}. "
        "Style: cartoon illustration, flat colors, bold clean black outlines, "
        "children's drawing style. A single subject, centered in frame. "
        "Background: plain, solid pure white, nothing else, no shadows, no "
        "ground line."
    )
    if negative_prompt:
        instruction += f" Avoid: {negative_prompt}."
    return _edit_image(image_bytes, instruction,
                       model_id=model_id or config.FAL_RENDER_MODEL,
                       seed=seed)


#: Without explicit whole-subject framing, image models regularly crop in to a
#: head-and-shoulders portrait — which then has no arms or legs left to put
#: into a T-pose at all. Once that's happened there is no recovering it: a
#: later pass asking for "full body" on an already-cropped image just redraws
#: the same bust. So this has to hold at every stage that touches an avatar's
#: image — the free-text render in app.py's render_avatar() included, since
#: its output is what the T-pose transform below prefers as its source.
#: Kept subject-neutral on purpose: body words ("full body", "head to feet",
#: "portrait", "headshot") push the model to draw a person even when the
#: drawing and description are a flower, a car, etc.
FULL_BODY_HINT = (
    "the whole subject fully visible with nothing cut off at the edges, "
    "wide shot with some margin around it, not a close-up, not cropped"
)
FULL_BODY_NEGATIVE_HINT = "close-up, cropped, zoomed in, cut off at the edges"

#: Fixed instruction for the T-pose transform — never caller-supplied, since
#: this endpoint always wants the same thing: a clean, riggable reference pose.
#: Naming an illustration style explicitly and ruling out photorealism keeps
#: the drawn character (hair, face, expression) recognisable.
_TPOSE_PROMPT = (
    "Redraw this exact character standing in a T-pose: both arms held "
    "straight out to the sides at shoulder height, standing perfectly "
    "upright, facing directly forward, symmetrical, centered in frame. "
    "Keep the same character design — hair, face, clothing, colors, body parts. "
    f"Framing: {FULL_BODY_HINT}. Style: cartoon character illustration, "
    "flat colors, bold clean black outlines, children's drawing style. "
    "Background: plain, solid pure white, nothing else, no shadows, no "
    "ground line."
)
_TPOSE_NEGATIVE_PROMPT = (
    f"{FULL_BODY_NEGATIVE_HINT}, photograph, photorealistic, realistic "
    "human skin, real person, side view, back view, three-quarter view, "
    "sitting, crouching, hands on hips, arms down, multiple characters, "
    "patterned background, shadow, text, watermark, extra limbs, extra "
    "arms, extra hands, deformed hands, bad anatomy, blurry"
)


def pose_to_tshape(image_bytes: bytes) -> dict:
    """Redraw a drawing standing in a forward-facing T-pose on white.

    Stage 1 of the T-pose pipeline (see ``tpose_transform``). Uses a separate
    model setting from ``render_sketch`` so it can be tuned independently of
    the free-text render feature.
    """
    instruction = f"{_TPOSE_PROMPT} Avoid: {_TPOSE_NEGATIVE_PROMPT}."
    return _edit_image(image_bytes, instruction,
                       model_id=config.FAL_TPOSE_MODEL)


def remove_background(image_bytes: bytes) -> bytes:
    """Strip the background from an image, returning a PNG with a true alpha
    channel.

    Stage 2 of the T-pose pipeline (see ``tpose_transform``). A separate call
    from ``pose_to_tshape`` because the editor can only draw against a flat
    colour, never transparency.
    """
    result = _call(config.FAL_BG_REMOVAL_MODEL, {
        "image_url": _data_uri(image_bytes),
        "output_format": "png",
        "sync_mode": True,
    })
    if not result.get("image"):
        raise FalError("The background couldn't be removed.", status=502,
                       detail=f"no image from {config.FAL_BG_REMOVAL_MODEL}")
    return _fetch(result["image"])


def tpose_transform(image_bytes: bytes) -> dict:
    """Turn a drawing into a forward-facing, T-pose, transparent-background
    PNG: pose first, then strip the background it was drawn against."""
    posed = pose_to_tshape(image_bytes)
    transparent = remove_background(posed["image_bytes"])
    return {"image_bytes": transparent, "output_format": "png"}


def _translate(exc: Exception) -> FalError:
    """Map a fal-client exception to something safe to return.

    The caller sees a message written here; the original goes to the log.
    """
    name = type(exc).__name__
    status = getattr(exc, "status_code", None)

    known = {
        400: ("That request wasn't valid for the image model.", 400),
        401: ("The server's fal.ai key is invalid.", 503),
        403: ("The server's fal.ai account can't use the image model "
              "(check its balance).", 503),
        404: ("The server's image model isn't available.", 503),
        422: ("That request wasn't valid for the image model.", 400),
        429: ("The image service is busy. Try again in a moment.", 429),
        504: ("The image model took too long to reply.", 504),
    }
    if status in known:
        message, http_status = known[status]
    elif isinstance(exc, TimeoutError) or "timeout" in name.lower():
        message, http_status = "The image model took too long to reply.", 504
    else:
        message, http_status = "The image service failed. Try again?", 502

    log(f"[fal] {name}{f' {status}' if status else ''}: {exc}")
    return FalError(message, status=http_status, detail=f"{name}: {exc}")
