"""Cut the subject of a photo out of its background, keeping its colours.

For "Upload a photo" on the Draw step: a photo of a toy or a character should
arrive on the canvas as itself, minus the room behind it — not traced into
black lines, which is what the in-browser line pipeline does to a photo of a
drawing on paper.

Tries seg_server's /segment first (rembg; free, and the same service that
rigs the result later), then fal's BiRefNet if a fal key is configured. The
first that answers wins; neither failing loudly is fatal until both have.
"""

from __future__ import annotations

from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import config
import fal_images
from logs import log

#: Long enough for seg_server to load rembg's model on its first call.
TIMEOUT = 90.0

_PNG_MAGIC = b"\x89PNG"


class CutoutError(Exception):
    """No background remover worked. ``message`` is safe to show a child."""

    def __init__(self, message: str, status: int = 502):
        super().__init__(message)
        self.message = message
        self.status = status


def _seg_server(image_bytes: bytes) -> bytes:
    request = Request(
        f"{config.RIGGING_SERVICE_URL}/segment", data=image_bytes, method="POST",
        headers={"Content-Type": "application/octet-stream",
                 "ngrok-skip-browser-warning": "true"})
    with urlopen(request, timeout=TIMEOUT) as response:
        data = response.read()
    if not data.startswith(_PNG_MAGIC):
        raise ValueError(f"/segment returned {len(data)} bytes that aren't a PNG")
    return data


def remove_background(image_bytes: bytes) -> bytes:
    """Return a PNG of the photo's subject on a transparent background."""
    if config.RIGGING_SERVICE_URL:
        try:
            return _seg_server(image_bytes)
        except (HTTPError, URLError, OSError, ValueError) as exc:
            log(f"[cutout] seg_server /segment failed: {exc}")

    if config.FAL_KEY:
        try:
            return fal_images.remove_background(image_bytes)
        except fal_images.FalError as exc:
            log(f"[cutout] fal background removal failed: {exc.detail or exc}")

    raise CutoutError("I couldn't cut your photo out of its background. "
                      "Try again in a moment?")
