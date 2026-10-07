"""Configuration. Everything is an env var with a sane default so the app runs
with no setup at all.

The three PROVIDER_* vars are the switches that swap mock implementations for
real ones — see CONTRACT.md.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent

# Local development defaults, from .env in the repo root.
#
# load_dotenv does not overwrite variables that are already set, so the
# precedence is: real environment > .env > the defaults below. That is what lets
# the same file work on Heroku — config vars arrive as real environment
# variables and win, and .env is not in the slug anyway.
#
# This has to run before the os.environ reads below. `flask run` also loads .env
# by itself once python-dotenv is installed, but gunicorn and pytest do not,
# which is why it is explicit here.
load_dotenv(ROOT / ".env")
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"

# "mock" | "real" — see providers/__init__.py
PROVIDER_RIGGING = os.environ.get("PROVIDER_RIGGING", "mock")
PROVIDER_POSING = os.environ.get("PROVIDER_POSING", "mock")
PROVIDER_TRAINING = os.environ.get("PROVIDER_TRAINING", "mock")

#: Server-to-server auto-rigging service. The browser never receives this URL.
RIGGING_SERVICE_URL = os.environ.get("RIGGING_SERVICE_URL", "").strip().rstrip("/")

#: Overall deadline for classification, T-pose augmentation, mesh generation,
#: joint inference, rigging, and the final GLB download.
RIGGING_SERVICE_TIMEOUT = float(
    os.environ.get("RIGGING_SERVICE_TIMEOUT", "300"))

#: Delay between remote mesh/rig task status checks.
RIGGING_POLL_INTERVAL = float(
    os.environ.get("RIGGING_POLL_INTERVAL", "5"))

#: Episodes per second pushed to the browser at speed 1.0. The training screen
#: multiplies this by its speed control.
EPISODE_RATE = float(os.environ.get("EPISODE_RATE", "20"))

#: Max upload size for a sketch.
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))

#: Path to a rigged GLB the mock rigger should serve instead of the procedural
#: figure. Lets the whole GLB path be exercised before a real rigger exists:
#:   MOCK_RIG_GLB=tests/fixtures/mixamo-style.glb flask --app app run
MOCK_RIG_GLB = os.environ.get("MOCK_RIG_GLB", "").strip()

#: How long a provider gets before the job runner gives up on it.
PROVIDER_TIMEOUT = float(os.environ.get("PROVIDER_TIMEOUT", "120"))


# --------------------------------------------------------------------------
# Real poser: LLM code-generation fallback (providers/real/posing.py)
# --------------------------------------------------------------------------
# The static keyword library (pose_library.py) handles common prompts for
# free; this only engages for everything else. An unset key disables the LLM
# path entirely — RealPoser falls back to the same deterministic hash-pose
# the mock uses, so the app (and the contract test suite, which instantiates
# RealPoser with no config) still works with zero setup.

#: OpenRouter API key. Empty disables the LLM fallback path entirely.
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()

#: OpenRouter's OpenAI-compatible endpoint.
OPENROUTER_BASE_URL = os.environ.get(
    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").strip()

#: Model id to request. See https://openrouter.ai/models for the catalogue.
OPENROUTER_MODEL = os.environ.get(
    "OPENROUTER_MODEL", "anthropic/claude-sonnet-5").strip()

#: Attempts given to the LLM before falling back to the hash pose: the first
#: try plus this many retries, each fed the previous attempt's validation
#: error so the model can fix it.
POSER_LLM_RETRIES = int(os.environ.get("POSER_LLM_RETRIES", "2"))


# --------------------------------------------------------------------------
# Claude prompt endpoint
# --------------------------------------------------------------------------
# Every default here is the locked-down one. The endpoint stays switched off
# and refuses every model until someone deliberately configures both.

#: Bearer token callers must present. Unset means the route returns 404.
#: This is a server-side secret — it must never be sent to the browser.
LLM_API_TOKEN = os.environ.get("LLM_API_TOKEN", "").strip()

#: Anthropic API key. Unset means every Claude call fails closed with a 503.
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()

#: Comma-separated Claude model IDs the endpoint may invoke. Empty refuses
#: everything, so a misconfigured deployment cannot be pointed at an
#: expensive model. e.g. "claude-opus-5-5,claude-sonnet-5-5,claude-haiku-4-5".
CLAUDE_ALLOWED_MODELS = tuple(
    m.strip() for m in os.environ.get("CLAUDE_ALLOWED_MODELS", "").split(",")
    if m.strip()
)

#: Ceiling on max_tokens, whatever the caller asks for.
CLAUDE_MAX_TOKENS = int(os.environ.get("CLAUDE_MAX_TOKENS", "4096"))

#: Seconds to wait for a model response.
CLAUDE_TIMEOUT = float(os.environ.get("CLAUDE_TIMEOUT", "60"))

#: Longest prompt accepted, in characters — by the prompt endpoint and the
#: sketch-render feature alike.
MAX_PROMPT_CHARS = int(os.environ.get("MAX_PROMPT_CHARS", "20000"))

#: Per-caller and whole-deployment request caps. The daily one bounds the bill.
LLM_RATE_PER_MINUTE = int(os.environ.get("LLM_RATE_PER_MINUTE", "10"))
LLM_RATE_PER_DAY = int(os.environ.get("LLM_RATE_PER_DAY", "500"))


# --------------------------------------------------------------------------
# fal.ai image features (POST /api/renders, POST /api/avatars/<id>/tpose)
# --------------------------------------------------------------------------
# Purpose-built, browser-facing endpoints — unlike the prompt endpoint above,
# the frontend is meant to call these. They take a fixed shape and always
# invoke the same models, so there's no caller-selectable model and therefore
# no allowlist. They fail closed the same way: no FAL_KEY means
# fal_images._get_client() refuses before anything runs.

#: fal.ai API key (https://fal.ai/dashboard/keys).
FAL_KEY = os.environ.get("FAL_KEY", "").strip()

#: Image editor for the "animated" sketch render: takes the drawing as a
#: reference image plus an instruction.
FAL_RENDER_MODEL = os.environ.get(
    "FAL_RENDER_MODEL", "fal-ai/flux-pro/kontext").strip()

#: Edge-conditioned (canny ControlNet) generator for the "lines" sketch
#: render: traces the drawing's strokes and colors them in. Must accept
#: fal's control_lora_image_url input.
FAL_LINES_MODEL = os.environ.get(
    "FAL_LINES_MODEL", "fal-ai/flux-control-lora-canny").strip()

#: Image editor for the T-pose redraw (stage 1). A separate setting so the
#: two features can diverge later.
FAL_TPOSE_MODEL = os.environ.get(
    "FAL_TPOSE_MODEL", "fal-ai/flux-pro/kontext").strip()

#: Background removal for the T-pose (stage 2), returning a PNG with a real
#: alpha channel.
FAL_BG_REMOVAL_MODEL = os.environ.get(
    "FAL_BG_REMOVAL_MODEL", "fal-ai/birefnet/v2").strip()

#: Seconds to wait for each fal call, queue time included.
FAL_TIMEOUT = float(os.environ.get("FAL_TIMEOUT", "120"))

#: These endpoints have no bearer token — every visitor's browser can reach
#: them, like the rest of the avatar API — so they need their own caps to
#: bound the bill. Tighter than LLM_RATE_PER_* because image generation costs
#: more per call than a short text completion; T-pose is tighter still since
#: each request is two fal calls.
RENDER_RATE_PER_MINUTE = int(os.environ.get("RENDER_RATE_PER_MINUTE", "5"))
RENDER_RATE_PER_DAY = int(os.environ.get("RENDER_RATE_PER_DAY", "50"))
TPOSE_RATE_PER_MINUTE = int(os.environ.get("TPOSE_RATE_PER_MINUTE", "3"))
TPOSE_RATE_PER_DAY = int(os.environ.get("TPOSE_RATE_PER_DAY", "30"))


def ensure_dirs() -> None:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
