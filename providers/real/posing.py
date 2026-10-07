"""Real poser: static library first, LLM code-generation for everything else.

Unlike the mock, which only ever keyword-matches a prompt against ~8 hand
authored clips (falling back to hash noise for anything else), this actually
tries to satisfy an arbitrary prompt: the same cheap static library runs
first (pose_library.match — free, deterministic, covers most real usage),
and only a prompt that misses it reaches the LLM. Asking the model to emit
raw per-joint keyframe numbers directly turns out to be unreliable — this
repo already learned that lesson twice, once in seg_server's own comments
and once in this repo's separate animator/llm_animator.py tool — so instead
the LLM writes a short Python function against one constrained helper
(pose_library.kf), which is executed in a sandbox, validated, and retried
with the exact error fed back on failure.

This domain is simpler than animator/'s: schemas.REST_POSE is identity for
every bone, so there is no Mixamo-style arbitrary rest rotation to convert
around — a bone's Euler degrees map straight to its local rotation via
pose_library.kf, no world-space delta math needed. If the LLM path is
unavailable (no OPENROUTER_API_KEY) or exhausts its retries, this falls back
to the same deterministic hash-pose the mock uses rather than raising —
providers.get_poser() must keep working with zero configuration, since
tests/test_contract.py instantiates it with no network access.
"""

from __future__ import annotations

import copy
import re
import threading
from typing import Any, Callable

import config
import pose_library
from providers.base import Poser, Progress
from schemas import (
    ARTICULATED_BONES,
    Clip,
    ContractError,
    Keyframe,
    ProviderError,
    Rig,
    validate_clip,
)

#: A leaked infinite loop in generated code should die fast, independent of
#: the LLM network budget (config.PROVIDER_TIMEOUT) — these bound different
#: failure modes and shouldn't share a number.
_SANDBOX_TIMEOUT_SECONDS = 10.0

#: Same shape as animator/llm_animator.py's blocklist: no dunders, no
#: import/open/exec/eval, no os/sys/subprocess access from generated code.
_FORBIDDEN = re.compile(r"__|import\s|open\s*\(|exec\s*\(|eval\s*\(|\bos\.|\bsys\.|subprocess")

#: __builtins__ is replaced wholesale with this, not merged — anything not
#: listed here (open, __import__, ...) is simply unavailable.
_SANDBOX_BUILTINS = {
    "range": range, "len": len, "list": list, "dict": dict, "tuple": tuple,
    "float": float, "int": int, "min": min, "max": max, "abs": abs,
    "enumerate": enumerate, "zip": zip, "True": True, "False": False, "None": None,
}

_STAGE_START = (0.05, "Reading your words...")
_STAGE_THINKING = (0.35, "Imagining the pose...")
_STAGE_DONE = (0.99, "Moving the joints...")

SYSTEM_PROMPT = f"""\
You write a short Python function that poses a rigged 3D character by
setting joint rotations. You are given exactly one helper — use ONLY it, do
not import anything or access the filesystem/network:

- `kf(t, **bone=(x_deg, y_deg, z_deg))` — builds one keyframe at time `t`
  (seconds), rotating each named bone by the given Euler degrees, relative
  to its rest pose. Omit a bone to leave it at rest for that keyframe.

Only these bone names are valid: {", ".join(ARTICULATED_BONES)}.

{pose_library.ROTATION_NOTES}
Write exactly one function:

    def animate(kf):
        return [
            kf(0.0, ...),
            kf(0.4, ...),
            ...
        ]

Optionally set a module-level `LOOP = True` if the motion should repeat
(a walk, wave, dance); omit it (or set `LOOP = False`) for a one-shot pose
like a bow or a stand.

Worked example — a two-armed wave:

    def animate(kf):
        return [
            kf(0.0, L_shoulder=(0, 0, -80), R_shoulder=(0, 0, 80),
               L_elbow=(0, 0, -15), R_elbow=(0, 0, 15)),
            kf(0.5, L_shoulder=(0, 0, -65), R_shoulder=(0, 0, 95),
               L_elbow=(0, 0, -25), R_elbow=(0, 0, 25), spine=(0, 0, 6)),
            kf(1.0, L_shoulder=(0, 0, -80), R_shoulder=(0, 0, 80),
               L_elbow=(0, 0, -15), R_elbow=(0, 0, 15)),
        ]
    LOOP = True

Rules:
- times must start at or after 0 and strictly increase between keyframes.
- keep the whole clip under a few seconds unless the action clearly needs
  more (a full dance can run longer; a wave or bow should not).
- only use the bone names listed above — no hands, feet, hips, or anything
  else; those follow automatically.
- Output ONLY the function definition (and LOOP=... if you set it). No
  markdown fences, no example call, no prose.
"""


class GenerationError(RuntimeError):
    """The LLM's code failed to parse, run, or produce a usable clip."""


def _extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return fence.group(1).strip() if fence else text.strip()


def _default_complete(system_prompt: str, user_prompt: str) -> str:
    from openai import OpenAI

    try:
        client = OpenAI(base_url=config.OPENROUTER_BASE_URL,
                        api_key=config.OPENROUTER_API_KEY,
                        timeout=config.PROVIDER_TIMEOUT)
        response = client.chat.completions.create(
            model=config.OPENROUTER_MODEL,
            temperature=0,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": user_prompt}],
        )
        return response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001 - any transport failure is retryable
        raise GenerationError(f"LLM request failed: {exc}") from exc


def _call_with_timeout(fn: Callable[..., Any], args: tuple, seconds: float) -> Any:
    """Run fn(*args) with a wall-clock budget, on a thread the process never waits on.

    A ThreadPoolExecutor looks like the obvious tool here, but isn't: its
    worker threads are non-daemon, and CPython's atexit machinery joins every
    ThreadPoolExecutor thread at interpreter shutdown regardless of
    shutdown(wait=False) — so a truly hung `animate()` (an infinite loop that
    never checks anything) would hang the whole server process at shutdown,
    not just this one request. A plain daemon Thread has no such join: the
    process can exit immediately no matter how long the generated code runs.
    """
    result: dict[str, Any] = {}

    def run() -> None:
        try:
            result["value"] = fn(*args)
        except Exception as exc:  # noqa: BLE001 - re-raised on the caller's thread
            result["error"] = exc

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout=seconds)
    if thread.is_alive():
        raise TimeoutError(f"generated code exceeded {seconds}s")
    if "error" in result:
        raise result["error"]
    return result.get("value")


def _validate_keyframes(keyframes: Any) -> list[Keyframe]:
    if not isinstance(keyframes, list) or not keyframes:
        raise GenerationError(
            f"animate() must return a non-empty list of keyframes, got {keyframes!r}")

    last_t = -1.0
    for i, kf_obj in enumerate(keyframes):
        if not isinstance(kf_obj, Keyframe):
            raise GenerationError(
                f"element {i} of animate()'s return value is not a keyframe "
                f"(build them with kf(...)): {kf_obj!r}")
        if not isinstance(kf_obj.t, (int, float)) or kf_obj.t < 0:
            raise GenerationError(f"keyframe {i} has an invalid time: {kf_obj.t!r}")
        if kf_obj.t <= last_t:
            raise GenerationError(
                f"keyframe {i} at t={kf_obj.t} does not strictly increase after "
                f"the previous keyframe at t={last_t}")
        last_t = kf_obj.t

        unknown = sorted(set(kf_obj.pose) - set(ARTICULATED_BONES))
        if unknown:
            raise GenerationError(
                f"keyframe {i} sets bone(s) {unknown} — only these bones are "
                f"allowed: {list(ARTICULATED_BONES)}")

    return keyframes


def _run_generated_code(code: str, sandbox_timeout: float) -> tuple[list[Keyframe], bool]:
    if _FORBIDDEN.search(code):
        raise GenerationError(f"generated code used a disallowed construct:\n{code}")

    sandbox_globals: dict[str, Any] = {
        "__builtins__": _SANDBOX_BUILTINS,
        "kf": pose_library.kf,
    }
    try:
        exec(code, sandbox_globals)  # noqa: S102 - sandboxed by _FORBIDDEN + restricted builtins
    except Exception as exc:  # noqa: BLE001 - any parse/exec failure is retryable
        raise GenerationError(
            f"generated code failed to parse/execute:\n{code}\n\nerror: {exc}") from exc

    animate_fn = sandbox_globals.get("animate")
    if not callable(animate_fn):
        raise GenerationError(f"generated code did not define animate():\n{code}")

    try:
        keyframes = _call_with_timeout(animate_fn, (pose_library.kf,), sandbox_timeout)
    except TimeoutError as exc:
        raise GenerationError(f"animate() timed out:\n{code}\n\nerror: {exc}") from exc
    except Exception as exc:  # noqa: BLE001 - any runtime failure is retryable
        raise GenerationError(
            f"animate() raised while running:\n{code}\n\nerror: {exc!r}") from exc

    keyframes = _validate_keyframes(keyframes)
    loop = bool(sandbox_globals.get("LOOP", False))
    return keyframes, loop


def _clone(clip: Clip, prompt: str) -> Clip:
    """A fresh, independently mutable Clip built from a cached template.

    store.add_clip() assigns clip.id in place and stores the exact object by
    reference — handing back the cached instance itself would let a second
    request for the same prompt silently share one mutable store entry with
    the first.
    """
    return Clip(name=clip.name, prompt=prompt,
                keyframes=copy.deepcopy(clip.keyframes), loop=clip.loop)


class RealPoser(Poser):
    def __init__(
        self,
        complete_fn: Callable[[str, str], str] | None = None,
        sandbox_timeout: float = _SANDBOX_TIMEOUT_SECONDS,
        retries: int | None = None,
    ) -> None:
        self._complete_fn = complete_fn or (
            _default_complete if config.OPENROUTER_API_KEY else None)
        self._sandbox_timeout = sandbox_timeout
        self._retries = config.POSER_LLM_RETRIES if retries is None else retries
        self._cache: dict[str, Clip] = {}

    def pose(self, prompt: str, rig: Rig, progress: Progress) -> Clip:
        text = (prompt or "").strip().lower()
        if not text:
            raise ProviderError("Tell me what you'd like your avatar to do!",
                                detail="empty prompt")

        progress(*_STAGE_START)

        if text in self._cache:
            progress(*_STAGE_DONE)
            return _clone(self._cache[text], prompt)

        matched = pose_library.match(text)
        if matched is not None:
            name, keyframes, loop = matched
            clip = validate_clip(Clip(name=name, prompt=prompt,
                                      keyframes=keyframes, loop=loop))
        else:
            clip = self._generate_with_llm(text, prompt, progress)

        self._cache[text] = clip
        progress(*_STAGE_DONE)
        return _clone(clip, prompt)

    def _generate_with_llm(self, text: str, prompt: str, progress: Progress) -> Clip:
        if self._complete_fn is None:
            return self._hash_fallback_clip(text, prompt)

        progress(*_STAGE_THINKING)
        user_prompt = f'Action to animate: "{prompt}"'
        last_error: str | None = None

        for _ in range(self._retries + 1):
            if last_error is not None:
                user_prompt += f"\n\nYour previous attempt failed with:\n{last_error}\nFix it."
            try:
                raw = self._complete_fn(SYSTEM_PROMPT, user_prompt)
                code = _extract_code(raw)
                keyframes, loop = _run_generated_code(code, self._sandbox_timeout)
                name = prompt[:40].strip().capitalize() or "Made-up move"
                return validate_clip(Clip(name=name, prompt=prompt,
                                          keyframes=keyframes, loop=loop))
            except (GenerationError, ContractError) as exc:
                last_error = str(exc)
            except Exception as exc:  # noqa: BLE001 - an injected/custom complete_fn
                # can fail in ways _default_complete already wraps for us; treat
                # any other failure here as retryable too rather than letting it
                # escape as a raw exception.
                last_error = f"the LLM call failed: {exc}"

        return self._hash_fallback_clip(text, prompt)

    def _hash_fallback_clip(self, text: str, prompt: str) -> Clip:
        name = prompt[:40].strip().capitalize() or "Made-up move"
        return validate_clip(Clip(name=name, prompt=prompt,
                                  keyframes=pose_library.fallback(text), loop=True))
