from __future__ import annotations

import pytest

import config
import pose_library
from providers.real.posing import RealPoser
from schemas import ARTICULATED_BONES, Clip, ProviderError, Rig, validate_clip

RIG = Rig()


def _validated_poses(name: str, keyframes, loop: bool) -> list[dict]:
    """Run raw pose_library output through the same validation RealPoser
    applies (which fills every non-articulated bone from rest), so it's
    comparable to what pose() actually returns."""
    clip = validate_clip(Clip(name=name, prompt="x", keyframes=keyframes, loop=loop))
    return [kf.pose for kf in clip.keyframes]


def noop(*_args) -> None:
    pass


def sequence_complete(responses: list[str]):
    """Fake complete_fn: pops one response per call, records every call's prompts."""
    remaining = list(responses)
    calls: list[tuple[str, str]] = []

    def complete_fn(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        return remaining.pop(0)

    complete_fn.calls = calls
    return complete_fn


def always_boom(*_args):
    raise AssertionError("the LLM must not be called for a library/cache hit")


def raising_complete(exc: Exception):
    def complete_fn(*_args):
        raise exc
    return complete_fn


GOOD_CODE = (
    "def animate(kf):\n"
    "    return [kf(0.0, spine=(10, 0, 0)), kf(0.5, spine=(-10, 0, 0))]\n"
    "LOOP = True\n"
)

DIFFERENT_GOOD_CODE = (
    "def animate(kf):\n"
    "    return [kf(0.0, neck=(5, 0, 0)), kf(0.6, neck=(-5, 0, 0))]\n"
    "LOOP = False\n"
)

FORBIDDEN_CODE_CASES = [
    "import os\ndef animate(kf):\n    return [kf(0.0)]\n",
    "def animate(kf):\n    open('/etc/passwd')\n    return [kf(0.0)]\n",
    "def animate(kf):\n    return [kf(0.0)]\nexec('1')\n",
    "def animate(kf):\n    return [kf(0.0)]\neval('1')\n",
    "def animate(kf):\n    return [kf(0.0)]\nx = ().__class__\n",
]

NON_ARTICULATED_BONE_CODE = (
    "def animate(kf):\n"
    "    return [kf(0.0, L_hand=(10, 0, 0))]\n"
)

HANGING_CODE = (
    "def animate(kf):\n"
    "    while True:\n"
    "        pass\n"
)


def test_static_library_prompt_never_calls_the_llm():
    poser = RealPoser(complete_fn=always_boom)
    clip = poser.pose("waves arms in the air", RIG, noop)
    name, keyframes, loop = pose_library.match("waves arms in the air")
    assert [kf.pose for kf in clip.keyframes] == _validated_poses(name, keyframes, loop)
    assert clip.loop == loop


def test_llm_path_returns_generated_pose_for_unmatched_prompt():
    complete_fn = sequence_complete([GOOD_CODE])
    poser = RealPoser(complete_fn=complete_fn)
    clip = poser.pose("does a cartwheel", RIG, noop)
    assert len(clip.keyframes) == 2
    assert clip.loop is True
    assert len(complete_fn.calls) == 1


def test_retries_with_feedback_on_invalid_generated_code():
    complete_fn = sequence_complete([NON_ARTICULATED_BONE_CODE, GOOD_CODE])
    poser = RealPoser(complete_fn=complete_fn)
    clip = poser.pose("does a cartwheel", RIG, noop)
    assert clip.loop is True
    assert len(complete_fn.calls) == 2
    second_user_prompt = complete_fn.calls[1][1]
    assert "L_hand" in second_user_prompt
    assert "Fix it." in second_user_prompt


def test_exhausted_retries_falls_back_to_hash_pose_not_error():
    text = "does a cartwheel"
    complete_fn = sequence_complete([NON_ARTICULATED_BONE_CODE] * 3)
    poser = RealPoser(complete_fn=complete_fn, retries=2)
    clip = poser.pose(text, RIG, noop)
    expected = _validated_poses("x", pose_library.fallback(text), True)
    assert [kf.pose for kf in clip.keyframes] == expected
    assert clip.loop is True
    assert len(complete_fn.calls) == 3


@pytest.mark.parametrize("bad_code", FORBIDDEN_CODE_CASES)
def test_sandbox_forbids_disallowed_constructs(bad_code):
    complete_fn = sequence_complete([bad_code, GOOD_CODE])
    poser = RealPoser(complete_fn=complete_fn)
    clip = poser.pose("does a cartwheel", RIG, noop)
    assert len(clip.keyframes) == 2
    assert "disallowed" in complete_fn.calls[1][1]


def test_sandbox_enforces_timeout():
    complete_fn = sequence_complete([HANGING_CODE, GOOD_CODE])
    poser = RealPoser(complete_fn=complete_fn, sandbox_timeout=0.2)
    clip = poser.pose("does a cartwheel", RIG, noop)
    assert len(clip.keyframes) == 2
    assert len(complete_fn.calls) == 2


def test_generated_code_cannot_set_non_articulated_bones():
    # L_hand is a valid contract bone (schemas.validate_pose fills every
    # untouched bone from rest, so the key is always present) but is NOT in
    # ARTICULATED_BONES — the sandbox must reject code that sets it, forcing
    # the retry onto GOOD_CODE, which never touches it. So L_hand should stay
    # at rest here, never at the (10, 0, 0) degree rotation the rejected
    # attempt tried to set.
    assert "L_hand" not in ARTICULATED_BONES
    complete_fn = sequence_complete([NON_ARTICULATED_BONE_CODE, GOOD_CODE])
    poser = RealPoser(complete_fn=complete_fn)
    clip = poser.pose("does a cartwheel", RIG, noop)
    for kf in clip.keyframes:
        assert kf.pose["L_hand"] == [0.0, 0.0, 0.0, 1.0]


def test_llm_result_is_cached_and_repeat_prompt_skips_the_llm():
    complete_fn = sequence_complete([GOOD_CODE])
    poser = RealPoser(complete_fn=complete_fn)
    a = poser.pose("does a cartwheel", RIG, noop)
    b = poser.pose("does a cartwheel", RIG, noop)
    assert len(complete_fn.calls) == 1
    assert a is not b
    assert a.id is None and b.id is None
    assert [kf.pose for kf in a.keyframes] == [kf.pose for kf in b.keyframes]


def test_is_deterministic_despite_llm_variance():
    complete_fn = sequence_complete([GOOD_CODE, DIFFERENT_GOOD_CODE])
    poser = RealPoser(complete_fn=complete_fn)
    a = poser.pose("does a cartwheel", RIG, noop)
    b = poser.pose("does a cartwheel", RIG, noop)
    assert [kf.pose for kf in a.keyframes] == [kf.pose for kf in b.keyframes]
    assert a.loop == b.loop


def test_empty_prompt_raises_provider_error():
    poser = RealPoser(complete_fn=always_boom)
    with pytest.raises(ProviderError, match="empty prompt") as caught:
        poser.pose("   ", RIG, noop)
    assert "Tell me what" in caught.value.user_message


@pytest.mark.parametrize("prompt,responses", [
    ("waves arms in the air", None),
    ("does a cartwheel", [GOOD_CODE]),
    ("does something with no match and bad code", [NON_ARTICULATED_BONE_CODE] * 3),
])
def test_progress_is_called_on_every_path(prompt, responses):
    complete_fn = sequence_complete(responses) if responses is not None else always_boom
    poser = RealPoser(complete_fn=complete_fn, retries=2)
    seen: list[tuple[float, str]] = []
    poser.pose(prompt, RIG, lambda fraction, message: seen.append((fraction, message)))
    assert seen
    assert all(0.0 <= fraction <= 1.0 for fraction, _ in seen)
    assert seen[-1][0] == pytest.approx(0.99)


def test_loop_flag_comes_from_generated_code():
    poser_true = RealPoser(complete_fn=sequence_complete([GOOD_CODE]))
    clip_true = poser_true.pose("does a cartwheel", RIG, noop)
    assert clip_true.loop is True

    poser_false = RealPoser(complete_fn=sequence_complete([DIFFERENT_GOOD_CODE]))
    clip_false = poser_false.pose("does a somersault", RIG, noop)
    assert clip_false.loop is False


def test_llm_transport_error_does_not_raise_and_falls_back():
    text = "does a cartwheel"
    poser = RealPoser(complete_fn=raising_complete(RuntimeError("connection refused")))
    clip = poser.pose(text, RIG, noop)
    expected = _validated_poses("x", pose_library.fallback(text), True)
    assert [kf.pose for kf in clip.keyframes] == expected


def test_no_api_key_and_no_injected_complete_fn_skips_llm_entirely(monkeypatch):
    monkeypatch.setattr(config, "OPENROUTER_API_KEY", "")
    text = "does a cartwheel"
    poser = RealPoser()
    clip = poser.pose(text, RIG, noop)
    expected = _validated_poses("x", pose_library.fallback(text), True)
    assert [kf.pose for kf in clip.keyframes] == expected
