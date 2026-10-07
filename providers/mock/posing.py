"""Mock poser: keyword-matches the prompt against hand-authored clips.

An unknown prompt still produces *something* — a deterministic pose seeded from
the prompt text — so a live demo never dead-ends in front of a classroom.

See providers/real/posing.py for the real implementation.
"""

from __future__ import annotations

import time

import pose_library
from providers.base import Poser, Progress
from schemas import Clip, ProviderError, Rig

_STAGES = [
    (0.2, "Reading your words..."),
    (0.55, "Imagining the pose..."),
    (0.85, "Moving the joints..."),
]


class MockPoser(Poser):
    def pose(self, prompt: str, rig: Rig, progress: Progress) -> Clip:
        text = (prompt or "").strip().lower()
        if not text:
            raise ProviderError("Tell me what you'd like your avatar to do!",
                                detail="empty prompt")

        for fraction, message in _STAGES:
            progress(fraction, message)
            time.sleep(0.3)

        matched = pose_library.match(text)
        if matched is not None:
            name, keyframes, loop = matched
            return Clip(name=name, prompt=prompt, keyframes=keyframes, loop=loop)

        return Clip(name=prompt[:40].strip().capitalize() or "Made-up move",
                    prompt=prompt, keyframes=pose_library.fallback(text), loop=True)
