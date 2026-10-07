"""Shared move vocabulary for both posers: keyword-matched, hand-authored
clips as a cheap deterministic first pass, plus a hash-seeded fallback so an
unmatched prompt still produces something. Used by providers/mock/posing.py
directly, and by providers/real/posing.py as its fast path before it falls
back to LLM code-generation for anything this library doesn't cover.

Neither providers/mock nor providers/real imports the other (see
CONTRACT.md rule 2) — this module is the neutral home both import from
instead.
"""

from __future__ import annotations

import hashlib

from rewards import quat_from_euler
from schemas import ARTICULATED_BONES, Keyframe

#: Rotation conventions for this skeleton, reused verbatim as domain
#: knowledge in both the hand-authored builders below and in
#: providers/real/posing.py's LLM system prompt — do not re-derive these,
#: they were worked out and verified by hand once already.
ROTATION_NOTES = """\
ARMS. The left arm rests along -X, the right along +X.
  raise:        L_shoulder z = -90 is straight up;  R_shoulder z = +90
                (past 90 the arm crosses the midline, so stay under ~110)
  swing fwd:    rotate about Y. +Y swings the LEFT arm forward and the
                RIGHT arm backward, which is what you want for a walk.
  (rotating an arm about X does nothing — it is already the X axis.)

LEGS. Both legs rest along -Y.
  swing fwd:    NEGATIVE x rotation (-30 = thigh forward)
  knee bend:    POSITIVE x rotation (knees bend backwards)
  spread out:   L_hip z = -25 opens left, R_hip z = +25 opens right

SPINE. +X leans forward, z tilts side to side.

All angles are degrees, applied XYZ.
"""


def kf(t: float, **bones_degrees) -> Keyframe:
    """Author a keyframe as `bone=(x_deg, y_deg, z_deg)`."""
    return Keyframe(t=t, pose={bone: list(quat_from_euler(*angles))
                               for bone, angles in bones_degrees.items()})


def wave() -> list[Keyframe]:
    """Right arm up, hand swinging side to side."""
    return [
        kf(0.0, R_shoulder=(0, 0, 85), R_elbow=(0, 0, 30)),
        kf(0.4, R_shoulder=(0, 0, 95), R_elbow=(0, 0, -20)),
        kf(0.8, R_shoulder=(0, 0, 85), R_elbow=(0, 0, 30)),
        kf(1.2, R_shoulder=(0, 0, 95), R_elbow=(0, 0, -20)),
        kf(1.6, R_shoulder=(0, 0, 85), R_elbow=(0, 0, 30)),
    ]


def arms_up() -> list[Keyframe]:
    """Both arms overhead, waving. The canonical 'waves arms in the air'."""
    return [
        kf(0.0, L_shoulder=(0, 0, -80), R_shoulder=(0, 0, 80),
           L_elbow=(0, 0, -15), R_elbow=(0, 0, 15)),
        kf(0.5, L_shoulder=(0, 0, -65), R_shoulder=(0, 0, 95),
           L_elbow=(0, 0, -25), R_elbow=(0, 0, 25), spine=(0, 0, 6)),
        kf(1.0, L_shoulder=(0, 0, -95), R_shoulder=(0, 0, 65),
           L_elbow=(0, 0, -25), R_elbow=(0, 0, 25), spine=(0, 0, -6)),
        kf(1.5, L_shoulder=(0, 0, -80), R_shoulder=(0, 0, 80),
           L_elbow=(0, 0, -15), R_elbow=(0, 0, 15)),
    ]


def jump() -> list[Keyframe]:
    return [
        # crouch: thighs forward, knees bent, lean in
        kf(0.0, L_hip=(-35, 0, 0), R_hip=(-35, 0, 0),
           L_knee=(70, 0, 0), R_knee=(70, 0, 0), spine=(20, 0, 0)),
        # launch: straighten out, arms up
        kf(0.35, L_shoulder=(0, 0, -75), R_shoulder=(0, 0, 75),
           L_knee=(5, 0, 0), R_knee=(5, 0, 0)),
        kf(0.7, L_shoulder=(0, 0, -95), R_shoulder=(0, 0, 95)),
        # land
        kf(1.1, L_hip=(-30, 0, 0), R_hip=(-30, 0, 0),
           L_knee=(60, 0, 0), R_knee=(60, 0, 0), spine=(15, 0, 0)),
        kf(1.5),
    ]


def dance() -> list[Keyframe]:
    return [
        kf(0.0, L_shoulder=(0, 0, -100), R_shoulder=(0, 0, 30),
           spine=(0, 0, 10), L_hip=(-20, 0, 0)),
        kf(0.4, L_shoulder=(0, 0, -30), R_shoulder=(0, 0, 100),
           spine=(0, 0, -10), R_hip=(-20, 0, 0)),
        kf(0.8, L_shoulder=(0, 0, -100), R_shoulder=(0, 0, 30),
           spine=(0, 0, 10), L_hip=(-20, 0, 0)),
        kf(1.2, L_shoulder=(0, 0, -30), R_shoulder=(0, 0, 100),
           spine=(0, 0, -10), R_hip=(-20, 0, 0)),
        kf(1.6, L_shoulder=(0, 0, -100), R_shoulder=(0, 0, 30),
           spine=(0, 0, 10)),
    ]


def walk() -> list[Keyframe]:
    return [
        # left leg forward, right leg trailing with a bent knee; arms opposite
        kf(0.0, L_hip=(-30, 0, 0), R_hip=(30, 0, 0), R_knee=(25, 0, 0),
           L_shoulder=(0, 25, 0), R_shoulder=(0, 25, 0)),
        kf(0.5, L_hip=(30, 0, 0), R_hip=(-30, 0, 0), L_knee=(25, 0, 0),
           L_shoulder=(0, -25, 0), R_shoulder=(0, -25, 0)),
        kf(1.0, L_hip=(-30, 0, 0), R_hip=(30, 0, 0), R_knee=(25, 0, 0),
           L_shoulder=(0, 25, 0), R_shoulder=(0, 25, 0)),
    ]


def t_pose() -> list[Keyframe]:
    return [kf(0.0)]


def star_jump() -> list[Keyframe]:
    return [
        kf(0.0),
        # arms and legs out on the diagonal
        kf(0.4, L_shoulder=(0, 0, -45), R_shoulder=(0, 0, 45),
           L_hip=(0, 0, -25), R_hip=(0, 0, 25)),
        kf(0.8),
    ]


def bow() -> list[Keyframe]:
    return [
        kf(0.0),
        kf(0.6, spine=(60, 0, 0), neck=(20, 0, 0)),
        kf(1.4, spine=(60, 0, 0), neck=(20, 0, 0)),
        kf(2.0),
    ]


#: keyword-tuple -> (clip name, keyframe builder, loop). First match wins,
#: so more specific phrases are listed before more general ones.
LIBRARY: list[tuple[tuple[str, ...], str, callable, bool]] = [
    (("arms in air", "arms up", "both arms", "reach up", "hands up",
      "arms in the air"), "Arms in the air", arms_up, True),
    (("star jump", "jumping jack"), "Star jump", star_jump, False),
    (("wave", "waving", "hello", "hi ", "greet"), "Wave", wave, True),
    (("jump", "hop", "leap"), "Jump", jump, True),
    (("dance", "dancing", "boogie", "wiggle"), "Dance", dance, True),
    (("walk", "walking", "step", "march"), "Walk", walk, True),
    (("bow", "bowing", "thank you"), "Bow", bow, False),
    (("t-pose", "t pose", "stand", "neutral", "rest", "still"), "Stand", t_pose, False),
]


def match(prompt_lower: str) -> tuple[str, list[Keyframe], bool] | None:
    """First-match-wins keyword lookup. Returns (name, keyframes, loop) or None."""
    for keywords, name, builder, loop in LIBRARY:
        if any(word in prompt_lower for word in keywords):
            return name, builder(), loop
    return None


def fallback(prompt: str) -> list[Keyframe]:
    """Deterministic nonsense pose derived from the prompt.

    Same words always give the same pose, which matters: a child typing the same
    thing twice and getting a different avatar would undermine the whole lesson
    about inputs mapping to outputs.
    """
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    keyframes = []
    for k in range(3):
        bones = {}
        for i, bone in enumerate(ARTICULATED_BONES):
            byte = digest[(i + k * 7) % len(digest)]
            # -70..70 degrees, mostly around the Z axis so it reads as a pose
            angle = (byte / 255.0) * 140.0 - 70.0
            bones[bone] = (angle * 0.3, 0.0, angle)
        keyframes.append(kf(k * 0.6, **bones))
    return keyframes
