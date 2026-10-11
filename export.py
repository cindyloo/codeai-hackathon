"""Turn a finished training run into something the next stage can consume.

Two artifacts from one place, because they have different audiences:

  build_glb()      the avatar's GLB carrying a looping T-pose -> learned
                   pose -> T-pose animation, with both poses and the training
                   metadata embedded in asset.extras. For rendering, for
                   downstream 3D tools, for a human to open — and for loading
                   back into the app, which imports its animations as moves.

  build_document() a small JSON document — bones, start pose, end pose,
                   provenance. This is what the animation step should be given.

DO NOT SEND THE GLB TO AN LLM. A real Meshy export is ~3.6MB, which is ~5M
characters base64'd, and essentially all of it is mesh, textures and skin
weights that no language model can act on. The JSON document is a few hundred
tokens and carries every fact the animation step actually needs.
"""

from __future__ import annotations

from typing import Any

import gltf
from schemas import (
    ARTICULATED_BONES,
    BONE_TREE,
    BONES,
    MIXAMO_BONE_MAP,
    REST_POSE,
)

#: Bump when the shape of the exported document changes, so a consumer written
#: against an older export can say so rather than misread it.
SCHEMA_VERSION = 1

#: Seconds into the baked animation at which each pose is reached: ease out of
#: the T-pose, hold the learned pose, ease back so the clip loops seamlessly.
ANIMATION_TIMING = (("start", 0.0), ("end", 1.0), ("end", 2.0), ("start", 3.0))


class ExportError(Exception):
    """Export cannot proceed. ``user_message`` is safe to show a child."""

    def __init__(self, user_message: str, status: int = 400, detail: str = ""):
        super().__init__(detail or user_message)
        self.user_message = user_message
        self.status = status
        self.detail = detail


def final_pose(run) -> dict[str, list[float]]:
    """The pose the learner ended on.

    The hill-climber only ever accepts improvements, so the last episode's pose
    is also the best one. Taken from the end of history rather than tracked
    separately so that a run stopped early still exports what is on screen.
    """
    history = getattr(run, "_history", None)
    if not history:
        raise ExportError("That run hasn't produced a pose yet.", 409,
                          "no episodes in run history")
    return history[-1].pose


def build_document(run, avatar, clip) -> dict[str, Any]:
    """The compact, LLM-ready description of what was learned."""
    end = final_pose(run)
    last = run._history[-1]

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.id,
        "avatar_id": avatar.id,

        # The vocabulary. A consumer needs no other file to interpret the poses.
        "skeleton": {
            "bones": list(BONES),
            "articulated": list(ARTICULATED_BONES),
            "hierarchy": {b: BONE_TREE[b][0] for b in BONES},
            "rest_offsets": {b: list(BONE_TREE[b][1]) for b in BONES},
            "pose_format": ("bone name -> local rotation relative to rest, "
                            "quaternion [x, y, z, w]"),
        },

        # The two ends of the animation the next step has to fill in.
        "poses": {
            "start": {b: list(q) for b, q in REST_POSE.items()},
            "end": {b: list(q) for b, q in end.items()},
        },
        "start_pose_note": ("T-pose: every bone at identity. This is both our "
                            "rest pose and the GLB's bind pose, so the two "
                            "skeletons agree at the start of any animation."),

        # Why this pose — the intent an animator would want.
        "target": {
            "name": clip.name,
            "prompt": clip.prompt,
        },

        "training": {
            "episodes": run.cfg.episodes,
            "episodes_run": last.episode,
            "best_reward": round(run.best_reward, 4),
            "match": round(last.match, 4),
            "reward_weights": dict(run.cfg.reward_weights),
        },
    }


def build_glb(run, avatar, clip) -> tuple[bytes, dict[str, Any]]:
    """The avatar's GLB with the learned move baked in as an animation.

    Returns (bytes, document). The animation moves from the START pose (the
    file's bind T-pose) to the END pose and back. The node rotations stay at
    bind: anything that reads a rig — this app included, when the file is
    loaded back — takes them as the rest pose, so baking the end pose into
    them would offset every later move by it.
    """
    if avatar.rig.format != "glb" or not avatar.rig.glb_bytes:
        raise ExportError(
            "This avatar is drawn as a stick figure, so there's no model to "
            "export. Give it a rigged body first.",
            409, f"rig.format={avatar.rig.format!r}")

    try:
        document, binary = gltf.read_glb(avatar.rig.glb_bytes)
    except gltf.GlbError as exc:
        raise ExportError("That avatar's model file couldn't be read.", 500,
                          str(exc)) from exc

    bones = gltf.resolve_bones(document)
    if not bones:
        raise ExportError(
            "That avatar's model has no bones I recognise, so it can't be posed.",
            409, "no contract bones resolved in the GLB")

    payload = build_document(run, avatar, clip)

    # Keep only the GLB's own animations the avatar still has as moves: one
    # deleted on Play (an old "(learned)" bake, say) stays out of this file.
    animations = document.get("animations") or []
    keep = set(avatar.model_animations.values())
    document["animations"] = [a for i, a in enumerate(animations) if i in keep]
    payload["kept_animations"] = [a.get("name", "") for a in document["animations"]]

    # Named as "Save what it learned" names it, so loading the file back
    # doesn't give two moves called "Walk" when the rig brought its own.
    name = f"{clip.name} (learned)"
    keyframes = [(t, payload["poses"][which]) for which, t in ANIMATION_TIMING]
    binary = gltf.add_pose_animation(document, binary, keyframes, bones,
                                     name=name)

    # Record what actually moved. A rig missing bones still exports, but the
    # consumer should be able to tell which joints were driven and which were
    # left at bind rather than inferring it from the geometry.
    nodes = document.get("nodes", [])
    payload["node_map"] = {
        bone: nodes[index].get("name", f"node[{index}]")
        for bone, index in sorted(bones.items())
    }
    payload["unmapped_bones"] = [b for b in BONES if b not in bones]
    payload["bone_aliases"] = {"mixamo": dict(MIXAMO_BONE_MAP)}
    payload["posed"] = "start"
    payload["animation"] = {
        "name": name,
        "duration": ANIMATION_TIMING[-1][1],
        "keyframes": [{"t": t, "pose": which} for which, t in ANIMATION_TIMING],
    }

    gltf.attach_extras(document, payload)
    return gltf.write_glb(document, binary), payload
