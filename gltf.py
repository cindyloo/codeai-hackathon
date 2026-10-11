"""Read a GLB, write a posed one back.

Two jobs:

  1. GLB container surgery — parse the header/chunks, edit the JSON, repack.
     Deliberately hand-rolled rather than pulling in pygltflib or trimesh: we
     touch node rotations, `extras` and appended animation data, never
     geometry, so the whole thing is small and adds nothing to the slug.

  2. The same world-space retarget the browser does, in Python. A contract pose
     is a local rotation per bone in OUR frame, where every bone binds at
     identity; a GLB bone's local frame is whatever its exporter produced. See
     `pose_glb` for the maths, and static/js/viewport.js #applyPoseGlb for the
     browser twin.

WARNING: that twin is a genuine duplication. If you change the retarget here,
change it there — tests/test_export.py pins the shapes but cannot see the
browser. The long-term fix is to compute poses once, server-side, and have the
viewport apply what it is given.
"""

from __future__ import annotations

import json
import math
import struct
from typing import Any

from rewards import quat_mul
from schemas import BONE_TREE, IDENTITY_QUAT, MIXAMO_BONE_MAP, OPTIONAL_BONES

_MAGIC = 0x46546C67          # 'glTF'
_CHUNK_JSON = 0x4E4F534A     # 'JSON'
_CHUNK_BIN = 0x004E4942      # 'BIN\0'


class GlbError(ValueError):
    """The bytes we were handed are not a GLB we can work with."""


class GlbNotRiggedError(GlbError):
    """A readable GLB with no skin: a mesh that could still be auto-rigged."""


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------

def read_glb(data: bytes) -> tuple[dict[str, Any], bytes]:
    """Split a GLB into its JSON document and binary buffer."""
    if len(data) < 12:
        raise GlbError("too short to be a GLB")
    magic, version, length = struct.unpack_from("<III", data, 0)
    if magic != _MAGIC:
        raise GlbError("not a GLB (bad magic — a .gltf JSON file needs "
                       "no unpacking, but this code path expects binary)")
    if version != 2:
        raise GlbError(f"unsupported glTF version {version}")
    if length < 12 or length > len(data):
        raise GlbError(
            f"invalid GLB length header ({length} bytes for a {len(data)} byte file)")

    document: dict[str, Any] | None = None
    binary = b""
    offset = 12
    # Trust the header length over len(data): some exporters append junk.
    end = min(length, len(data))
    while offset + 8 <= end:
        chunk_len, chunk_type = struct.unpack_from("<II", data, offset)
        start = offset + 8
        if start + chunk_len > end:
            raise GlbError("truncated GLB chunk")
        chunk = data[start:start + chunk_len]
        if chunk_type == _CHUNK_JSON:
            try:
                parsed = json.loads(chunk.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GlbError(f"invalid GLB JSON: {exc}") from exc
            if not isinstance(parsed, dict):
                raise GlbError("GLB JSON root is not an object")
            document = parsed
        elif chunk_type == _CHUNK_BIN:
            binary = chunk
        offset = start + chunk_len + (-chunk_len % 4)   # chunks are 4-aligned

    if document is None:
        raise GlbError("GLB has no JSON chunk")
    return document, binary


def write_glb(document: dict[str, Any], binary: bytes) -> bytes:
    """Repack a JSON document and buffer into GLB bytes."""
    json_bytes = json.dumps(document, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * (-len(json_bytes) % 4)         # JSON pads with spaces
    binary = binary + b"\x00" * (-len(binary) % 4)      # BIN pads with zeros

    total = 12 + 8 + len(json_bytes) + (8 + len(binary) if binary else 0)
    out = bytearray()
    out += struct.pack("<III", _MAGIC, 2, total)
    out += struct.pack("<II", len(json_bytes), _CHUNK_JSON) + json_bytes
    if binary:
        out += struct.pack("<II", len(binary), _CHUNK_BIN) + binary
    return bytes(out)


# --------------------------------------------------------------------------
# Bone resolution — mirrors normaliseBoneName in static/js/viewport.js
# --------------------------------------------------------------------------

def normalise_bone_name(name: str) -> str:
    """Reduce a bone name to something comparable across exporters.

    A bone authored as "mixamorig:LeftArm" has been seen as `mixamorigLeftArm`
    (three's sanitizeNodeName), `mixamorig_LeftArm` (FBX conversion) and
    `mixamorig_LeftArm_011` (exporter index). Drop one trailing _<digits>, then
    every separator, then lowercase.

    The index is dropped in a SINGLE pass on purpose — stripping all trailing
    digits would fold "Spine_02" and "Spine1_03" onto one key.
    """
    if not name:
        return ""
    trimmed = name
    if "_" in trimmed:
        head, _, tail = trimmed.rpartition("_")
        if head and tail.isdigit():
            trimmed = head
    return "".join(c for c in trimmed if c.isalnum()).lower()


def resolve_bones(document: dict[str, Any]) -> dict[str, int]:
    """Map each contract bone to a glTF node index."""
    nodes = document.get("nodes", [])
    by_name: dict[str, int] = {}
    for index, node in enumerate(nodes):
        name = node.get("name")
        if not name:
            continue
        by_name.setdefault(name, index)
        by_name.setdefault(normalise_bone_name(name), index)

    resolved: dict[str, int] = {}
    for bone in BONE_TREE:
        mixamo_name = MIXAMO_BONE_MAP.get(bone)
        bare_mixamo_name = (
            mixamo_name.split(":", 1)[1]
            if mixamo_name and ":" in mixamo_name
            else mixamo_name
        )
        for candidate in (bone, mixamo_name, bare_mixamo_name):
            if not candidate:
                continue
            hit = by_name.get(candidate)
            if hit is None:
                hit = by_name.get(normalise_bone_name(candidate))
            if hit is not None:
                resolved[bone] = hit
                break
    return resolved


def validate_avatar_glb(data: bytes) -> dict[str, int]:
    """Validate a sideloaded GLB and return its contract bone mapping.

    A matching node name alone is not enough: each resolved node must also be
    a joint in a skin, otherwise posing it would not deform the model.
    """
    document, _binary = read_glb(data)
    nodes = document.get("nodes")
    skins = document.get("skins")
    if (not isinstance(nodes, list)
            or not all(isinstance(node, dict) for node in nodes)):
        raise GlbError("GLB has no valid node list")
    if any("name" in node and not isinstance(node["name"], str) for node in nodes):
        raise GlbError("GLB contains a node with an invalid name")
    if not isinstance(skins, list) or not skins:
        raise GlbNotRiggedError("GLB does not contain a skinned skeleton")

    joint_indices: set[int] = set()
    for skin_number, skin in enumerate(skins):
        joints = skin.get("joints") if isinstance(skin, dict) else None
        if not isinstance(joints, list):
            raise GlbError(f"skin {skin_number} has no joint list")
        for node_index in joints:
            if (not isinstance(node_index, int) or isinstance(node_index, bool)
                    or not 0 <= node_index < len(nodes)):
                raise GlbError(
                    f"skin {skin_number} references invalid joint {node_index!r}")
            joint_indices.add(node_index)

    bones = resolve_bones(document)
    missing = [bone for bone in BONE_TREE
               if bone not in bones and bone not in OPTIONAL_BONES]
    if missing:
        raise GlbError(f"missing required joints: {', '.join(missing)}")

    non_skin = [bone for bone, index in bones.items() if index not in joint_indices]
    if non_skin:
        raise GlbError(
            f"required nodes are not skin joints: {', '.join(non_skin)}")
    return bones


# --------------------------------------------------------------------------
# Posing
# --------------------------------------------------------------------------

def _conjugate(q):
    return (-q[0], -q[1], -q[2], q[3])


def _decompose(matrix: list[float]):
    """Split a glTF 4x4 (column-major) into translation, rotation, scale.

    glTF lets a node carry EITHER translation/rotation/scale OR a single
    `matrix`, and when `matrix` is present the TRS fields are ignored entirely.
    three's GLTFExporter writes `matrix`; Meshy's FBX pipeline writes TRS. Miss
    this and posing a matrix-form rig silently does nothing: the JSON gains a
    `rotation` nobody reads and the model renders at bind.
    """
    # Column c, row r lives at matrix[c * 4 + r].
    col = [matrix[0:3], matrix[4:7], matrix[8:11]]
    translation = list(matrix[12:15])
    scale = [max((sum(c * c for c in axis)) ** 0.5, 1e-12) for axis in col]

    # A negative determinant means the basis is mirrored; fold that into the
    # first axis so the remainder is a pure rotation.
    det = (col[0][0] * (col[1][1] * col[2][2] - col[1][2] * col[2][1])
           - col[1][0] * (col[0][1] * col[2][2] - col[0][2] * col[2][1])
           + col[2][0] * (col[0][1] * col[1][2] - col[0][2] * col[1][1]))
    if det < 0:
        scale[0] = -scale[0]

    r = [[col[c][row] / scale[c] for c in range(3)] for row in range(3)]
    trace = r[0][0] + r[1][1] + r[2][2]
    if trace > 0:
        s = 0.5 / ((trace + 1.0) ** 0.5)
        q = ((r[2][1] - r[1][2]) * s, (r[0][2] - r[2][0]) * s,
             (r[1][0] - r[0][1]) * s, 0.25 / s)
    elif r[0][0] > r[1][1] and r[0][0] > r[2][2]:
        s = 2.0 * ((1.0 + r[0][0] - r[1][1] - r[2][2]) ** 0.5)
        q = (0.25 * s, (r[0][1] + r[1][0]) / s,
             (r[0][2] + r[2][0]) / s, (r[2][1] - r[1][2]) / s)
    elif r[1][1] > r[2][2]:
        s = 2.0 * ((1.0 + r[1][1] - r[0][0] - r[2][2]) ** 0.5)
        q = ((r[0][1] + r[1][0]) / s, 0.25 * s,
             (r[1][2] + r[2][1]) / s, (r[0][2] - r[2][0]) / s)
    else:
        s = 2.0 * ((1.0 + r[2][2] - r[0][0] - r[1][1]) ** 0.5)
        q = ((r[0][2] + r[2][0]) / s, (r[1][2] + r[2][1]) / s,
             0.25 * s, (r[1][0] - r[0][1]) / s)

    length = sum(c * c for c in q) ** 0.5 or 1.0
    return translation, tuple(c / length for c in q), scale


def _node_rotation(node: dict[str, Any]):
    """The node's local rotation, whichever form it is stored in."""
    if "matrix" in node:
        return _decompose(node["matrix"])[1]
    r = node.get("rotation")
    return tuple(r) if r else IDENTITY_QUAT


def _set_node_rotation(node: dict[str, Any], q) -> None:
    """Write a local rotation, converting a matrix node to TRS if needed.

    Converting rather than recomposing the matrix is deliberate: glTF
    animations target `translation`/`rotation`/`scale` channels and cannot
    target `matrix`, so a TRS node is what the later animation-baking step will
    need anyway.
    """
    if "matrix" in node:
        translation, _rotation, scale = _decompose(node.pop("matrix"))
        if any(abs(c) > 1e-9 for c in translation):
            node["translation"] = [round(c, 6) for c in translation]
        if any(abs(c - 1.0) > 1e-9 for c in scale):
            node["scale"] = [round(c, 6) for c in scale]
    node["rotation"] = [round(c, 6) for c in q]


def _parents(document: dict[str, Any]) -> dict[int, int]:
    parents: dict[int, int] = {}
    for index, node in enumerate(document.get("nodes", [])):
        for child in node.get("children", []) or []:
            parents[child] = index
    return parents


def _bind_world_rotations(document: dict[str, Any]) -> dict[int, tuple]:
    """World rotation of every node in the file's current (bind) state."""
    nodes = document.get("nodes", [])
    parents = _parents(document)
    roots = [i for i in range(len(nodes)) if i not in parents]

    world: dict[int, tuple] = {}
    stack = [(r, IDENTITY_QUAT) for r in roots]
    while stack:
        index, parent_world = stack.pop()
        here = quat_mul(parent_world, _node_rotation(nodes[index]))
        world[index] = here
        for child in nodes[index].get("children", []) or []:
            stack.append((child, here))
    return world


def contract_world_deltas(pose: dict[str, list[float]]) -> dict[str, tuple]:
    """FK our canonical skeleton to a world rotation per contract bone.

    Our bind is identity everywhere, so a bone's world rotation IS its delta
    from bind — which is exactly what has to be applied to the GLB.
    """
    world: dict[str, tuple] = {}
    for bone, (parent, _offset) in BONE_TREE.items():
        local = tuple(pose.get(bone, IDENTITY_QUAT))
        world[bone] = quat_mul(world[parent], local) if parent else local
    return world


def _solve_locals(document: dict[str, Any], pose: dict[str, list[float]],
                  bones: dict[str, int]) -> dict[int, tuple]:
    """The local rotation each mapped node needs to show `pose`.

    Walks the node tree top-down carrying each node's POSED world rotation:
    a mapped bone gets `delta * bind_world` and its local is solved against its
    parent's posed world; an unmapped bone (fingers, Spine1/2) keeps its local
    rotation and simply inherits the movement of its parents.

    Reads the document's current node rotations as bind, so call it before
    anything writes a pose into them.
    """
    nodes = document.get("nodes", [])
    bind_world = _bind_world_rotations(document)
    deltas = contract_world_deltas(pose)
    node_to_bone = {index: bone for bone, index in bones.items()}
    parents = _parents(document)
    roots = [i for i in range(len(nodes)) if i not in parents]

    locals_: dict[int, tuple] = {}
    stack = [(r, IDENTITY_QUAT) for r in roots]
    while stack:
        index, parent_world = stack.pop()
        bone = node_to_bone.get(index)

        if bone is not None:
            # World-space rotations pre-multiply: delta THEN bind orientation.
            desired = quat_mul(deltas[bone], bind_world[index])
            locals_[index] = quat_mul(_conjugate(parent_world), desired)
            here = desired
        else:
            here = quat_mul(parent_world, _node_rotation(nodes[index]))

        for child in nodes[index].get("children", []) or []:
            stack.append((child, here))

    return locals_


def pose_glb(document: dict[str, Any], pose: dict[str, list[float]],
             bones: dict[str, int] | None = None) -> dict[str, int]:
    """Write a contract pose into a glTF document's node rotations.

    Returns the bone -> node-index map that was used, so a caller can record
    which joints actually moved.
    """
    if bones is None:
        bones = resolve_bones(document)
    nodes = document.get("nodes", [])
    for index, q in _solve_locals(document, pose, bones).items():
        _set_node_rotation(nodes[index], q)
    return bones


# --------------------------------------------------------------------------
# Animation
# --------------------------------------------------------------------------

_FLOAT = 5126   # componentType FLOAT


def add_pose_animation(document: dict[str, Any], binary: bytes,
                       keyframes: list[tuple[float, dict[str, list[float]]]],
                       bones: dict[str, int], name: str) -> bytes:
    """Bake contract-pose keyframes into a glTF animation. Returns the new BIN.

    One LINEAR rotation channel per mapped bone, all sharing one time accessor.
    Must run while the node rotations are still at bind — every keyframe is
    retargeted against them — so call it BEFORE `pose_glb`.
    """
    nodes = document.get("nodes", [])
    frames = [_solve_locals(document, pose, bones) for _t, pose in keyframes]
    times = [float(t) for t, _pose in keyframes]

    # Channels cannot target `matrix`; convert to TRS, keeping the bind value.
    for index in bones.values():
        if "matrix" in nodes[index]:
            _set_node_rotation(nodes[index], _node_rotation(nodes[index]))

    buffers = document.setdefault("buffers", [])
    if not buffers:
        buffers.append({"byteLength": 0})
    views = document.setdefault("bufferViews", [])
    accessors = document.setdefault("accessors", [])
    out = bytearray(binary)

    def push(values: list[float], kind: str, count: int, **extra) -> int:
        out.extend(b"\x00" * (-len(out) % 4))
        views.append({"buffer": 0, "byteOffset": len(out),
                      "byteLength": 4 * len(values)})
        out.extend(struct.pack(f"<{len(values)}f", *values))
        accessors.append({"bufferView": len(views) - 1, "componentType": _FLOAT,
                          "count": count, "type": kind, **extra})
        return len(accessors) - 1

    time_accessor = push(times, "SCALAR", len(times),
                         min=[min(times)], max=[max(times)])

    samplers, channels = [], []
    for index in sorted(bones.values()):
        values: list[float] = []
        previous = None
        for frame in frames:
            q = frame[index]
            # Keep neighbours in the same hemisphere so the slerp takes the
            # short way round; not every viewer corrects for it.
            if previous and sum(a * b for a, b in zip(q, previous)) < 0:
                q = tuple(-c for c in q)
            values.extend(q)
            previous = q
        samplers.append({"input": time_accessor, "interpolation": "LINEAR",
                         "output": push(values, "VEC4", len(frames))})
        channels.append({"sampler": len(samplers) - 1,
                         "target": {"node": index, "path": "rotation"}})

    # First, because viewers autoplay animation 0 and an uploaded GLB may
    # already carry its own (a Mixamo idle, say).
    document.setdefault("animations", []).insert(
        0, {"name": name, "samplers": samplers, "channels": channels})
    buffers[0]["byteLength"] = len(out)
    return bytes(out)


# --------------------------------------------------------------------------
# Reading animations back — the inverse of add_pose_animation
# --------------------------------------------------------------------------

#: componentType -> (struct code, normaliser). glTF allows rotation outputs as
#: normalised integers as well as floats.
_COMPONENTS = {
    5126: ("f", None),
    5120: ("b", lambda v: max(v / 127.0, -1.0)),
    5121: ("B", lambda v: v / 255.0),
    5122: ("h", lambda v: max(v / 32767.0, -1.0)),
    5123: ("H", lambda v: v / 65535.0),
}
_WIDTH = {"SCALAR": 1, "VEC3": 3, "VEC4": 4}

#: Sampling rate for imported clips. Plenty for the viewport's slerp, and
#: keeps a 20s clip inside the contract's keyframe limit.
IMPORT_FPS = 12


def _read_accessor(document: dict[str, Any], binary: bytes,
                   index: int) -> list[tuple[float, ...]]:
    accessor = document["accessors"][index]
    if "sparse" in accessor or "bufferView" not in accessor:
        raise GlbError(f"accessor {index} is sparse or has no data")
    code, normalise = _COMPONENTS[accessor["componentType"]]
    width = _WIDTH[accessor["type"]]
    view = document["bufferViews"][accessor["bufferView"]]
    if view.get("buffer", 0) != 0:
        raise GlbError(f"accessor {index} is not in the GLB's own buffer")

    size = struct.calcsize(code)
    stride = view.get("byteStride") or size * width
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    out = []
    for i in range(accessor["count"]):
        values = struct.unpack_from(f"<{width}{code}", binary, start + i * stride)
        if normalise:
            values = tuple(normalise(v) for v in values)
        out.append(tuple(float(v) for v in values))
    return out


def _slerp(a, b, alpha: float):
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-c for c in b), -dot
    if dot > 0.9995:   # nearly parallel: nlerp is exact enough
        q = tuple(x + (y - x) * alpha for x, y in zip(a, b))
    else:
        theta = math.acos(dot)
        sa = math.sin((1 - alpha) * theta) / math.sin(theta)
        sb = math.sin(alpha * theta) / math.sin(theta)
        q = tuple(sa * x + sb * y for x, y in zip(a, b))
    length = math.sqrt(sum(c * c for c in q)) or 1.0
    return tuple(c / length for c in q)


def _sample(times: list[float], values: list[tuple], interpolation: str,
            t: float):
    if interpolation == "CUBICSPLINE":
        # Each key is (in-tangent, value, out-tangent); keep the values and
        # interpolate those linearly — close enough at our sampling rate.
        values = values[1::3]
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    i = 0
    while times[i + 1] < t:
        i += 1
    if interpolation == "STEP":
        return values[i]
    span = times[i + 1] - times[i]
    return _slerp(values[i], values[i + 1],
                  (t - times[i]) / span if span > 0 else 0.0)


def read_clips(document: dict[str, Any], binary: bytes,
               bones: dict[str, int] | None = None) -> list[dict[str, Any]]:
    """Every glTF animation in the file, as contract clip dicts.

    Each frame is FK'd through the GLB's own node tree (so unmapped bones such
    as a chest still carry their children), then each contract bone's world
    rotation is turned back into a delta from bind and made local against its
    contract parent — `pose_glb` run backwards. Only rotations come across;
    translation (a root bob) and scale have no place in a contract pose.

    Reads the node rotations as bind, so the file must be at rest — which is
    how riggers and our own export leave it.
    """
    if bones is None:
        bones = resolve_bones(document)
    nodes = document.get("nodes", [])
    bind_world = _bind_world_rotations(document)
    rest_local = {i: _node_rotation(n) for i, n in enumerate(nodes)}
    parents = _parents(document)
    roots = [i for i in range(len(nodes)) if i not in parents]

    clips = []
    for number, animation in enumerate(document.get("animations", []) or []):
        channels = {}
        try:
            for channel in animation.get("channels", []):
                target = channel.get("target", {})
                if target.get("path") != "rotation" or "node" not in target:
                    continue
                sampler = animation["samplers"][channel["sampler"]]
                channels[target["node"]] = (
                    [t[0] for t in _read_accessor(document, binary, sampler["input"])],
                    _read_accessor(document, binary, sampler["output"]),
                    sampler.get("interpolation", "LINEAR"),
                )
        except (GlbError, KeyError, IndexError, TypeError, struct.error):
            continue   # one unreadable clip shouldn't cost the others
        if not channels:
            continue

        duration = min(max(c[0][-1] for c in channels.values()), 30.0)
        count = min(int(round(duration * IMPORT_FPS)) + 1, 240)
        step = duration / (count - 1) if count > 1 else 0.0

        keyframes = []
        for k in range(count):
            t = k * step
            world: dict[int, tuple] = {}
            stack = [(r, IDENTITY_QUAT) for r in roots]
            while stack:
                index, parent_world = stack.pop()
                local = (_sample(*channels[index][:2], channels[index][2], t)
                         if index in channels else rest_local[index])
                world[index] = quat_mul(parent_world, local)
                for child in nodes[index].get("children", []) or []:
                    stack.append((child, world[index]))

            deltas = {bone: quat_mul(world[index], _conjugate(bind_world[index]))
                      for bone, index in bones.items()}
            pose = {}
            for bone in bones:
                parent = BONE_TREE[bone][0]
                q = (quat_mul(_conjugate(deltas[parent]), deltas[bone])
                     if parent in deltas else deltas[bone])
                pose[bone] = [round(c, 5) for c in q]
            keyframes.append({"t": round(t, 4), "pose": pose})

        if all(abs(q[3]) > 0.9999 for frame in keyframes
               for q in frame["pose"].values()):
            continue   # drives nothing we can show (e.g. only fingers)

        clips.append({
            "index": number,   # which of the file's animations this came from
            "name": (animation.get("name") or f"Move {number + 1}").strip(),
            "fps": IMPORT_FPS,
            "loop": True,
            "keyframes": keyframes,
        })
    return clips


def attach_extras(document: dict[str, Any], payload: dict[str, Any]) -> None:
    """Embed our metadata under asset.extras, namespaced.

    Namespaced because `extras` is a shared free-for-all — another tool in the
    chain may well want to put its own keys there.
    """
    asset = document.setdefault("asset", {})
    extras = asset.setdefault("extras", {})
    extras["avatarTrainer"] = payload
