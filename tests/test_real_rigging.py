from __future__ import annotations

import json
import struct
from collections.abc import Callable
from urllib.error import HTTPError
from urllib.parse import urlsplit

import pytest

from providers.real.rigging import GlbError, RealRigger, rewrite_glb_joint_names
from schemas import ARM_BONES, BONES, LEG_BONES, ProviderError

REMOTE_NAMES = {
    "hips": "joint_pelvis",
    "spine": "joint_spine",
    "neck": "joint_neck",
    "head": "joint_head",
    "L_shoulder": "joint_left_shoulder",
    "L_elbow": "joint_left_elbow",
    "L_hand": "joint_left_hand",
    "R_shoulder": "joint_right_shoulder",
    "R_elbow": "joint_right_elbow",
    "R_hand": "joint_right_hand",
    "L_hip": "joint_left_hip",
    "L_knee": "joint_left_knee",
    "L_foot": "joint_left_foot",
    "R_hip": "joint_right_hip",
    "R_knee": "joint_right_knee",
    "R_foot": "joint_right_foot",
    "tail": "joint_tail_base",
    "tail_tip": "joint_tail_tip",
}


def make_glb(
    *,
    missing: str | None = None,
    duplicate: str | None = None,
    drop: frozenset[str] = frozenset(),
) -> bytes:
    nodes = [{"name": "joint_root"}, {"name": "joint_chest"}]
    joints = [0, 1]
    for bone in BONES:
        if bone == missing or bone in drop:
            continue
        joints.append(len(nodes))
        nodes.append({"name": REMOTE_NAMES[bone]})
        if bone == duplicate:
            joints.append(len(nodes))
            nodes.append({"name": REMOTE_NAMES[bone]})

    document = {
        "asset": {"version": "2.0"},
        "nodes": nodes,
        "skins": [{"joints": joints}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    json_chunk = json.dumps(document, separators=(",", ":")).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    chunks = [
        (b"JSON", json_chunk),
        (b"BIN\x00", b"\x01\x02\x03\x04"),
        (b"TEST", b"keep-this-chunk"),
    ]
    body = b"".join(
        struct.pack("<I4s", len(payload), kind) + payload
        for kind, payload in chunks
    )
    return struct.pack("<4sII", b"glTF", 2, len(body) + 12) + body


def read_glb(glb: bytes) -> tuple[dict, list[tuple[bytes, bytes]]]:
    chunks = []
    offset = 12
    document = None
    while offset < len(glb):
        length, kind = struct.unpack_from("<I4s", glb, offset)
        payload = glb[offset + 8:offset + 8 + length]
        chunks.append((kind, payload))
        if kind == b"JSON":
            document = json.loads(payload.rstrip(b" \x00"))
        offset += 8 + length
    assert document is not None
    return document, chunks


class FakeResponse:
    def __init__(self, body: bytes):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self, size: int = -1) -> bytes:
        return self.body if size < 0 else self.body[:size]


class SequenceOpener:
    def __init__(self, responses: list[tuple[str, dict | bytes | Exception]]):
        self.responses = list(responses)
        self.requests = []

    def __call__(self, request, *, timeout: float):
        assert timeout > 0
        self.requests.append(request)
        expected_target, response = self.responses.pop(0)
        parsed = urlsplit(request.full_url)
        actual_target = parsed.path + (f"?{parsed.query}" if parsed.query else "")
        assert actual_target == expected_target
        if isinstance(response, Exception):
            raise response
        if isinstance(response, dict):
            response = json.dumps(response).encode()
        return FakeResponse(response)


def rigger_with(
    responses: list[tuple[str, dict | bytes | Exception]],
    **kwargs,
) -> tuple[RealRigger, SequenceOpener]:
    opener = SequenceOpener(responses)
    rigger = RealRigger(
        "https://rigging.example.test",
        timeout=30,
        poll_interval=0,
        opener=opener,
        **kwargs,
    )
    return rigger, opener


def not_found() -> HTTPError:
    return HTTPError("https://rigging.example.test/missing", 404, "missing", {}, None)


def server_error() -> HTTPError:
    return HTTPError(
        "https://rigging.example.test/results/missing",
        500,
        "internal server error",
        {},
        None,
    )


def successful_responses(
    glb: bytes,
    *,
    cache_miss: HTTPError | None = None,
) -> list[tuple[str, dict | bytes | Exception]]:
    return [
        ("/classify", {"category": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", cache_miss or not_found()),
        ("/augment_image?classify_id=drawing-1", {
            "status": "ok",
            "image_a_url": "/results/drawing-1/augmented-a.png",
            "image_b_url": "/results/drawing-1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing-1&choice=a", {
            "status": "ok",
            "choice": "a",
            "active_image_url": "/results/drawing-1/augmented-a.png",
        }),
        ("/mesh?classify_id=drawing-1", {"status": "completed",
                   "mesh_url": "http://worker.internal/files/mesh.glb"}),
        ("/infer_joints?classify_id=drawing-1",
         {"joint_hints": [{"name": "joint_pelvis"}]}),
        ("/rig?classify_id=drawing-1", {
            "status": "completed",
            "glb_url": "https://assets.example.test/unrigged.glb",
            "rigged_url": "http://worker.internal/files/rigged.glb",
        }),
        ("/files/rigged.glb", glb),
    ]


def test_immediate_cache_hits_return_rewritten_glb():
    original = make_glb()
    rigger, opener = rigger_with(successful_responses(original))
    progress = []

    rig = rigger.rig(
        b"png", "image/png", lambda value, message: progress.append((value, message)))

    assert rig.format == "glb"
    assert rig.skeleton == BONES
    document, chunks = read_glb(rig.glb_bytes)
    skin_names = {
        document["nodes"][index]["name"]
        for index in document["skins"][0]["joints"]
    }
    assert set(BONES) <= skin_names
    assert {"joint_root", "joint_chest"} <= skin_names
    assert chunks[1:] == read_glb(original)[1][1:]
    assert progress[-1][0] == pytest.approx(0.99)
    assert opener.responses == []

    assert opener.requests[0].data == b"png"
    assert opener.requests[0].get_header("Content-type") == "image/png"
    assert all(request.data is None for request in opener.requests[2:5])


# -- a cached fish or bird from before tails existed --------------------------

TAIL = frozenset({"tail", "tail_tip"})


def _skin_names(glb):
    document, _ = read_glb(glb)
    return {document["nodes"][i]["name"] for i in document["skins"][0]["joints"]}


def test_a_cached_fish_without_a_tail_is_rerigged_with_one():
    old, new = make_glb(drop=TAIL), make_glb()
    rigger, opener = rigger_with([
        ("/classify", {"category": "animal", "classify_id": "fish-1"}),
        ("/results/fish-1/fish-1_rigged.glb", old),
        # cached_only: never runs the vision model. seg_server has added the
        # tail to the stored joints and marked the rig stale.
        ("/infer_joints?classify_id=fish-1&cached_only=true",
         {"joint_hints": [{"name": "joint_pelvis"}, {"name": "joint_tail_base"},
                          {"name": "joint_tail_tip"}]}),
        ("/rig?classify_id=fish-1", {
            "status": "completed",
            "rigged_url": "http://worker.internal/files/rigged.glb"}),
        ("/files/rigged.glb", new),
    ])

    rig = rigger.rig(b"png", "image/png", lambda *_: None)

    assert TAIL <= _skin_names(rig.glb_bytes)
    assert opener.responses == []


def test_a_cached_rig_that_needs_no_tail_costs_one_lookup():
    # A bee, say: its stored joints come back without a tail, so the cached
    # rig is used as it is.
    rigger, opener = rigger_with([
        ("/classify", {"category": "animal", "classify_id": "bee-1"}),
        ("/results/bee-1/bee-1_rigged.glb", make_glb(drop=TAIL)),
        ("/infer_joints?classify_id=bee-1&cached_only=true",
         {"joint_hints": [{"name": "joint_pelvis"}]}),
    ])

    rig = rigger.rig(b"png", "image/png", lambda *_: None)

    assert not TAIL & _skin_names(rig.glb_bytes)
    assert opener.responses == []


@pytest.mark.parametrize("failure", [not_found, server_error])
def test_a_failed_tail_upgrade_keeps_the_cached_rig(failure):
    rigger, opener = rigger_with([
        ("/classify", {"category": "animal", "classify_id": "old-1"}),
        ("/results/old-1/old-1_rigged.glb", make_glb(drop=TAIL)),
        ("/infer_joints?classify_id=old-1&cached_only=true", failure()),
    ])

    rig = rigger.rig(b"png", "image/png", lambda *_: None)

    assert rig.format == "glb"
    assert opener.responses == []


def test_existing_rig_is_downloaded_without_restarting_remote_tasks():
    rigger, opener = rigger_with([
        ("/classify", {"category": "humanoid", "classify_id": "cached-1"}),
        ("/results/cached-1/cached-1_rigged.glb", make_glb()),
    ])

    rig = rigger.rig(b"png", "image/png", lambda *_: None)

    assert rig.glb_bytes
    assert len(opener.requests) == 2


def test_result_endpoint_server_error_is_treated_as_cache_miss():
    rigger, opener = rigger_with(
        successful_responses(make_glb(), cache_miss=server_error()))

    rig = rigger.rig(b"png", "image/png", lambda *_: None)

    assert rig.glb_bytes
    assert any(
        urlsplit(request.full_url).path == "/mesh"
        for request in opener.requests
    )


def test_async_mesh_and_rig_tasks_are_polled():
    rigger, opener = rigger_with([
        ("/classify", {
            "result": {"label": "person"}, "classify_id": "drawing/1"}),
        ("/results/drawing%2F1/drawing%2F1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing%2F1", {
            "image_a_url": "/results/drawing%2F1/augmented-a.png",
            "image_b_url": "/results/drawing%2F1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing%2F1&choice=a",
         {"status": "ok", "choice": "a"}),
        ("/mesh?classify_id=drawing%2F1",
         {"status": "queued", "task_id": "mesh 1"}),
        ("/mesh/status/mesh%201", {"status": "meshy", "progress": 10}),
        ("/mesh/status/mesh%201", {"status": "decimating", "progress": 85}),
        ("/mesh/status/mesh%201", {"status": "done",
                            "result": {"url": "/files/mesh.glb"}}),
        ("/infer_joints?classify_id=drawing%2F1",
         {"result": {"joints": {"hips": [0, 0, 0]}}}),
        ("/rig?classify_id=drawing%2F1",
         {"state": "accepted", "job_id": "rig-1"}),
        ("/rig/status/rig-1", {"state": "rigging", "progress": 10}),
        ("/rig/status/rig-1", {"state": "decimating", "progress": 20}),
        ("/rig/status/rig-1", {"state": "inferring_skeleton", "progress": 30}),
        ("/rig/status/rig-1", {"state": "injecting_keyframes", "progress": 40}),
        ("/rig/status/rig-1", {"state": "visualizing", "progress": 50}),
        ("/rig/status/rig-1", {"state": "rigging_blender", "progress": 60}),
        ("/rig/status/rig-1", {"state": "finalizing", "progress": 90}),
        ("/rig/status/rig-1", {"state": "succeeded",
                        "result": {"download_url": "/files/avatar.glb"}}),
        ("/files/avatar.glb", make_glb()),
    ])
    seen = []

    rig = rigger.rig(b"png", "image/png", lambda f, m: seen.append((f, m)))

    assert rig.glb_bytes
    assert sum(request.method == "GET" for request in opener.requests) == 13
    assert any(0.26 < fraction <= 0.52 for fraction, _ in seen)
    assert any(0.70 < fraction <= 0.90 for fraction, _ in seen)


@pytest.mark.parametrize("response", [
    {"classification": "tree"},
    {"result": {"category": "building"}},
    {"is_humanoid": False},
])
def test_unsupported_drawings_get_a_child_safe_error(response):
    rigger, _ = rigger_with([("/classify", response)])
    with pytest.raises(ProviderError, match="unrecognized classification") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)
    assert "people, animals, and vehicles" in caught.value.user_message


@pytest.mark.parametrize("label", ["car", "dog", "animal"])
def test_animals_and_vehicles_get_past_classification(label):
    # No classify_id in the response, so rigging stops at the next check —
    # past the label gate.
    rigger, _ = rigger_with([("/classify", {"classification": label})])
    with pytest.raises(ProviderError) as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)
    assert "unrecognized classification" not in caught.value.detail


@pytest.mark.parametrize("rig_type", ["humanoid", "biped", "quadruped"])
def test_food_characters_with_a_body_get_past_classification(rig_type):
    # What seg_server returns for the walking broccoli and the tomato.
    rigger, _ = rigger_with([("/mesh/upload",
                              {"category": "other", "rig_type": rig_type})])
    with pytest.raises(ProviderError) as caught:
        rigger.rig_glb(b"glTF mesh", lambda *_: None)
    assert "unrecognized classification" not in caught.value.detail


def test_other_with_no_body_is_still_refused():
    rigger, _ = rigger_with([("/classify",
                              {"category": "other", "rig_type": "other"})])
    with pytest.raises(ProviderError, match="unrecognized classification"):
        rigger.rig(b"png", "image/png", lambda *_: None)


def test_uploaded_glb_skips_augment_and_mesh_and_goes_to_joints():
    rigged = make_glb()
    rigger, opener = rigger_with([
        ("/mesh/upload", {"category": "humanoid", "classify_id": "upload-1"}),
        ("/results/upload-1/upload-1_rigged.glb", not_found()),
        ("/infer_joints?classify_id=upload-1",
         {"joint_hints": [{"name": "joint_pelvis"}]}),
        ("/rig?classify_id=upload-1", {
            "status": "completed",
            "rigged_url": "http://worker.internal/files/rigged.glb",
        }),
        ("/files/rigged.glb", rigged),
    ])

    rig = rigger.rig_glb(b"glTF mesh", lambda *_: None)

    assert rig.format == "glb"
    assert opener.responses == []
    assert opener.requests[0].data == b"glTF mesh"
    assert opener.requests[0].get_header("Content-type") == "model/gltf-binary"


def test_uploaded_glb_of_an_unsupported_subject_is_refused():
    rigger, _ = rigger_with([
        ("/mesh/upload", {"category": "building", "classify_id": "upload-2"}),
    ])
    with pytest.raises(ProviderError, match="unrecognized classification"):
        rigger.rig_glb(b"glTF mesh", lambda *_: None)


def test_malformed_json_is_reported_as_provider_error():
    rigger, _ = rigger_with([("/classify", b"<html>not json</html>")])
    with pytest.raises(ProviderError, match="malformed JSON"):
        rigger.rig(b"png", "image/png", lambda *_: None)


def test_remote_http_error_is_reported_as_provider_error():
    error = HTTPError(
        "https://rigging.example.test/classify", 503, "unavailable", {}, None)
    error.read = lambda _size=-1: b"service unavailable"
    rigger, _ = rigger_with([("/classify", error)])
    with pytest.raises(ProviderError, match="HTTP 503"):
        rigger.rig(b"png", "image/png", lambda *_: None)


def test_failed_remote_task_is_reported():
    rigger, _ = rigger_with([
        ("/classify", {"classification": "human", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing-1", {
            "image_a_url": "/results/drawing-1/augmented-a.png",
            "image_b_url": "/results/drawing-1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing-1&choice=a",
         {"status": "ok", "choice": "a"}),
        ("/mesh?classify_id=drawing-1",
         {"status": "failed", "error": "out of GPU memory"}),
    ])
    with pytest.raises(ProviderError, match="out of GPU memory") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)
    assert caught.value.user_message == \
        "I couldn't finish building that avatar. Try the drawing again?"


def test_invalid_download_is_rejected():
    rigger, _ = rigger_with(successful_responses(b"not a glb"))
    with pytest.raises(ProviderError, match="invalid rigged GLB"):
        rigger.rig(b"png", "image/png", lambda *_: None)


@pytest.mark.parametrize("augmentation", [
    {"image_b_url": "/results/drawing-1/augmented-b.png"},
    {
        "image_a_url": "/results/drawing-1/augmented-a.png",
        "image_b_url": "",
    },
])
def test_incomplete_augmentation_response_stops_before_mesh(augmentation):
    rigger, opener = rigger_with([
        ("/classify", {
            "classification": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing-1", augmentation),
    ])

    with pytest.raises(ProviderError, match="invalid image URLs") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)

    assert "incomplete pose" in caught.value.user_message
    assert len(opener.requests) == 3


def test_incomplete_augmentation_confirmation_stops_before_mesh():
    rigger, opener = rigger_with([
        ("/classify", {
            "classification": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing-1", {
            "image_a_url": "/results/drawing-1/augmented-a.png",
            "image_b_url": "/results/drawing-1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing-1&choice=a",
         {"status": "ok"}),
    ])

    with pytest.raises(ProviderError, match="confirmation was incomplete") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)

    assert "couldn't choose a pose" in caught.value.user_message
    assert len(opener.requests) == 4


@pytest.mark.parametrize("failed_target", ["augment", "confirm"])
def test_augmentation_http_errors_stop_before_mesh(failed_target):
    error = HTTPError(
        "https://rigging.example.test/augment_image",
        503,
        "unavailable",
        {},
        None,
    )
    responses = [
        ("/classify", {
            "classification": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
    ]
    if failed_target == "augment":
        responses.append(("/augment_image?classify_id=drawing-1", error))
    else:
        responses.extend([
            ("/augment_image?classify_id=drawing-1", {
                "image_a_url": "/results/drawing-1/augmented-a.png",
                "image_b_url": "/results/drawing-1/augmented-b.png",
            }),
            ("/augment_image/confirm?classify_id=drawing-1&choice=a", error),
        ])
    rigger, opener = rigger_with(responses)

    with pytest.raises(ProviderError, match="HTTP 503"):
        rigger.rig(b"png", "image/png", lambda *_: None)

    assert len(opener.requests) == len(responses)


def test_overall_deadline_applies_while_polling():
    now = [0.0]

    def clock() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        now[0] += seconds

    opener = SequenceOpener([
        ("/classify", {
            "classification": "humanoid", "classify_id": "drawing-1"}),
        ("/results/drawing-1/drawing-1_rigged.glb", not_found()),
        ("/augment_image?classify_id=drawing-1", {
            "image_a_url": "/results/drawing-1/augmented-a.png",
            "image_b_url": "/results/drawing-1/augmented-b.png",
        }),
        ("/augment_image/confirm?classify_id=drawing-1&choice=a",
         {"status": "ok", "choice": "a"}),
        ("/mesh?classify_id=drawing-1",
         {"status": "queued", "task_id": "slow"}),
    ])
    rigger = RealRigger(
        "https://rigging.example.test",
        timeout=1,
        poll_interval=2,
        opener=opener,
        clock=clock,
        sleep=sleep,
    )
    with pytest.raises(ProviderError, match="exceeded 1.000s deadline") as caught:
        rigger.rig(b"png", "image/png", lambda *_: None)
    assert "took too long" in caught.value.user_message


def test_http_urls_are_rebuilt_on_the_configured_https_origin():
    rigger = RealRigger("https://public.ngrok-free.app/api")
    assert rigger._service_url(
        "http://127.0.0.1:8000/output/avatar.glb?token=abc"
    ) == "https://public.ngrok-free.app/output/avatar.glb?token=abc"
    assert rigger._service_url(
        "/tasks/123"
    ) == "https://public.ngrok-free.app/api/tasks/123"


def test_glb_rewriter_preserves_non_json_chunks_and_extra_nodes():
    original = make_glb()
    rewritten = rewrite_glb_joint_names(original)
    document, chunks = read_glb(rewritten)
    original_document, original_chunks = read_glb(original)

    assert chunks[1:] == original_chunks[1:]
    assert [node["name"] for node in document["nodes"][:2]] == \
        [node["name"] for node in original_document["nodes"][:2]]
    assert struct.unpack_from("<I", rewritten, 8)[0] == len(rewritten)


@pytest.mark.parametrize(("kwargs", "message"), [
    ({"missing": "neck"}, "missing contract bones: neck"),
    ({"duplicate": "L_elbow"}, "duplicate contract bones: L_elbow"),
])
def test_glb_rewriter_rejects_invalid_contract_skins(kwargs, message):
    with pytest.raises(GlbError, match=message):
        rewrite_glb_joint_names(make_glb(**kwargs))


def test_glb_rewriter_accepts_an_armless_rig():
    # seg_server rigs a tomato with legs as a biped: no shoulders, elbows or
    # hands.
    rewritten = rewrite_glb_joint_names(make_glb(drop=ARM_BONES))
    names = {node["name"] for node in read_glb(rewritten)[0]["nodes"]}
    assert names >= set(BONES) - ARM_BONES
    assert not names & ARM_BONES


def _skin_glb(names):
    nodes = [{"name": n} for n in names]
    document = {"asset": {"version": "2.0"}, "nodes": nodes,
                "skins": [{"joints": list(range(len(nodes)))}],
                "scenes": [{"nodes": [0]}], "scene": 0}
    json_chunk = json.dumps(document).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    body = (struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
            + struct.pack("<I4s", 4, b"BIN\x00") + b"\0\0\0\0")
    return struct.pack("<4sII", b"glTF", 2, len(body) + 12) + body


def test_glb_rewriter_rigs_a_birds_wings_as_arms():
    # seg_server's flying skeleton: no pelvis (root is the hips), wings with
    # base, mid and tip, and legs that go hip -> foot with no knee.
    glb = _skin_glb([
        "joint_root", "joint_spine", "joint_chest", "joint_neck", "joint_head",
        "joint_wing_base_left", "joint_wing_mid_left", "joint_wing_tip_left",
        "joint_wing_base_right", "joint_wing_mid_right", "joint_wing_tip_right",
        "joint_hip_left", "joint_foot_left", "joint_hip_right", "joint_foot_right",
        "joint_tail_base", "joint_tail_tip"])

    names = [n["name"] for n in read_glb(rewrite_glb_joint_names(glb))[0]["nodes"]]

    assert names == [
        "hips", "spine", "joint_chest", "neck", "head",
        "L_shoulder", "L_elbow", "L_hand", "R_shoulder", "R_elbow", "R_hand",
        "L_hip", "L_foot", "R_hip", "R_foot", "tail", "tail_tip"]


def test_root_stays_root_when_there_is_a_pelvis():
    glb = _skin_glb(["joint_root", "joint_pelvis", "joint_spine",
                     "joint_neck", "joint_head"])
    names = [n["name"] for n in read_glb(rewrite_glb_joint_names(glb))[0]["nodes"]]
    assert names[:2] == ["joint_root", "hips"]


def test_glb_rewriter_rigs_a_fishs_fins_as_arms():
    # seg_server's pufferfish: a spine, a dorsal fin, two pectoral fins, a
    # tail (add_fish_tail), no legs. The fins become the arm bones and the
    # tail the tail; the dorsal fin stays as it is.
    nodes = [{"name": n} for n in (
        "joint_pelvis", "joint_spine", "joint_chest", "joint_neck",
        "joint_head", "joint_dorsal_base", "joint_dorsal_tip",
        "joint_fin_base_left", "joint_fin_tip_left",
        "joint_fin_base_right", "joint_fin_tip_right",
        "joint_tail_base", "joint_tail_tip")]
    document = {"asset": {"version": "2.0"}, "nodes": nodes,
                "skins": [{"joints": list(range(len(nodes)))}],
                "scenes": [{"nodes": [0]}], "scene": 0}
    json_chunk = json.dumps(document).encode()
    json_chunk += b" " * (-len(json_chunk) % 4)
    body = (struct.pack("<I4s", len(json_chunk), b"JSON") + json_chunk
            + struct.pack("<I4s", 4, b"BIN\x00") + b"\0\0\0\0")
    glb = struct.pack("<4sII", b"glTF", 2, len(body) + 12) + body

    names = [n["name"] for n in read_glb(rewrite_glb_joint_names(glb))[0]["nodes"]]

    assert names == ["hips", "spine", "joint_chest", "neck", "head",
                     "joint_dorsal_base", "joint_dorsal_tip",
                     "L_shoulder", "L_elbow", "R_shoulder", "R_elbow",
                     "tail", "tail_tip"]
    assert not set(names) & LEG_BONES


def test_glb_rewriter_can_rename_without_verifying():
    rewritten = rewrite_glb_joint_names(make_glb(missing="head"), verify=False)
    names = {node["name"] for node in read_glb(rewritten)[0]["nodes"]}
    assert "hips" in names and "head" not in names


@pytest.mark.parametrize("mutate", [
    lambda value: b"bad!" + value[4:],
    lambda value: value[:8] + struct.pack("<I", len(value) + 4) + value[12:],
    lambda value: value[:20] + b"!" + value[21:],
])
def test_glb_rewriter_rejects_malformed_files(
    mutate: Callable[[bytes], bytes],
):
    with pytest.raises(GlbError):
        rewrite_glb_joint_names(mutate(make_glb()))
