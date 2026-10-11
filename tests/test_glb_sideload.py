"""A compatible GLB can enter the experience without running phase 1."""

from __future__ import annotations

import io
import time
from pathlib import Path

import pytest

import app as app_module
import gltf
import providers
from jobs import runner
from rewards import quat_from_euler
from schemas import BONES, OPTIONAL_BONES, Rig
from store import store

FIXTURE = Path(__file__).parent / "fixtures" / "mixamo-style.glb"


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    existing = set(store.avatars)
    with app_module.app.test_client() as test_client:
        yield test_client
    for avatar_id in set(store.avatars) - existing:
        store.avatars.pop(avatar_id, None)


def post_glb(client, data: bytes, filename: str = "avatar.glb"):
    return client.post(
        "/api/avatars/glb",
        data={"glb": (io.BytesIO(data), filename)},
    )


def test_sideload_creates_a_glb_avatar_without_calling_the_rigger(
        client, monkeypatch):
    def unexpected_rigger():
        pytest.fail("sideloading a GLB must not invoke the phase 1 rigger")

    monkeypatch.setattr(providers, "get_rigger", unexpected_rigger)
    glb_bytes = FIXTURE.read_bytes()

    response = post_glb(client, glb_bytes)

    assert response.status_code == 201
    avatar_json = response.get_json()
    assert avatar_json["image_url"] is None
    assert avatar_json["rig"]["format"] == "glb"
    assert avatar_json["rig"]["glb_url"].endswith("/glb")
    assert avatar_json["rig"]["skeleton"] == BONES

    avatar = store.get_avatar(avatar_json["id"])
    assert avatar is not None
    assert avatar.rig.glb_bytes == glb_bytes

    downloaded = client.get(avatar_json["rig"]["glb_url"])
    assert downloaded.status_code == 200
    assert downloaded.mimetype == "model/gltf-binary"
    assert downloaded.data == glb_bytes


def test_sideload_is_not_subject_to_the_sketch_upload_limit(client):
    previous_limit = app_module.app.config["MAX_CONTENT_LENGTH"]
    app_module.app.config["MAX_CONTENT_LENGTH"] = 1024
    try:
        response = post_glb(client, FIXTURE.read_bytes())
    finally:
        app_module.app.config["MAX_CONTENT_LENGTH"] = previous_limit

    assert response.status_code == 201


def test_sideload_accepts_mixamo_names_without_the_mixamorig_prefix(client):
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    for node in document["nodes"]:
        name = node.get("name", "")
        if name.startswith("mixamorig:"):
            node["name"] = name.split(":", 1)[1]

    response = post_glb(client, gltf.write_glb(document, binary))

    assert response.status_code == 201


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "No GLB was uploaded."),
        ({"glb": (io.BytesIO(b""), "empty.glb")}, "That GLB was empty."),
        ({"glb": (io.BytesIO(b"not a glb"), "broken.glb")},
         "That GLB can't be used:"),
    ],
)
def test_sideload_rejects_missing_empty_and_malformed_files(
        client, data, message):
    response = client.post("/api/avatars/glb", data=data)

    assert response.status_code == 400
    assert message in response.get_json()["error"]


def unskinned_glb() -> bytes:
    return gltf.write_glb({
        "asset": {"version": "2.0"},
        "nodes": [{"name": bone} for bone in BONES],
        "scenes": [{"nodes": list(range(len(BONES)))}],
        "scene": 0,
    }, b"")


def wait_for_job(job_id: str):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        job = runner.get(job_id)
        if job and job.status not in ("queued", "running"):
            return job
        time.sleep(0.01)
    pytest.fail(f"job {job_id} did not finish")


def test_an_unskinned_model_is_sent_to_the_rigger(client, monkeypatch):
    received = {}

    class AutoRigger:
        def rig_glb(self, glb_bytes, progress):
            received["glb"] = glb_bytes
            progress(0.5, "Rigging...")
            return Rig(format="glb", skeleton=list(BONES),
                       glb_bytes=FIXTURE.read_bytes())

    monkeypatch.setattr(providers, "get_rigger", lambda: AutoRigger())
    upload = unskinned_glb()

    response = post_glb(client, upload)

    assert response.status_code == 202
    job = wait_for_job(response.get_json()["id"])
    assert job.status == "done", job.error
    assert received["glb"] == upload
    assert job.result["id"] in store.avatars
    assert job.result["rig"]["format"] == "glb"


def test_an_unskinned_model_fails_kindly_without_an_auto_rigger(
        client, monkeypatch):
    from providers.mock.rigging import MockRigger

    monkeypatch.setattr(providers, "get_rigger", lambda: MockRigger())

    response = post_glb(client, unskinned_glb())

    assert response.status_code == 202
    job = wait_for_job(response.get_json()["id"])
    assert job.status == "error"
    assert "no skeleton" in job.error


def test_sideload_rejects_invalid_node_names_cleanly(client):
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    document["nodes"][0]["name"] = 42

    response = post_glb(client, gltf.write_glb(document, binary))

    assert response.status_code == 400
    assert "node with an invalid name" in response.get_json()["error"]


def test_sideload_names_the_missing_required_joint(client):
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    head_index = gltf.resolve_bones(document)["head"]
    document["nodes"][head_index]["name"] = "unsupported_head_joint"

    response = post_glb(client, gltf.write_glb(document, binary))

    assert response.status_code == 400
    assert "missing required joints: head" in response.get_json()["error"]


def test_sideload_accepts_a_rig_without_arms(client):
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    bones = gltf.resolve_bones(document)
    for bone in OPTIONAL_BONES:
        document["nodes"][bones[bone]]["name"] = f"unmapped_{bone}"

    response = post_glb(client, gltf.write_glb(document, binary))

    assert response.status_code == 201


def test_sideload_renames_seg_server_joints(client):
    # What a rig downloaded straight from seg_server looks like.
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    remote = {"hips": "joint_pelvis", "spine": "joint_spine",
              "neck": "joint_neck", "head": "joint_head"}
    for bone, index in gltf.resolve_bones(document).items():
        side = {"L": "left", "R": "right"}.get(bone[0]) if bone[1] == "_" else None
        document["nodes"][index]["name"] = remote.get(
            bone, f"joint_{bone[2:]}_{side}")

    response = post_glb(client, gltf.write_glb(document, binary))

    assert response.status_code == 201
    stored = store.get_avatar(response.get_json()["id"]).rig.glb_bytes
    assert set(gltf.resolve_bones(gltf.read_glb(stored)[0])) == set(BONES)


def animated_fixture(name="walk"):
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    bones = gltf.resolve_bones(document)
    up = {"L_shoulder": list(quat_from_euler(0, 0, -90))}
    binary = gltf.add_pose_animation(
        document, binary, [(0.0, {}), (1.0, up), (2.0, {})], bones, name=name)
    return gltf.write_glb(document, binary)


def test_sideload_imports_the_glbs_animations_as_moves(client):
    response = post_glb(client, animated_fixture())

    assert response.status_code == 201
    avatar = response.get_json()
    [clip] = avatar["clips"]
    assert clip["name"] == "Walk"
    assert clip["loop"] is True
    assert clip["duration"] == 2.0
    assert store.get_clip(clip["id"]) is not None

    behaviours = client.get(f"/api/behaviours?avatar_id={avatar['id']}").get_json()
    [behaviour] = behaviours["behaviours"]
    assert behaviour["source"] == "model"
    assert behaviour["trained"] is False


def test_sideload_without_animations_has_no_moves(client):
    response = post_glb(client, FIXTURE.read_bytes())
    assert response.get_json()["clips"] == []


def two_move_fixture():
    document, binary = gltf.read_glb(FIXTURE.read_bytes())
    bones = gltf.resolve_bones(document)
    up = {"L_shoulder": list(quat_from_euler(0, 0, -90))}
    kick = {"L_hip": list(quat_from_euler(-40, 0, 0))}
    for name, pose in (("kick", kick), ("wave", up)):   # each goes first
        binary = gltf.add_pose_animation(
            document, binary, [(0.0, {}), (1.0, pose), (2.0, {})], bones, name=name)
    return gltf.write_glb(document, binary)


def exported_animation_names(avatar_id):
    import threading

    import export
    from schemas import TrainConfig
    from training import TrainingRun

    avatar = store.get_avatar(avatar_id)
    clip = store.add_clip(providers.get_poser().pose(
        "waves arms in the air", avatar.rig, lambda *_: None))
    run = TrainingRun(avatar.id, clip, TrainConfig(episodes=20, seed=1))
    for episode in providers.get_trainer().train(
            avatar.rig, clip, run.cfg, threading.Event()):
        run._history.append(episode)
    data, _ = export.build_glb(run, avatar, clip)
    return [a["name"] for a in gltf.read_glb(data)[0]["animations"]]


def test_deleting_a_model_move_leaves_it_out_of_the_export(client):
    avatar = post_glb(client, two_move_fixture()).get_json()
    behaviours = client.get(f"/api/behaviours?avatar_id={avatar['id']}").get_json()
    wave = next(b for b in behaviours["behaviours"] if b["name"] == "Wave")

    response = client.delete(f"/api/behaviours/{wave['id']}")

    assert response.status_code == 200
    assert [c["name"] for c in response.get_json()["avatar"]["clips"]] == ["Kick"]
    assert exported_animation_names(avatar["id"]) == ["Arms in the air (learned)", "kick"]


def test_without_deletions_the_export_keeps_every_model_move(client):
    avatar = post_glb(client, two_move_fixture()).get_json()
    assert exported_animation_names(avatar["id"]) == [
        "Arms in the air (learned)", "wave", "kick"]


def test_deleting_a_missing_behaviour_is_a_404(client):
    assert client.delete("/api/behaviours/bhv_nope").status_code == 404


def test_sideload_works_when_the_server_terminates_the_input_stream(client):
    # gunicorn (production) sets wsgi.input_terminated, which makes Werkzeug
    # treat max_content_length as a hard cap on the stream; a cap equal to the
    # exact body size used to 413 every upload there, while the dev server and
    # the plain test client never saw it.
    response = client.post(
        "/api/avatars/glb",
        data={"glb": (io.BytesIO(FIXTURE.read_bytes()), "avatar.glb")},
        environ_overrides={"wsgi.input_terminated": True},
    )

    assert response.status_code == 201, response.get_json()


def test_an_oversized_glb_names_the_glb_limit(client, monkeypatch):
    monkeypatch.setattr(app_module.config, "MAX_GLB_UPLOAD_BYTES", 1024 * 1024)
    response = post_glb(client, b"x" * (2 * 1024 * 1024))

    assert response.status_code == 413
    assert "GLB is too big" in response.get_json()["error"]
