"""The real animation path, on rigs seg_server actually produced.

Everything else in the suite runs on mocks and synthetic fixtures. These
fetch real rigged GLBs from the seg_server in .env — the broccoli and the
tomato, by default — and check what the synthetic fixtures can't: seg_server's
joint names, its armless biped, its baked idle/walk/wave clips, and the export
of a learned move back into one of its files.

Run with:   pytest --real
Pick rigs:  REAL_TEST_RIGS=da008649,60964d1d pytest --real

They read cached results only (GET /results/<id>/<id>_rigged.glb): no
classification, no rigging, no LLM spend. A rig the server doesn't have is
skipped, not failed.
"""

from __future__ import annotations

import io
import math
import os
import threading
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

import app as app_module
import config
import export
import gltf
import providers
from providers.real.rigging import rewrite_glb_joint_names
from schemas import BONES, OPTIONAL_BONES, TrainConfig
from store import store
from training import TrainingRun

pytestmark = pytest.mark.real

#: The broccoli (arms, idle/walk/wave), the tomato (no arm joints), the
#: pufferfish (fins as arms, no legs) and a bee (the flying skeleton: wings
#: as arms, no pelvis, no legs).
RIGS = [r.strip() for r in os.environ.get(
    "REAL_TEST_RIGS", "da008649,60964d1d,865fee5d,cea1a599").split(",") if r.strip()]

#: Which limbs each known rig has, as the Teach sliders should show them.
EXPECTED_LIMBS = {
    "da008649": {"arms": True, "legs": True, "tail": False},    # broccoli
    "60964d1d": {"arms": False, "legs": True, "tail": False},   # tomato
    "865fee5d": {"arms": True, "legs": False, "tail": True},    # pufferfish
    "cea1a599": {"arms": True, "legs": False, "tail": False},   # bee
}


def _get(path: str) -> bytes:
    request = Request(f"{config.RIGGING_SERVICE_URL}{path}",
                      headers={"ngrok-skip-browser-warning": "true"})
    with urlopen(request, timeout=60) as response:
        return response.read()


@pytest.fixture(scope="session")
def seg_server():
    if not config.RIGGING_SERVICE_URL:
        pytest.skip("RIGGING_SERVICE_URL isn't set in .env")
    try:
        _get("/classify")      # GET answers {"status": "ok"}
    except (HTTPError, URLError, OSError) as exc:
        pytest.skip(f"seg_server at {config.RIGGING_SERVICE_URL} is unreachable: {exc}")


@pytest.fixture(scope="session", params=RIGS)
def rig_glb(request, seg_server):
    rig_id = request.param
    try:
        raw = _get(f"/results/{rig_id}/{rig_id}_rigged.glb")
    except HTTPError as exc:
        pytest.skip(f"seg_server has no rig {rig_id!r} (HTTP {exc.code})")
    if not raw.startswith(b"glTF"):
        pytest.skip(f"seg_server's rig {rig_id!r} isn't a GLB")
    return rig_id, raw


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def _degrees(q) -> float:
    return math.degrees(2 * math.acos(min(1.0, abs(q[3]))))


def _swing(clip, bone) -> float:
    """The largest rotation a clip gives a bone, in degrees."""
    return max(_degrees(k["pose"][bone]) for k in clip["keyframes"]
               if bone in k["pose"])


def test_joints_rename_to_a_valid_contract_rig(rig_glb):
    _, raw = rig_glb
    bones = gltf.validate_avatar_glb(rewrite_glb_joint_names(raw, verify=False))
    # Every required bone resolves; only the arm bones may be absent.
    assert set(BONES) - set(bones) <= OPTIONAL_BONES


def test_the_avatar_lists_the_bones_its_model_has(client, rig_glb):
    rig_id, raw = rig_glb
    if rig_id not in EXPECTED_LIMBS:
        pytest.skip(f"no expectation written down for {rig_id}")
    avatar = client.post("/api/avatars/glb",
                         data={"glb": (io.BytesIO(raw), f"{rig_id}.glb")}).get_json()
    bones = set(avatar["bones"])

    assert {"hips", "spine", "neck", "head"} <= bones
    limbs = {"arms": "L_shoulder", "legs": "L_hip", "tail": "tail"}
    for limb, has in EXPECTED_LIMBS[rig_id].items():
        assert (limbs[limb] in bones) == has, f"{rig_id}: {limb} {sorted(bones)}"


def test_its_baked_clips_import_as_moves_that_move(rig_glb):
    rig_id, raw = rig_glb
    document, binary = gltf.read_glb(rewrite_glb_joint_names(raw, verify=False))
    clips = {c["name"].lower(): c for c in gltf.read_clips(document, binary)}

    assert clips, f"{rig_id} came with no moves"
    # Something that walks bakes a walk that swings both legs, visibly. A
    # fish has no legs, and what seg_server bakes for it varies by run (its
    # pufferfish has come back with fold/idle/walk/wave and with idle/wave).
    if "L_hip" in next(iter(clips.values()))["keyframes"][0]["pose"]:
        assert "walk" in clips, f"{rig_id} has no walk clip: {sorted(clips)}"
        for bone in ("L_hip", "R_hip"):
            assert _swing(clips["walk"], bone) > 15, f"{rig_id} walk barely moves {bone}"

    if "wave" in clips:
        wave = clips["wave"]
        # A fish waves a fin through ~29°, an arm much further.
        assert max(_swing(wave, "L_shoulder"), _swing(wave, "R_shoulder")) > 20


def test_sideloading_it_gives_an_avatar_with_its_moves(client, rig_glb):
    rig_id, raw = rig_glb
    response = client.post("/api/avatars/glb",
                           data={"glb": (io.BytesIO(raw), f"{rig_id}.glb")})
    assert response.status_code == 201, response.get_json()
    avatar = response.get_json()

    names = {c["name"] for c in avatar["clips"]}
    assert names
    behaviours = client.get(
        f"/api/behaviours?avatar_id={avatar['id']}").get_json()["behaviours"]
    assert {b["name"] for b in behaviours if b["source"] == "model"} == names


def test_a_learned_move_exports_into_it_and_comes_back(client, rig_glb):
    rig_id, raw = rig_glb
    avatar = store.get_avatar(client.post(
        "/api/avatars/glb",
        data={"glb": (io.BytesIO(raw), f"{rig_id}.glb")}).get_json()["id"])
    clip = store.add_clip(providers.get_poser().pose(
        "jumps up high", avatar.rig, lambda *_: None))

    run = TrainingRun(avatar.id, clip, TrainConfig(episodes=60, seed=7))
    store.add_run(run)
    for episode in providers.get_trainer().train(
            avatar.rig, clip, run.cfg, threading.Event()):
        run._history.append(episode)
        run.best_reward = max(run.best_reward, episode.best_reward)

    data, _ = export.build_glb(run, avatar, clip)
    document, binary = gltf.read_glb(data)
    names = [c["name"] for c in gltf.read_clips(document, binary)]

    # The learned move leads; the rig's own clips survive behind it.
    assert names[0] == "Jump (learned)"
    own = [c["name"] for c in gltf.read_clips(*gltf.read_glb(
        rewrite_glb_joint_names(raw, verify=False)))]
    assert [n.lower() for n in names[1:]] == [n.lower() for n in own]
