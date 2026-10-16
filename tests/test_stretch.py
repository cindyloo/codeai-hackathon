"""Squash and stretch: a whole-figure scale per keyframe, made on Teach.

It rides on Keyframe.stretch, plays in the viewport, and bakes into the GLB
as a scale animation on a pivot node at the model's feet — which comes back
as stretch when the file is loaded in again.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

import app as app_module
import gltf
from schemas import MAX_STRETCH, ContractError, Keyframe, validate_clip

FIXTURE = Path(__file__).parent / "fixtures" / "mixamo-style.glb"


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def _clip(*stretches, t_step=0.8):
    return {"name": "Bounce", "fps": 24, "loop": True, "keyframes": [
        {"t": i * t_step, "pose": {}, **({"stretch": s} if s else {})}
        for i, s in enumerate(stretches)]}


# -- the contract --------------------------------------------------------------

def test_a_keyframe_carries_its_stretch():
    clip = validate_clip(_clip(0.3, -0.25))
    assert [k.stretch for k in clip.keyframes] == [0.3, -0.25]
    assert clip.to_json()["keyframes"][0]["stretch"] == 0.3


def test_no_stretch_leaves_the_json_as_it_was():
    clip = validate_clip(_clip(0.0, 0.0))
    assert all("stretch" not in k for k in clip.to_json()["keyframes"])


@pytest.mark.parametrize("bad", [MAX_STRETCH + 0.01, -0.9, "lots", float("nan")])
def test_a_wild_stretch_is_refused(bad):
    with pytest.raises(ContractError, match="stretch"):
        validate_clip(_clip(bad))


def test_stretching_keeps_the_volume():
    for stretch in (-0.4, 0.0, 0.25, 0.5):
        x, y, z = gltf.stretch_scale(stretch)
        assert y == pytest.approx(1 + stretch)
        assert x * y * z == pytest.approx(1.0)


# -- the GLB -------------------------------------------------------------------

def _sideload(client, data):
    response = client.post("/api/avatars/glb",
                           data={"glb": (io.BytesIO(data), "avatar.glb")})
    assert response.status_code == 201, response.get_json()
    return response.get_json()


def _save(client, avatar_id, clip):
    response = client.post("/api/behaviours", json={
        "avatar_id": avatar_id, "name": clip["name"], "clip": clip})
    assert response.status_code in (200, 201), response.get_json()


def _download(client, avatar_id):
    response = client.get(f"/api/avatars/{avatar_id}/export.glb")
    assert response.status_code == 200, response.get_json()
    return response.data


def test_a_stretching_move_bakes_a_scale_on_a_pivot_at_the_feet(client):
    avatar = _sideload(client, FIXTURE.read_bytes())
    _save(client, avatar["id"], _clip(0.0, 0.4, -0.3))

    document, binary = gltf.read_glb(_download(client, avatar["id"]))

    pivot = gltf.find_stretch_pivot(document)
    assert pivot is not None
    # Everything hangs under the pivot, which stands on the model's lowest point.
    assert document["scenes"][document.get("scene", 0)]["nodes"] == [pivot]
    original, _ = gltf.read_glb(FIXTURE.read_bytes())
    assert document["nodes"][pivot]["translation"][1] == pytest.approx(
        gltf._ground_point(original)[1])

    [bounce] = [a for a in document["animations"] if a["name"] == "Bounce"]
    [channel] = [c for c in bounce["channels"] if c["target"]["node"] == pivot]
    assert channel["target"]["path"] == "scale"
    values = gltf._read_accessor(
        document, binary, bounce["samplers"][channel["sampler"]]["output"])
    assert [round(v[1], 4) for v in values] == [1.0, 1.4, 0.7]


def test_stretch_comes_back_when_the_file_is_loaded_again(client):
    avatar = _sideload(client, FIXTURE.read_bytes())
    _save(client, avatar["id"], _clip(0.0, 0.4, -0.3))

    again = _sideload(client, _download(client, avatar["id"]))

    [bounce] = [c for c in again["clips"] if c["name"] == "Bounce"]
    stretches = [k.get("stretch", 0.0) for k in bounce["keyframes"]]
    # Imports resample at 12 fps, rotations and stretch alike, so a peak
    # between two samples comes back a little rounded off.
    assert max(stretches) == pytest.approx(0.4, abs=0.03)
    assert min(stretches) == pytest.approx(-0.3, abs=0.03)


def test_exporting_twice_keeps_one_pivot(client):
    avatar = _sideload(client, FIXTURE.read_bytes())
    _save(client, avatar["id"], _clip(0.0, 0.4))
    again = _sideload(client, _download(client, avatar["id"]))
    _save(client, again["id"], _clip(0.0, -0.2))

    document, _ = gltf.read_glb(_download(client, again["id"]))

    pivots = [n for n in document["nodes"] if n.get("name") == gltf.STRETCH_PIVOT]
    assert len(pivots) == 1


def test_once_there_is_a_pivot_every_baked_move_drives_it(client):
    # Otherwise a player switching from "Bounce" to "Wave" would wave squashed.
    avatar = _sideload(client, FIXTURE.read_bytes())
    _save(client, avatar["id"], _clip(0.0, 0.4))
    _save(client, avatar["id"], {**_clip(0.0, 0.0), "name": "Still"})

    document, _ = gltf.read_glb(_download(client, avatar["id"]))

    pivot = gltf.find_stretch_pivot(document)
    for name in ("Bounce", "Still"):
        [animation] = [a for a in document["animations"] if a["name"] == name]
        assert any(c["target"] == {"node": pivot, "path": "scale"}
                   for c in animation["channels"]), name


def test_without_stretch_the_file_gets_no_pivot(client):
    avatar = _sideload(client, FIXTURE.read_bytes())
    _save(client, avatar["id"], _clip(0.0, 0.0))

    document, _ = gltf.read_glb(_download(client, avatar["id"]))

    assert gltf.find_stretch_pivot(document) is None


def test_the_pivot_is_not_mistaken_for_a_bone():
    document, _ = gltf.read_glb(FIXTURE.read_bytes())
    before = gltf.resolve_bones(document)
    gltf.ensure_stretch_pivot(document)
    assert gltf.resolve_bones(document) == before


def test_a_learned_single_pose_keeps_its_stretch_through_the_hold(client):
    import export
    from schemas import Clip
    clip = Clip(name="Tall", keyframes=[Keyframe(t=0, pose={}, stretch=0.3)])
    assert [s for _t, _p, s in export.clip_keyframes(clip)] == [0.0, 0.3, 0.3, 0.0]
