"""Moves built on the Teach step's sliders are saved like any other move."""

from __future__ import annotations

import pytest

import app as app_module
from rewards import quat_from_euler
from store import store

ARMS_UP = {"L_shoulder": list(quat_from_euler(0, 0, -90)),
           "R_shoulder": list(quat_from_euler(0, 0, 90))}


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def test_saves_a_hand_built_move(client):
    response = client.post("/api/clips", json={
        "name": "  Arms up then down  ", "fps": 24, "loop": True,
        "keyframes": [{"t": 0, "pose": ARMS_UP}, {"t": 0.8, "pose": {}},
                      {"t": 1.6, "pose": ARMS_UP}],
    })

    assert response.status_code == 201
    clip = response.get_json()
    assert clip["name"] == "Arms up then down"
    assert clip["duration"] == 1.6
    assert store.get_clip(clip["id"]) is not None


def test_a_single_pose_is_a_move(client):
    response = client.post("/api/clips", json={
        "name": "Arms up", "keyframes": [{"t": 0, "pose": ARMS_UP}]})
    assert response.status_code == 201


@pytest.mark.parametrize("body", [
    {"name": "", "keyframes": [{"t": 0, "pose": ARMS_UP}]},
    {"name": "Nothing", "keyframes": []},
    {"name": "Bad bone", "keyframes": [{"t": 0, "pose": {"tail": [0, 0, 0, 1]}}]},
])
def test_refuses_a_move_it_cannot_play(client, body):
    response = client.post("/api/clips", json=body)
    assert response.status_code == 400
    assert response.get_json()["error"]
