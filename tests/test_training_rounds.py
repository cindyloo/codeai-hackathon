"""Rounds of training: "Keep training" carries on from an earlier round.

A child should be able to see practice add up — each round starting from the
pose the last one reached, so the match keeps climbing — and pick the round
they like best. Mock trainer throughout; see conftest.py.
"""

from __future__ import annotations

import statistics
import threading
import time

import pytest

import app as app_module
from providers.mock.posing import MockPoser
from providers.mock.training import MockTrainer
from rewards import pose_distance
from schemas import REST_POSE, TrainConfig


@pytest.fixture
def client():
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as test_client:
        yield test_client


def _final(clip, *, seed, start_pose=None, episodes=300):
    last = None
    for last in MockTrainer().train(
            None, clip, TrainConfig(episodes=episodes, seed=seed,
                                    start_pose=start_pose),
            threading.Event()):
        pass
    return last


def test_a_round_starts_from_the_pose_it_is_given():
    clip = MockPoser().pose("walks forward", None, lambda *_: None)
    start = _final(clip, seed=3).pose
    first = next(iter(MockTrainer().train(
        None, clip, TrainConfig(episodes=300, seed=4, start_pose=start),
        threading.Event())))
    # One step from where the last round ended, not from the T-pose.
    assert pose_distance(first.pose, start) < pose_distance(first.pose, REST_POSE)


def test_keep_training_never_loses_match_with_the_default_rewards():
    clip = MockPoser().pose("walks forward", None, lambda *_: None)
    for seed in range(1, 11):
        start, matches = None, []
        for r in range(3):
            last = _final(clip, seed=seed * 10 + r, start_pose=start)
            start = last.pose
            matches.append(last.match)
        assert matches == sorted(matches), f"seed {seed}: {matches}"


def test_more_rounds_get_closer_on_average():
    # Walk is the hardest default move: one round of 300 tries lands anywhere
    # from ~10% to ~55%. Three rounds should clearly beat one.
    clip = MockPoser().pose("walks forward", None, lambda *_: None)
    one, three = [], []
    for seed in range(1, 16):
        start = None
        for r in range(3):
            last = _final(clip, seed=seed * 10 + r, start_pose=start)
            start = last.pose
            if r == 0:
                one.append(last.match)
        three.append(last.match)
    assert statistics.mean(three) > statistics.mean(one) + 0.15


def test_match_is_measured_from_the_t_pose_in_every_round():
    clip = MockPoser().pose("arms in the air", None, lambda *_: None)
    first = _final(clip, seed=1)
    # A round that starts where the first ended and learns nothing more still
    # reports the first round's match, not 0%.
    still = next(iter(MockTrainer().train(
        None, clip, TrainConfig(episodes=1, seed=2, exploration=0.0,
                                start_pose=first.pose), threading.Event())))
    assert still.match >= first.match - 0.01


# -- the API -----------------------------------------------------------------

def _avatar_and_clip(client):
    from schemas import BONES, Rig
    rig = Rig(format="procedural", skeleton=list(BONES))
    avatar = app_module._add_avatar(rig).to_json()
    clip = client.post("/api/clips", json={
        "name": "Arms up", "fps": 24, "loop": True,
        "keyframes": [{"t": 0, "pose": MockPoser().pose(
            "arms in the air", None, lambda *_: None).keyframes[0].pose}],
    }).get_json()
    return avatar, clip


def _train(client, avatar, clip, **extra):
    response = client.post("/api/training/runs", json={
        "avatar_id": avatar["id"], "target_clip_id": clip["id"],
        "config": {"episodes": 40, "seed": 5}, "speed": 20, **extra})
    return response


def _finish(client, run_id):
    for _ in range(500):
        run = client.get(f"/api/training/runs/{run_id}").get_json()
        if run["status"] in ("done", "stopped", "error"):
            return run
        time.sleep(0.02)
    raise AssertionError("run never finished")


def test_keep_training_counts_rounds_and_starts_from_the_last_pose(client):
    avatar, clip = _avatar_and_clip(client)
    first = _train(client, avatar, clip)
    assert first.status_code == 201, first.get_json()
    first = first.get_json()
    assert (first["round"], first["continued_from"]) == (1, None)
    _finish(client, first["id"])

    second = _train(client, avatar, clip, continue_from=first["id"]).get_json()
    assert (second["round"], second["continued_from"]) == (2, first["id"])

    from export import final_pose
    from store import store
    assert store.get_run(second["id"]).cfg.start_pose == final_pose(
        store.get_run(first["id"]))
    _finish(client, second["id"])


def test_keep_training_from_a_missing_round_is_a_404(client):
    avatar, clip = _avatar_and_clip(client)
    response = _train(client, avatar, clip, continue_from="run_nope")
    assert response.status_code == 404


def test_move_smoothly_slows_learning_instead_of_freezing_it():
    # Penalising every bit of movement made standing still score best, so
    # with this slider up the learner never left the T-pose (0% every time).
    clip = MockPoser().pose("arms in the air", None, lambda *_: None)
    weights = {"pose_match": 1.0, "stillness": 0.5}
    finals, steps = [], []
    for seed in range(1, 11):
        previous = None
        for ep in MockTrainer().train(
                None, clip, TrainConfig(episodes=300, seed=seed,
                                        reward_weights=weights),
                threading.Event()):
            if previous:
                steps.append(pose_distance(ep.pose, previous))
            previous = ep.pose
        finals.append(ep.match)
    assert min(finals) > 0.05, finals
    # ...and it moves in smaller steps than it does without the slider.
    unsmoothed = []
    previous = None
    for ep in MockTrainer().train(None, clip, TrainConfig(episodes=300, seed=1),
                                  threading.Event()):
        if previous:
            unsmoothed.append(pose_distance(ep.pose, previous))
        previous = ep.pose
    assert max(steps) < max(unsmoothed)
