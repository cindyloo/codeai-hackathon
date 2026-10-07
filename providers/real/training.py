"""Real trainer: cross-entropy method (CEM) over per-joint rotations.

Unlike the mock's (1+1) hill-climb — one random jitter, kept if it helps — this
samples a whole population of candidate poses each generation from a Gaussian
per articulated joint, scores them all with rewards.reward(), and refits the
Gaussian to the best-scoring fraction (the "elites"). That refit is what makes
this "systematic": every generation's mean is a fit to genuine evidence about
what worked, not a single lucky guess, so the search reliably converges rather
than wandering. cfg.learning_rate controls how fast the mean/spread move
toward that new fit (0 = never update, 1 = jump straight to it) and
cfg.exploration sets the initial spread — the same two knobs the mock uses,
now driving a population search instead of a single random walk.

One Episode is yielded per candidate evaluated, in the order it was tried
(so the reward trace still looks like honest, noisy attempts), showing the
pose as the *current* generation's fitted mean — the population's individual
tries are what the reward curve reports, the displayed avatar is always the
agent's best current guess.
"""

from __future__ import annotations

import random
import threading
from typing import Iterator

from providers.base import Trainer
from rewards import per_joint_error, pose_distance, quat_from_euler, reward
from schemas import ARTICULATED_BONES, Clip, Episode, REST_POSE, Rig, TrainConfig

#: Candidates sampled per generation, and the fraction of them kept as elites
#: to refit the search distribution. 10/0.3 -> 3 elites, a population small
#: enough to stay cheap at 5000 episodes and large enough for the elite mean
#: to be a real signal rather than one lucky sample.
POPULATION = 10
ELITE_FRACTION = 0.3

#: Degrees. The Gaussian's spread never shrinks below this, so late in
#: training the search can still correct a joint it settled wrong on instead
#: of freezing — the same reason the mock never lets its exploration hit zero.
MIN_SIGMA_DEG = 4.0


def characteristic_pose(clip: Clip) -> dict[str, list[float]]:
    """The keyframe that best represents a clip — the one furthest from rest."""
    if not clip.keyframes:
        return dict(REST_POSE)
    return max(clip.keyframes,
               key=lambda kf: pose_distance(kf.pose, REST_POSE)).pose


def _decode(angles: dict[str, list[float]]) -> dict[str, list[float]]:
    """Per-joint Euler degrees -> a full 16-bone pose, rest for the rest."""
    pose = dict(REST_POSE)
    for bone in ARTICULATED_BONES:
        pose[bone] = list(quat_from_euler(*angles[bone]))
    return pose


class RealTrainer(Trainer):
    def train(self, rig: Rig, target: Clip, cfg: TrainConfig,
              stop: threading.Event) -> Iterator[Episode]:
        rng = random.Random(cfg.seed or None)
        goal = characteristic_pose(target)

        mu = {b: [0.0, 0.0, 0.0] for b in ARTICULATED_BONES}
        init_sigma = 15.0 + cfg.exploration * 45.0
        sigma = {b: [init_sigma] * 3 for b in ARTICULATED_BONES}

        current_pose = _decode(mu)
        best_reward, _ = reward(current_pose, goal, cfg.reward_weights)
        previous_pose: dict[str, list[float]] | None = None

        baseline = max(pose_distance(REST_POSE, goal), 1e-6)

        def match_of(pose: dict[str, list[float]]) -> float:
            return max(0.0, min(1.0, 1.0 - pose_distance(pose, goal) / baseline))

        episode = 0
        while episode < cfg.episodes:
            batch = min(POPULATION, cfg.episodes - episode)
            tried: list[tuple[dict[str, list[float]], dict[str, list[float]], float]] = []
            for _ in range(batch):
                if stop.is_set():
                    break
                angles = {b: [rng.gauss(mu[b][i], sigma[b][i]) for i in range(3)]
                          for b in ARTICULATED_BONES}
                pose = _decode(angles)
                r, _ = reward(pose, goal, cfg.reward_weights, previous=previous_pose)
                tried.append((angles, pose, r))

            if not tried:
                break  # stop was set before this generation could start

            # Refit the search distribution to the elites: this is the whole
            # method — everything else is bookkeeping for the live UI.
            elite_n = max(1, round(len(tried) * ELITE_FRACTION))
            elites = sorted(tried, key=lambda t: t[2], reverse=True)[:elite_n]
            for bone in ARTICULATED_BONES:
                for i in range(3):
                    values = [e[0][bone][i] for e in elites]
                    elite_mean = sum(values) / len(values)
                    elite_std = (sum((v - elite_mean) ** 2 for v in values)
                                 / len(values)) ** 0.5
                    mu[bone][i] += cfg.learning_rate * (elite_mean - mu[bone][i])
                    sigma[bone][i] += cfg.learning_rate * (
                        max(MIN_SIGMA_DEG, elite_std) - sigma[bone][i])

            current_pose = _decode(mu)
            mean_reward, _ = reward(current_pose, goal, cfg.reward_weights,
                                    previous=previous_pose)
            best_reward = max(best_reward, mean_reward)
            avg_sigma = sum(sigma[b][i] for b in ARTICULATED_BONES for i in range(3)) \
                / (len(ARTICULATED_BONES) * 3)
            per_joint = per_joint_error(current_pose, goal)
            match = match_of(current_pose)

            stopped_early = len(tried) < batch
            for angles, pose, r in tried:
                episode += 1
                note = ""
                if r > best_reward:
                    best_reward = r
                    note = "New best!"
                yield Episode(
                    episode=episode,
                    reward=r,
                    best_reward=best_reward,
                    pose=current_pose,
                    per_joint_error=per_joint,
                    exploration=round(avg_sigma / 36.0, 3),
                    done=(episode == cfg.episodes) and not stopped_early,
                    note=note,
                    match=match,
                )

            if stopped_early:
                yield Episode(episode=episode, reward=mean_reward,
                              best_reward=best_reward, pose=current_pose,
                              per_joint_error=per_joint, exploration=0.0,
                              done=True, note="Stopped", match=match)
                return

            previous_pose = current_pose
