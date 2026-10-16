// Pose/clip maths shared by the viewport and the training screen.
// Mirrors the server-side helpers in rewards.py — keep the two in step.

import * as THREE from "three";

const _a = new THREE.Quaternion();
const _b = new THREE.Quaternion();
const _out = new THREE.Quaternion();

const IDENTITY = [0, 0, 0, 1];

/** Seconds between poses when a sequence of them plays as a move. */
export const POSE_SECONDS = 0.8;

/**
 * Poses in order -> keyframes, one every POSE_SECONDS. Two or more end on the
 * first pose again so the loop doesn't snap back; a single pose stays a
 * single keyframe, which plays as T-pose -> pose -> T-pose.
 *
 * `stretches`, if given, is one squash/stretch amount per pose (see
 * Keyframe.stretch in schemas.py); zero is left out of the keyframe.
 */
export function sequenceKeyframes(poses, stretches = []) {
  const frames = poses.map((pose, i) => {
    const frame = { t: i * POSE_SECONDS, pose };
    if (stretches[i]) frame.stretch = stretches[i];
    return frame;
  });
  if (frames.length > 1) {
    frames.push({ ...frames[0], t: frames.length * POSE_SECONDS });
  }
  return frames;
}

/**
 * A clip that visibly moves. A single pose (what training saves) is one
 * keyframe, which samples to a statue; play it as T-pose -> pose -> hold ->
 * T-pose instead, on a loop. Timing matches ANIMATION_TIMING in export.py so
 * the browser and the downloaded GLB perform the same move.
 */
export function performableClip(clip) {
  if (clip?.keyframes?.length !== 1) return clip;
  const { pose, stretch = 0 } = clip.keyframes[0];
  const rest = Object.fromEntries(Object.keys(pose).map((b) => [b, IDENTITY]));
  return {
    ...clip,
    loop: true,
    duration: 3,
    keyframes: [
      { t: 0, pose: rest },
      { t: 1, pose, stretch },
      { t: 2, pose, stretch },
      { t: 3, pose: rest },
    ],
  };
}

/** A clip's squash/stretch at time `t`, interpolated linearly; 0 if none. */
export function sampleStretch(clip, t) {
  const frames = clip?.keyframes;
  if (!frames?.length) return 0;
  if (frames.length === 1) return frames[0].stretch || 0;

  const duration = clip.duration || frames[frames.length - 1].t;
  const time = clip.loop && duration > 0 ? t % duration : Math.min(t, duration);

  let index = 0;
  while (index < frames.length - 2 && frames[index + 1].t <= time) index += 1;

  const from = frames[index];
  const to = frames[index + 1];
  const span = to.t - from.t;
  const alpha = span > 0 ? Math.min(Math.max((time - from.t) / span, 0), 1) : 0;
  return (from.stretch || 0) + ((to.stretch || 0) - (from.stretch || 0)) * alpha;
}

/** Sample a clip at time `t` (seconds), interpolating between keyframes. */
export function samplePose(clip, t) {
  const frames = clip.keyframes;
  if (!frames?.length) return {};
  if (frames.length === 1) return frames[0].pose;

  const duration = clip.duration || frames[frames.length - 1].t;
  const time = clip.loop && duration > 0 ? t % duration : Math.min(t, duration);

  let index = 0;
  while (index < frames.length - 2 && frames[index + 1].t <= time) index += 1;

  const from = frames[index];
  const to = frames[index + 1];
  const span = to.t - from.t;
  const alpha = span > 0 ? (time - from.t) / span : 0;

  const pose = {};
  for (const bone of Object.keys(from.pose)) {
    const q0 = from.pose[bone];
    const q1 = to.pose[bone] || q0;
    _a.set(q0[0], q0[1], q0[2], q0[3]);
    _b.set(q1[0], q1[1], q1[2], q1[3]);
    _out.slerpQuaternions(_a, _b, alpha);
    pose[bone] = [_out.x, _out.y, _out.z, _out.w];
  }
  return pose;
}

/** Angle in radians between two quaternions, 0..PI. */
export function quatAngle(a, b) {
  const dot = Math.abs(a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]);
  return 2 * Math.acos(Math.min(1, Math.max(-1, dot)));
}

/**
 * The keyframe that best represents a clip — the one furthest from rest.
 *
 * This must match characteristic_pose() in providers/mock/training.py, because
 * it is what the training screen shows as "the target" and the server trains
 * towards. If they disagree, a child sees the avatar learning a pose that isn't
 * the one on screen.
 */
export function characteristicPose(clip, restPose) {
  if (!clip?.keyframes?.length) return { ...restPose };
  let best = clip.keyframes[0];
  let bestDistance = -1;
  for (const frame of clip.keyframes) {
    const distance = poseDistance(frame.pose, restPose);
    if (distance > bestDistance) {
      bestDistance = distance;
      best = frame;
    }
  }
  return best.pose;
}

/** Mirrors pose_distance() in rewards.py: half the mean error, half the worst. */
export function poseDistance(current, target, bones) {
  const names = bones || Object.keys(target);
  if (!names.length) return 0;
  let sum = 0;
  let worst = 0;
  for (const bone of names) {
    const error = quatAngle(current[bone] || [0, 0, 0, 1],
                            target[bone] || [0, 0, 0, 1]) / Math.PI;
    sum += error;
    worst = Math.max(worst, error);
  }
  return 0.5 * (sum / names.length) + 0.5 * worst;
}

// -- combining moves ----------------------------------------------------

/** Frames per second a combined move is sampled at, like an imported one. */
export const COMBINE_FPS = 12;
/** The contract's limits on a clip (schemas.MAX_KEYFRAMES, MAX_CLIP_DURATION). */
const MAX_KEYFRAMES = 240;
const MAX_SECONDS = 20;

/** How long one pass of a clip lasts, as it plays (a single pose: 3s). */
export function clipSeconds(clip) {
  const played = performableClip(clip);
  return played.duration || played.keyframes.at(-1)?.t || 0;
}

/** Does `clip` move any of `bones` (or, with `stretch`, squash at all)? */
export function clipMoves(clip, { bones = [], stretch = false } = {}) {
  return clip.keyframes.some((k) => (stretch
    ? Boolean(k.stretch)
    : bones.some((b) => k.pose[b] && quatAngle(k.pose[b], IDENTITY) > 0.02)));
}

/**
 * Layer moves by body part into one move.
 *
 * `layers` is a list of { bones, clip, speed, stretch }: each layer's bones
 * come from its own clip, played at its own speed and looped; bones in no
 * layer stay at rest. A layer with `stretch` supplies the whole-body squash
 * and stretch instead. The result lasts one pass of the longest layer, so
 * shorter ones repeat inside it — a 0.5s flap over a 1s wag flaps twice —
 * and every layer is looped, so the combined move loops too.
 *
 * Returns keyframes for a clip, sampled at COMBINE_FPS.
 */
export function combineClips(layers, restPose) {
  const active = layers
    .filter((l) => l.clip)
    .map((l) => {
      const speed = l.speed || 1;
      const clip = { ...performableClip(l.clip), loop: true };
      clip.duration = clipSeconds(l.clip);
      return { ...l, speed, clip, seconds: clip.duration / speed };
    })
    .filter((l) => l.seconds > 0);
  if (!active.length) return [];

  const seconds = Math.min(Math.max(...active.map((l) => l.seconds)), MAX_SECONDS);
  const count = Math.min(Math.round(seconds * COMBINE_FPS) + 1, MAX_KEYFRAMES);
  const frames = [];
  for (let i = 0; i < count; i += 1) {
    const t = count > 1 ? (i * seconds) / (count - 1) : 0;
    const pose = { ...restPose };
    let stretch = 0;
    for (const layer of active) {
      const local = t * layer.speed;
      if (layer.stretch) {
        stretch = sampleStretch(layer.clip, local);
        continue;
      }
      const sampled = samplePose(layer.clip, local);
      for (const bone of layer.bones) pose[bone] = sampled[bone] || IDENTITY;
    }
    const frame = { t: Math.round(t * 1000) / 1000, pose };
    if (Math.abs(stretch) > 1e-4) frame.stretch = Math.round(stretch * 1000) / 1000;
    frames.push(frame);
  }
  return frames;
}
