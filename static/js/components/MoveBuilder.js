// Build a move by hand: a slider for each way the head, arms and legs can
// move, a strip of poses captured from them, and a save button.
//
// Sliders are in plain degrees from the T-pose and are turned into contract
// rotations here, using the conventions in CONTRACT.md. The parent shows the
// result: a live pose while sliders move, or the poses played as a clip.

import * as THREE from "three";
import { computed, reactive, ref } from "vue";
import { api } from "../api.js";
import { sequenceKeyframes } from "../pose.js";
import { addClip, state } from "../store.js";

const ARM = [
  { id: "raise", label: "Raise", min: -80, max: 110 },
  { id: "swing", label: "Forward / back", min: -70, max: 70 },
  { id: "bend", label: "Bend elbow", min: 0, max: 130 },
];
const LEG = [
  { id: "swing", label: "Forward / back", min: -70, max: 70 },
  { id: "out", label: "Out to the side", min: 0, max: 45 },
  { id: "bend", label: "Bend knee", min: 0, max: 120 },
];
const PARTS = [
  { id: "head", label: "Head", controls: [
    { id: "nod", label: "Nod", min: -40, max: 40 },
    { id: "turn", label: "Turn", min: -70, max: 70 },
    { id: "tilt", label: "Tilt", min: -35, max: 35 },
  ] },
  { id: "L_arm", label: "Left arm", controls: ARM },
  { id: "R_arm", label: "Right arm", controls: ARM },
  { id: "L_leg", label: "Left leg", controls: LEG },
  { id: "R_leg", label: "Right leg", controls: LEG },
];

function restValues() {
  return Object.fromEntries(PARTS.map((part) => [
    part.id, Object.fromEntries(part.controls.map((c) => [c.id, 0])),
  ]));
}

const _euler = new THREE.Euler();
const _quat = new THREE.Quaternion();

/** XYZ Euler degrees -> [x, y, z, w], matching rewards.quat_from_euler. */
function quat(x, y, z) {
  const d = THREE.MathUtils.degToRad;
  _quat.setFromEuler(_euler.set(d(x), d(y), d(z), "XYZ"));
  return [_quat.x, _quat.y, _quat.z, _quat.w];
}

/**
 * Slider values -> a contract pose. Left and right mirror each other: the
 * left arm lies along -X, so raising it is a negative Z rotation and swinging
 * it forward a positive Y one; the right arm is the opposite on both.
 */
function toPose(v) {
  // Every bone, so nothing a previous move bent stays bent.
  return {
    ...state.schema.rest_pose,
    head: quat(v.head.nod, v.head.turn, v.head.tilt),
    L_shoulder: quat(0, v.L_arm.swing, -v.L_arm.raise),
    R_shoulder: quat(0, -v.R_arm.swing, v.R_arm.raise),
    L_elbow: quat(0, 0, -v.L_arm.bend),
    R_elbow: quat(0, 0, v.R_arm.bend),
    L_hip: quat(-v.L_leg.swing, 0, -v.L_leg.out),
    R_hip: quat(-v.R_leg.swing, 0, v.R_leg.out),
    L_knee: quat(v.L_leg.bend, 0, 0),
    R_knee: quat(v.R_leg.bend, 0, 0),
  };
}

const copy = (values) => JSON.parse(JSON.stringify(values));

export const MoveBuilder = {
  emits: ["preview", "saved"],
  setup(_props, { emit }) {
    const values = reactive(restValues());
    const poses = ref([]);         // captured slider snapshots
    const name = ref("My move");
    const busy = ref(false);
    const error = ref(null);

    const livePose = () => emit("preview", { pose: toPose(values), clip: null });

    const keyframes = () => sequenceKeyframes(poses.value.map(toPose));

    function addPose() {
      poses.value.push(copy(values));
    }

    function loadPose(index) {
      Object.assign(values, copy(poses.value[index]));
      livePose();
    }

    function removePose(index) {
      poses.value.splice(index, 1);
    }

    function resetSliders() {
      Object.assign(values, restValues());
      livePose();
    }

    function play() {
      emit("preview", {
        pose: null,
        clip: { name: name.value, fps: 24, loop: true, keyframes: keyframes() },
      });
    }

    async function save() {
      if (!poses.value.length || busy.value) return;
      busy.value = true;
      error.value = null;
      try {
        const clip = await api.createClip({
          name: name.value.trim() || "My move", fps: 24, loop: true,
          keyframes: keyframes(),
        });
        addClip(clip);
        poses.value = [];
        resetSliders();
        emit("saved", clip);
      } catch (err) {
        error.value = err.message;
      } finally {
        busy.value = false;
      }
    }

    livePose();

    return {
      PARTS, values, poses, name, busy, error,
      canSave: computed(() => poses.value.length > 0 && name.value.trim()),
      livePose, addPose, loadPose, removePose, resetSliders, play, save,
    };
  },
  template: `
    <div class="move-builder">
      <fieldset v-for="part in PARTS" :key="part.id">
        <legend>{{ part.label }}</legend>
        <div v-for="c in part.controls" :key="c.id" class="slider-row">
          <label :for="part.id + '-' + c.id">{{ c.label }}</label>
          <input :id="part.id + '-' + c.id" type="range"
                 :min="c.min" :max="c.max" step="1"
                 v-model.number="values[part.id][c.id]" @input="livePose">
          <output>{{ values[part.id][c.id] }}°</output>
        </div>
      </fieldset>

      <div class="row">
        <button class="primary" @click="addPose">Add this pose</button>
        <button class="ghost" @click="resetSliders">Back to T-pose</button>
      </div>

      <template v-if="poses.length">
        <h3>Your poses</h3>
        <p class="muted">One pose holds still; two or more play in order, on
           a loop. Tap a pose to load it back into the sliders.</p>
        <div class="chips">
          <span v-for="(p, i) in poses" :key="i" class="chip pose-chip">
            <button class="ghost small" @click="loadPose(i)">Pose {{ i + 1 }}</button>
            <button class="ghost small" :aria-label="'Remove pose ' + (i + 1)"
                    @click="removePose(i)">×</button>
          </span>
        </div>
        <div class="prompt-row">
          <input v-model="name" maxlength="60" placeholder="Name your move">
          <button class="ghost" @click="play">Play</button>
          <button class="primary" :disabled="!canSave || busy" @click="save">
            {{ busy ? 'Saving…' : 'Save move' }}
          </button>
        </div>
      </template>
      <p v-if="error" class="error">{{ error }}</p>
    </div>
  `,
};
