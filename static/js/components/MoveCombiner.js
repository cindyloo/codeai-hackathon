// Combine moves by body part: the tail from one move, the fins from
// another, and so on, saved as one new move.
//
// Each part plays its own move at its own speed, looped; parts with no move
// stay still. The merge itself is combineClips() in pose.js. The parent shows
// the result as a clip, as it does for the move builder.

import { computed, reactive, ref, watch } from "vue";
import { api } from "../api.js";
import { clipMoves, clipSeconds, combineClips, COMBINE_FPS } from "../pose.js";
import { addClip, state } from "../store.js";

const PARTS = [
  { id: "body", label: "Body", bones: ["hips", "spine"] },
  { id: "head", label: "Head", bones: ["neck", "head"] },
  { id: "arms", label: "Arms, fins or wings", bones: [
    "L_shoulder", "L_elbow", "L_hand", "R_shoulder", "R_elbow", "R_hand"] },
  { id: "legs", label: "Legs", bones: [
    "L_hip", "L_knee", "L_foot", "R_hip", "R_knee", "R_foot"] },
  { id: "tail", label: "Tail", bones: ["tail", "tail_tip"] },
  { id: "stretch", label: "Stretch / squash", bones: [], stretch: true },
];

const SPEEDS = [
  { value: 0.5, label: "½×" },
  { value: 1, label: "1×" },
  { value: 2, label: "2×" },
  { value: 3, label: "3×" },
];

export const MoveCombiner = {
  emits: ["preview", "saved"],
  setup(_props, { emit }) {
    const picks = reactive(Object.fromEntries(
      PARTS.map((p) => [p.id, { clipId: "", speed: 1 }])));
    const name = ref("");
    const busy = ref(false);
    const error = ref(null);

    // Only the parts this model has (stretch needs no bone), and for each,
    // only the moves that actually move it.
    const parts = computed(() => {
      const has = new Set(state.avatar?.bones || []);
      return PARTS
        .filter((p) => p.stretch || p.bones.some((b) => has.has(b)))
        .map((p) => ({
          ...p,
          bones: p.bones.filter((b) => has.has(b)),
          options: state.clips.filter((clip) => clipMoves(clip, p)),
        }));
    });

    const clipById = (id) => state.clips.find((c) => c.id === id) || null;

    const layers = computed(() => parts.value.map((p) => ({
      bones: p.bones,
      stretch: Boolean(p.stretch),
      clip: clipById(picks[p.id].clipId),
      speed: picks[p.id].speed,
    })));

    const chosen = computed(() => layers.value.filter((l) => l.clip));
    const keyframes = computed(() =>
      combineClips(layers.value, state.schema.rest_pose));
    const seconds = computed(() => keyframes.value.at(-1)?.t || 0);

    const suggestedName = computed(() => {
      const names = [...new Set(chosen.value.map((l) => l.clip.name))];
      return names.join(" + ") || "My combined move";
    });

    const clip = computed(() => ({
      name: name.value.trim() || suggestedName.value,
      fps: COMBINE_FPS,
      loop: true,
      duration: seconds.value,
      keyframes: keyframes.value,
    }));

    function preview() {
      emit("preview", {
        pose: state.schema.rest_pose, stretch: 0,
        clip: keyframes.value.length ? clip.value : null,
      });
    }
    // Show the combination as it's put together.
    watch(keyframes, preview, { immediate: true });

    async function save() {
      if (!keyframes.value.length || busy.value) return;
      busy.value = true;
      error.value = null;
      try {
        const { duration: _d, ...body } = clip.value;
        const saved = await api.createClip(body);
        addClip(saved);
        for (const part of PARTS) picks[part.id].clipId = "";
        name.value = "";
        emit("saved", saved);
      } catch (err) {
        error.value = err.message;
      } finally {
        busy.value = false;
      }
    }

    return {
      parts, picks, name, busy, error, SPEEDS, chosen, seconds,
      suggestedName, clipSeconds, save,
    };
  },
  template: `
    <div class="move-builder move-combiner">
      <p class="muted">Pick a move for each part. Each part plays its own move
         at its own speed, over and over; parts you leave alone stay still.</p>

      <fieldset v-for="part in parts" :key="part.id">
        <legend>{{ part.label }}</legend>
        <div class="combine-row">
          <select v-model="picks[part.id].clipId" :aria-label="part.label + ' move'">
            <option value="">Stay still</option>
            <option v-for="c in part.options" :key="c.id" :value="c.id">
              {{ c.name }} ({{ clipSeconds(c).toFixed(1) }}s)
            </option>
          </select>
          <div class="render-segmented four" role="group" :aria-label="part.label + ' speed'">
            <button v-for="s in SPEEDS" :key="s.value"
                    :class="{ active: picks[part.id].speed === s.value }"
                    :disabled="!picks[part.id].clipId"
                    @click="picks[part.id].speed = s.value">{{ s.label }}</button>
          </div>
        </div>
        <p v-if="!part.options.length" class="muted small">
          None of your moves move this part yet.
        </p>
      </fieldset>

      <template v-if="chosen.length">
        <p class="muted">{{ seconds.toFixed(1) }}s, looping. The longest part
           sets the length; shorter ones repeat inside it.</p>
        <div class="prompt-row">
          <input v-model="name" maxlength="60" :placeholder="suggestedName">
          <button class="primary" :disabled="busy" @click="save">
            {{ busy ? 'Saving…' : 'Save move' }}
          </button>
        </div>
      </template>
      <p v-if="error" class="error">{{ error }}</p>
    </div>
  `,
};
