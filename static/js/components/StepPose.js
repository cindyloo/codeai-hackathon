// Phase 2 — describe a move in words, get a clip of poses back.

import { computed, ref } from "vue";
import { api } from "../api.js";
import { addClip, goTo, isModelClip, refreshBehaviours, state } from "../store.js";
import { AvatarCanvas } from "./AvatarCanvas.js";
import { MoveBuilder } from "./MoveBuilder.js";
import { MoveCombiner } from "./MoveCombiner.js";

const SUGGESTIONS = [
  "waves arms in the air",
  "waves hello",
  "jumps up high",
  "does a star jump",
  "walks along",
  "does a silly dance",
  "takes a bow",
];

export const StepPose = {
  components: { AvatarCanvas, MoveBuilder, MoveCombiner },
  setup() {
    const prompt = ref("waves arms in the air");
    const busy = ref(false);
    const progress = ref(0);
    const message = ref("");
    const error = ref(null);
    const saved = ref(null);
    // Building a move by hand ("build") or combining moves ("combine") takes
    // over the canvas: the sliders' live pose, the captured poses played, or
    // the combination as it's put together.
    const mode = ref("");
    const building = computed(() => Boolean(mode.value));
    const builderPreview = ref({ pose: null, clip: null });

    async function generate() {
      if (!prompt.value.trim() || busy.value) return;
      busy.value = true;
      error.value = null;
      try {
        const clip = await api.createPose(state.avatar.id, prompt.value,
          (fraction, msg) => { progress.value = fraction; message.value = msg; });
        addClip(clip);
      } catch (err) {
        error.value = err.message;
      } finally {
        busy.value = false;
      }
    }

    async function saveAsBehaviour(clip) {
      // Untrained: this is the move as the model imagined it, not something the
      // avatar has learned yet. The playground marks the difference.
      await api.saveBehaviour({
        avatar_id: state.avatar.id, clip_id: clip.id, name: clip.name,
      });
      await refreshBehaviours();
      saved.value = clip.id;
      setTimeout(() => { saved.value = null; }, 1800);
    }

    // Built-in moves are saved as behaviours when the avatar loads and are
    // never trained, so they get neither button.
    const fromModel = isModelClip;

    return {
      prompt, busy, progress, message, error, state, saved, fromModel,
      SUGGESTIONS, generate, saveAsBehaviour, goTo, building, builderPreview, mode,
      select: (clip) => { mode.value = ""; state.activeClip = clip; },
      open: (which) => {
        builderPreview.value = { pose: null, clip: null };
        mode.value = mode.value === which ? "" : which;
      },
      builderSaved: () => { mode.value = ""; },
      use: (text) => { prompt.value = text; generate(); },
    };
  },
  template: `
    <section class="step">
      <header class="step-head">
        <h2>Tell your avatar what to do</h2>
        <p>Describe a move in your own words. A language model turns your words
           into positions for each joint.</p>
      </header>

      <div class="split">
        <div class="panel">
          <div class="prompt-row">
            <input v-model="prompt" placeholder="e.g. waves arms in the air"
                   @keyup.enter="generate" :disabled="busy">
            <button class="primary" :disabled="busy || !prompt.trim()"
                    @click="generate">
              {{ busy ? 'Thinking…' : 'Make the move' }}
            </button>
          </div>

          <div class="chips">
            <button v-for="s in SUGGESTIONS" :key="s" class="chip"
                    :disabled="busy" @click="use(s)">{{ s }}</button>
          </div>

          <div v-if="busy" class="progress">
            <div class="progress-bar" :style="{ width: (progress*100) + '%' }"></div>
            <span>{{ message }}</span>
          </div>
          <p v-if="error" class="error">{{ error }}</p>

          <div class="row">
            <button class="ghost" :class="{ active: mode === 'build' }"
                    @click="open('build')">
              {{ mode === 'build' ? 'Close the builder' : 'Build a move by hand' }}
            </button>
            <button class="ghost" :class="{ active: mode === 'combine' }"
                    :disabled="state.clips.length < 1" @click="open('combine')">
              {{ mode === 'combine' ? 'Close the combiner' : 'Combine moves' }}
            </button>
          </div>
          <MoveBuilder v-if="mode === 'build'"
                       @preview="builderPreview = $event"
                       @saved="builderSaved" />
          <MoveCombiner v-if="mode === 'combine'"
                        @preview="builderPreview = $event"
                        @saved="builderSaved" />

          <h3 v-if="state.clips.length">Moves you've made</h3>
          <ul class="clip-list">
            <li v-for="clip in state.clips" :key="clip.id"
                :class="{ active: state.activeClip?.id === clip.id }"
                @click="select(clip)">
              <div>
                <strong>{{ clip.name }}</strong>
                <small>{{ fromModel(clip) ? 'from your model · ' : '' }}{{ clip.keyframes.length }} poses ·
                       {{ clip.duration.toFixed(1) }}s</small>
              </div>
              <div class="clip-actions">
                <button v-if="!fromModel(clip)" class="ghost small"
                        @click.stop="saveAsBehaviour(clip)">
                  {{ saved === clip.id ? 'Saved!' : 'Save' }}
                </button>
                <!-- A built-in move is already a finished animation; training
                     toward it would only give back a worse copy. -->
                <button v-if="!fromModel(clip)" class="primary small"
                        @click.stop="select(clip); goTo('train')">
                  Train it
                </button>
              </div>
            </li>
          </ul>
        </div>

        <div class="panel">
          <AvatarCanvas v-if="building" :rig="state.avatar.rig"
                        :pose="builderPreview.pose" :stretch="builderPreview.stretch || 0"
                        :clip="builderPreview.clip"
                        :label="mode === 'combine' ? (builderPreview.clip ? 'Combined' : 'Pick some moves')
                               : builderPreview.clip ? 'Your poses' : 'Sliders'" />
          <AvatarCanvas v-else :rig="state.avatar.rig" :clip="state.activeClip"
                        :label="state.activeClip ? state.activeClip.name : 'Waiting'" />
          <p class="muted" v-if="mode === 'combine'">
            Each part of the body plays the move you picked for it, all at
            once. Try a tail wag with a fin flap at 2× speed.
          </p>
          <p class="muted" v-else-if="building">
            Move the sliders to pose your avatar, then add the pose. Add a few
            to make it move.
          </p>
          <p class="muted" v-else-if="!state.activeClip">
            Make a move and it'll play here on a loop.
          </p>
          <p class="muted" v-else>
            This is what the model <em>imagined</em>. Your avatar hasn't learned
            it yet — that's the next step.
          </p>
        </div>
      </div>
    </section>
  `,
};
