// Phases 3 & 4 — the training screen. This is the lesson.
//
// Left: the target pose the model imagined. Right: the learner, updating live.
// Below: what earns points, and the learning curve that results.

import { computed, onBeforeUnmount, ref } from "vue";
import { api } from "../api.js";
import { characteristicPose, sequenceKeyframes } from "../pose.js";
import { isModelClip, refreshBehaviours, state } from "../store.js";
import { AvatarCanvas } from "./AvatarCanvas.js";
import { RewardChart } from "./RewardChart.js";

export const StepTrain = {
  components: { AvatarCanvas, RewardChart },
  setup() {
    const weights = ref({ ...state.schema.reward_weights });
    const episodesTotal = ref(300);
    const speed = ref(1);
    const history = ref([]);
    const latest = ref(null);
    const run = ref(null);
    const status = ref("idle"); // idle | running | paused | finished | error
    const error = ref(null);
    const saved = ref(false);
    const sequenceName = ref("My learned dance");
    const sequenceSaved = ref(false);
    const previewing = ref(false);
    const running = computed(() =>
      status.value === "running" || status.value === "paused");
    // The round picked in "Your rounds": what is shown, saved, downloaded,
    // and kept training from.
    const selectedRunId = ref(null);
    let closeStream = null;

    const attempts = computed(() =>
      state.attempts.filter((a) => a.clipId === state.activeClip?.id));
    const selected = computed(() =>
      attempts.value.find((a) => a.runId === selectedRunId.value) || null);
    const bestAttempt = computed(() =>
      attempts.value.reduce((best, a) =>
        (!best || a.last.match > best.last.match ? a : best), null));

    // Coming back to this step: pick up the latest round of this move.
    selectedRunId.value = attempts.value.at(-1)?.runId ?? null;

    // While training, the live episode; afterwards, the picked round's last.
    const shown = computed(() =>
      running.value || !selected.value ? latest.value : selected.value.last);

    /** How much a round improved on the one it carried on from, in points. */
    function gain(attempt) {
      const parent = state.attempts.find((a) => a.runId === attempt.continuedFrom);
      return parent
        ? Math.round((attempt.last.match - parent.last.match) * 100) : null;
    }

    function recordAttempt() {
      if (!run.value || !latest.value) return;
      if (state.attempts.some((a) => a.runId === run.value.id)) return;
      state.attempts.push({
        runId: run.value.id,
        clipId: run.value.target_clip_id,
        round: run.value.round,
        continuedFrom: run.value.continued_from,
        last: latest.value,
      });
      selectedRunId.value = run.value.id;
    }

    function selectAttempt(attempt) {
      if (running.value) return;
      selectedRunId.value = attempt.runId;
      previewing.value = false;
      saved.value = false;
    }

    function removeAttempt(attempt) {
      state.attempts = state.attempts.filter((a) => a.runId !== attempt.runId);
      if (selectedRunId.value === attempt.runId) {
        selectedRunId.value = attempts.value.at(-1)?.runId ?? null;
      }
    }

    const targetPose = computed(() =>
      state.activeClip
        ? characteristicPose(state.activeClip, state.schema.rest_pose)
        : state.schema.rest_pose);

    const learnerPose = computed(() => shown.value?.pose || state.schema.rest_pose);
    const jointHeat = computed(() => shown.value?.per_joint_error || null);
    const matchPercent = computed(() =>
      Math.round((shown.value?.match ?? 0) * 100));

    // Only a GLB rig has a model to export; a stick figure would just 409.
    const glbUrl = computed(() =>
      selected.value && state.avatar.rig.format === "glb"
        ? api.exportGlbUrl(selected.value.runId) : null);

    // With every slider at zero there is nothing to earn, so the learner gets
    // no signal at all. Worth saying out loud — otherwise it just looks broken.
    const noReward = computed(() =>
      Object.values(weights.value).every((w) => w <= 0));

    // keepGoing: carry on from the picked round instead of the T-pose.
    async function start(keepGoing = false) {
      const from = keepGoing ? selected.value?.runId : null;
      stopStream();
      history.value = [];
      latest.value = null;
      error.value = null;
      saved.value = false;
      status.value = "running";

      try {
        run.value = await api.startRun(
          state.avatar.id, state.activeClip.id,
          { episodes: episodesTotal.value, reward_weights: weights.value },
          speed.value, from);

        closeStream = api.streamRun(run.value.id, {
          onEpisode(episode) {
            latest.value = episode;
            // Thin the chart data on long runs — one point per pixel is plenty
            // and keeps the redraw cheap.
            const stride = Math.ceil(episodesTotal.value / 400);
            if (episode.episode % stride === 0 || episode.done) {
              history.value.push(episode);
            }
          },
          onEnd() {
            status.value = "finished";
            recordAttempt();
          },
          onError(err) {
            error.value = err.message;
            status.value = "error";
          },
        });
      } catch (err) {
        error.value = err.message;
        status.value = "error";
      }
    }

    async function stop() {
      if (!run.value) return;
      await api.runControl(run.value.id, "stop");
      status.value = "finished";
    }

    async function togglePause() {
      if (!run.value) return;
      const paused = status.value === "paused";
      await api.runControl(run.value.id, paused ? "resume" : "pause");
      status.value = paused ? "running" : "paused";
    }

    async function changeSpeed(value) {
      speed.value = Number(value);
      if (run.value && status.value !== "idle") {
        await api.runControl(run.value.id, "speed", { speed: speed.value });
      }
    }

    function stopStream() {
      closeStream?.();
      closeStream = null;
    }

    // -- pose sequence: catch poses from the learner, save them as a move --

    function catchPose() {
      if (!shown.value) return;
      state.caughtPoses.push({
        pose: shown.value.pose,
        reward: shown.value.reward,
        from: state.activeClip?.name || "",
      });
      sequenceSaved.value = false;
    }

    function moveCaught(index, step) {
      const list = state.caughtPoses;
      const to = index + step;
      if (to < 0 || to >= list.length) return;
      [list[index], list[to]] = [list[to], list[index]];
      sequenceSaved.value = false;
    }

    function removeCaught(index) {
      state.caughtPoses.splice(index, 1);
      if (!state.caughtPoses.length) previewing.value = false;
      sequenceSaved.value = false;
    }

    const sequenceClip = computed(() => ({
      name: sequenceName.value.trim() || "My learned dance",
      fps: 24,
      loop: true,
      keyframes: sequenceKeyframes(state.caughtPoses.map((c) => c.pose)),
    }));

    // Previewing borrows the learner's canvas, so only while nothing is
    // training on it.
    const previewClip = computed(() =>
      previewing.value && !running.value && state.caughtPoses.length
        ? sequenceClip.value : null);

    async function saveSequence() {
      if (!state.caughtPoses.length) return;
      await api.saveBehaviour({
        avatar_id: state.avatar.id,
        name: sequenceClip.value.name,
        trained: true,
        best_reward: Math.max(...state.caughtPoses.map((c) => c.reward)),
        clip: sequenceClip.value,
      });
      await refreshBehaviours();
      sequenceSaved.value = true;
    }

    async function saveLearned() {
      // Save what the avatar actually LEARNED — a single pose it reached — as
      // distinct from the clip the model imagined. That difference is the point
      // of the whole exercise.
      await api.saveBehaviour({
        avatar_id: state.avatar.id,
        name: `${state.activeClip.name} (learned)`,
        trained: true,
        best_reward: selected.value.last.best_reward,
        clip: {
          name: `${state.activeClip.name} (learned)`,
          fps: 24,
          keyframes: [{ t: 0, pose: selected.value.last.pose }],
        },
      });
      await refreshBehaviours();
      saved.value = true;
    }

    onBeforeUnmount(stopStream);

    // A built-in move can be selected on Teach to preview it, but it is
    // played, not trained.
    const trainable = computed(() =>
      Boolean(state.activeClip) && !isModelClip(state.activeClip));

    return {
      sequenceName, sequenceSaved, previewing, previewClip,
      catchPose, moveCaught, removeCaught, saveSequence,
      trainable, state, weights, episodesTotal, speed, history, latest, status, error, run,
      targetPose, learnerPose, jointHeat, matchPercent, noReward, saved, glbUrl,
      start, stop, togglePause, changeSpeed, saveLearned,
      attempts, selected, bestAttempt, shown, gain, selectAttempt, removeAttempt,
      labels: state.schema.reward_labels,
      running,
    };
  },
  template: `
    <section class="step">
      <header class="step-head">
        <h2>Train your avatar</h2>
        <p>Your avatar starts out knowing nothing. It tries small changes, and
           keeps the ones that earn points. You decide what earns points.</p>
      </header>

      <div v-if="!trainable" class="placeholder">
        <p v-if="state.activeClip">
          "{{ state.activeClip.name }}" came built into your model, so it
          already knows it — find it on the Play step. Make a new move on the
          previous step to train.
        </p>
        <p v-else>Pick a move on the previous step first.</p>
      </div>

      <template v-else>
        <div class="arena">
          <AvatarCanvas :rig="state.avatar.rig" :pose="targetPose"
                        label="Target" />
          <div class="arena-meter">
            <div class="match-ring" :style="{ '--pct': matchPercent }">
              <span>{{ matchPercent }}%</span>
              <small>match</small>
            </div>
            <dl>
              <div v-if="running ? run : selected"><dt>Round</dt><dd>{{ running ? run.round : selected?.round }}</dd></div>
              <div><dt>Try</dt><dd>{{ shown?.episode ?? 0 }} / {{ episodesTotal }}</dd></div>
              <div><dt>Score</dt><dd>{{ (shown?.reward ?? 0).toFixed(2) }}</dd></div>
              <div><dt>Best</dt><dd>{{ (shown?.best_reward ?? 0).toFixed(2) }}</dd></div>
              <div><dt>Trying new things</dt>
                   <dd>{{ Math.round((shown?.exploration ?? 0) * 100) }}%</dd></div>
            </dl>
            <p v-if="shown?.note" class="note">{{ shown.note }}</p>
          </div>
          <AvatarCanvas :rig="state.avatar.rig" :pose="learnerPose"
                        :clip="previewClip"
                        :jointHeat="previewClip ? null : jointHeat"
                        :label="previewClip ? 'Your sequence' : 'Learner'" />
        </div>
        <p class="muted legend">
          <i class="dot green"></i> joint is on target
          <i class="dot red"></i> joint is still wrong
        </p>

        <div v-if="attempts.length" class="panel rounds">
          <h3>Your rounds</h3>
          <p class="muted">Each bar is one round of training. "Keep training"
             starts the next round from the pose you picked, so practice adds
             up. Tap a bar to pick the round to keep.</p>
          <ol class="round-bars">
            <li v-for="a in attempts" :key="a.runId"
                :class="{ fresh: !a.continuedFrom, picked: a === selected }">
              <button class="round-bar" :disabled="running"
                      :aria-label="'Round ' + a.round + ', ' + Math.round(a.last.match * 100) + '% match'"
                      @click="selectAttempt(a)">
                <span class="round-gain" v-if="gain(a) !== null"
                      :class="{ down: gain(a) < 0 }">
                  {{ gain(a) >= 0 ? '+' : '' }}{{ gain(a) }}
                </span>
                <span class="round-fill"
                      :style="{ height: Math.max(4, Math.round(a.last.match * 100)) + '%' }">
                  {{ Math.round(a.last.match * 100) }}%
                </span>
              </button>
              <small>{{ a === bestAttempt ? '★ ' : '' }}Round {{ a.round }}</small>
              <button class="ghost small" :disabled="running"
                      :aria-label="'Remove round ' + a.round"
                      @click="removeAttempt(a)">×</button>
            </li>
          </ol>
          <p class="muted" v-if="bestAttempt && selected && selected !== bestAttempt">
            ★ Round {{ bestAttempt.round }} matched best
            ({{ Math.round(bestAttempt.last.match * 100) }}%).
            <button class="ghost small" @click="selectAttempt(bestAttempt)">Pick it</button>
          </p>
        </div>

        <div class="split">
          <div class="panel">
            <h3>What earns points?</h3>
            <p class="muted">Change these and train again — the avatar will
               learn something different.</p>
            <div v-for="(label, key) in labels" :key="key" class="slider-row">
              <label>{{ label }}</label>
              <input type="range" min="0" max="1" step="0.05"
                     v-model.number="weights[key]" :disabled="running">
              <output>{{ weights[key].toFixed(2) }}</output>
            </div>
            <p v-if="noReward" class="warn">
              Nothing earns points right now, so there's nothing to learn.
              Turn at least one slider up.
            </p>

            <div class="slider-row">
              <label>How many tries</label>
              <input type="range" min="50" max="1000" step="50"
                     v-model.number="episodesTotal" :disabled="running">
              <output>{{ episodesTotal }}</output>
            </div>
            <div class="slider-row">
              <label>Speed</label>
              <input type="range" min="0.25" max="5" step="0.25"
                     :value="speed" @input="changeSpeed($event.target.value)">
              <output>{{ speed }}×</output>
            </div>

            <div class="row">
              <button class="primary" v-if="!running && selected" @click="start(true)">
                Keep training round {{ selected.round }}
              </button>
              <button :class="selected ? 'ghost' : 'primary'" v-if="!running"
                      @click="start(false)">
                {{ selected ? 'Start over' : 'Start training' }}
              </button>
              <button class="ghost" v-if="running" @click="togglePause">
                {{ status === 'paused' ? 'Resume' : 'Pause' }}
              </button>
              <button class="ghost" v-if="running" @click="stop">Stop</button>
              <button class="ghost" v-if="shown" @click="catchPose">
                Catch this pose
              </button>
              <button class="primary" v-if="!running && selected"
                      @click="saveLearned" :disabled="saved">
                {{ saved ? 'Saved!' : 'Save round ' + selected.round }}
              </button>
              <a class="button ghost" v-if="!running && glbUrl"
                 :href="glbUrl" download>
                Download animated GLB
              </a>
            </div>
            <p v-if="error" class="error">{{ error }}</p>

            <template v-if="state.caughtPoses.length">
              <h3>Your pose sequence</h3>
              <p class="muted">Poses you caught, in the order they'll play.
                 Train again with different rewards and catch more to add on.</p>
              <ol class="caught-list">
                <li v-for="(c, i) in state.caughtPoses" :key="i">
                  <span>
                    <strong>Pose {{ i + 1 }}</strong>
                    <small>{{ c.from ? c.from + ' · ' : '' }}score {{ c.reward.toFixed(2) }}</small>
                  </span>
                  <span class="clip-actions">
                    <button class="ghost small" :disabled="i === 0"
                            :aria-label="'Move pose ' + (i + 1) + ' earlier'"
                            @click="moveCaught(i, -1)">↑</button>
                    <button class="ghost small"
                            :disabled="i === state.caughtPoses.length - 1"
                            :aria-label="'Move pose ' + (i + 1) + ' later'"
                            @click="moveCaught(i, 1)">↓</button>
                    <button class="ghost small"
                            :aria-label="'Remove pose ' + (i + 1)"
                            @click="removeCaught(i)">×</button>
                  </span>
                </li>
              </ol>
              <div class="prompt-row">
                <input v-model="sequenceName" maxlength="60"
                       placeholder="Name your sequence" @input="sequenceSaved = false">
                <button class="ghost" :disabled="running"
                        :title="running ? 'Stop training to preview' : ''"
                        @click="previewing = !previewing">
                  {{ previewing && !running ? 'Stop preview' : 'Preview' }}
                </button>
                <button class="primary" :disabled="sequenceSaved"
                        @click="saveSequence">
                  {{ sequenceSaved ? 'Saved!' : 'Save sequence' }}
                </button>
              </div>
            </template>
          </div>

          <div class="panel">
            <h3>The learning curve</h3>
            <RewardChart :episodes="history" :total="episodesTotal" />
            <p class="muted">
              The blue line jumps around because the avatar is guessing. The
              green line only ever goes up — that's its best score so far. If
              green flattens out, it has stopped improving.
            </p>
          </div>
        </div>
      </template>
    </section>
  `,
};
