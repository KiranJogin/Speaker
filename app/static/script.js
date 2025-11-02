const form = document.getElementById("upload-form");
const fileInput = document.getElementById("file-input");
const loading = document.getElementById("loading");
const output = document.getElementById("output");
const scriptOutput = document.getElementById("script-output");

const globalPlayer = document.getElementById("global-player");
const globalPlay = document.getElementById("global-play");
const globalPause = document.getElementById("global-pause");
const globalProgress = document.getElementById("global-progress");

let fullAudioList = [];
let currentGlobalIndex = 0;
let globalAudio = null;

// ✅ Always hidden at load
globalPlayer.classList.add("hidden");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const file = fileInput.files[0];
  if (!file) return alert("Please select an audio file.");

  // Reset UI
  loading.classList.remove("hidden");
  output.classList.add("hidden");
  globalPlayer.classList.add("hidden");
  globalPlayer.classList.remove("show");
  scriptOutput.innerHTML = "";

  const formData = new FormData();
  formData.append("file", file);

  try {
    const res = await fetch("/transcribe", { method: "POST", body: formData });
    const data = await res.json();

    if (data.error) {
      alert("Error: " + data.error);
      return;
    }

    fullAudioList = data.transcription || [];
    renderTranscription(fullAudioList);

    if (fullAudioList.length > 0) {
      output.classList.remove("hidden");
      // show global player smoothly
      globalPlayer.classList.remove("hidden");
      setTimeout(() => globalPlayer.classList.add("show"), 100);
    } else {
      output.classList.add("hidden");
      globalPlayer.classList.add("hidden");
      globalPlayer.classList.remove("show");
      alert("No transcription data found. Please check the audio file.");
    }
  } catch (err) {
    alert("Failed: " + err.message);
  } finally {
    loading.classList.add("hidden");
  }
});

function renderTranscription(turns) {
  scriptOutput.innerHTML = "";
  turns.forEach((turn, idx) => {
    const div = document.createElement("div");
    div.classList.add("turn");
    div.innerHTML = `
      <div class="turn-header">
        <strong>${turn.speaker}:</strong>
        <div>
          <button id="play-${idx}">▶</button>
          <button id="pause-${idx}" class="hidden">⏸</button>
        </div>
      </div>
      <div>${turn.text}</div>
      <input type="range" id="progress-${idx}" class="progress" value="0" min="0" max="100" step="0.5">
    `;
    scriptOutput.appendChild(div);

    const audio = new Audio(turn.audio_path);
    turn.audio = audio;

    const playBtn = div.querySelector(`#play-${idx}`);
    const pauseBtn = div.querySelector(`#pause-${idx}`);
    const progress = div.querySelector(`#progress-${idx}`);

    playBtn.addEventListener("click", () => {
      audio.play();
      playBtn.classList.add("hidden");
      pauseBtn.classList.remove("hidden");
    });

    pauseBtn.addEventListener("click", () => {
      audio.pause();
      playBtn.classList.remove("hidden");
      pauseBtn.classList.add("hidden");
    });

    audio.addEventListener("timeupdate", () => {
      progress.value = (audio.currentTime / audio.duration) * 100;
    });

    audio.addEventListener("ended", () => {
      playBtn.classList.remove("hidden");
      pauseBtn.classList.add("hidden");
      progress.value = 100;
    });
  });
}

// ---------- GLOBAL PLAYER ----------
globalPlay.addEventListener("click", () => {
  if (!fullAudioList.length) return;
  playGlobalAudio(currentGlobalIndex);
});

globalPause.addEventListener("click", () => {
  if (globalAudio) globalAudio.pause();
  globalPlay.classList.remove("hidden");
  globalPause.classList.add("hidden");
});

function playGlobalAudio(index) {
  if (index >= fullAudioList.length) {
    currentGlobalIndex = 0;
    globalPlay.classList.remove("hidden");
    globalPause.classList.add("hidden");
    globalProgress.value = 100;
    return;
  }

  currentGlobalIndex = index;
  const turn = fullAudioList[index];

  if (globalAudio) {
    globalAudio.pause();
  }

  globalAudio = new Audio(turn.audio_path);
  globalAudio.play();

  globalPlay.classList.add("hidden");
  globalPause.classList.remove("hidden");

  highlightActiveTurn(index);

  globalAudio.addEventListener("timeupdate", updateGlobalProgress);
  globalAudio.addEventListener("ended", () => {
    currentGlobalIndex++;
    playGlobalAudio(currentGlobalIndex);
  });
}

function highlightActiveTurn(index) {
  document.querySelectorAll(".turn").forEach((el, i) => {
    el.classList.toggle("active-turn", i === index);
  });
  document.querySelectorAll(".turn")[index].scrollIntoView({
    behavior: "smooth",
    block: "center",
  });
}

function updateGlobalProgress() {
  let totalDuration = fullAudioList.reduce((acc, t) => acc + (t.audio?.duration || 0), 0);
  let elapsed = 0;
  for (let i = 0; i < currentGlobalIndex; i++) {
    elapsed += fullAudioList[i].audio?.duration || 0;
  }
  elapsed += globalAudio.currentTime;

  let progressPercent = (elapsed / totalDuration) * 100;
  globalProgress.value = progressPercent;
}
