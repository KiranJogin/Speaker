import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal
import io
import os
import tempfile
from pathlib import Path

from services.audio_processor import AudioProcessor
from services.speaker_separator import SpeakerSeparator
from services.transcribe_service import transcribe_streamlit


st.markdown("""
<style>
.line-item {
    padding: 10px;
    margin-bottom: 5px;
    border-radius: 6px;
    background-color: #222;
}
.line-item.active {
    background-color: #ff4d4d;
    color: white !important;
    border-left: 4px solid white;
}
</style>
""", unsafe_allow_html=True)

# ==============================
# Streamlit base config
# ==============================
st.set_page_config(
    page_title="SpeakerStudio – Separation + Diarization",
    page_icon="🎧",
    layout="wide",
)

st.title("🎧 SpeakerStudio")
st.markdown("**Unified Speaker Separation + Diarization & Transcription**")
st.markdown("---")

# ==============================
# Session state
# ==============================
for key, default in [
    ("processed_audio", None),
    ("separated_speakers", None),
    ("original_audio", None),
    ("sample_rate", None),
    ("dia_turns", None),
    ("dia_script", None),
    ("dia_session_name", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ==============================
# Helper: session folder for SPEAKER separation
# ==============================
def create_speaker_session_folder():
    base_dir = "saved_speaker"
    os.makedirs(base_dir, exist_ok=True)

    existing = [d for d in os.listdir(base_dir) if d.startswith("session_")]
    session_num = len(existing) + 1

    date_str = Path(tempfile.gettempdir()).stat().st_mtime  # not needed, use datetime
    import datetime as _dt
    date_str = _dt.datetime.now().strftime("%Y-%m-%d")

    session_folder = os.path.join(base_dir, f"session_{session_num:03d}_{date_str}")
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

# ==============================
# Sidebar
# ==============================
st.sidebar.header("⚙️ Separation Settings")
noise_reduction = st.sidebar.checkbox("Enable Noise Reduction", value=True)
normalize_audio = st.sidebar.checkbox("Normalize Audio", value=True)
num_speakers = st.sidebar.slider("Expected Number of Speakers", 2, 6, 2)

st.sidebar.markdown("---")
st.sidebar.markdown("### About")
st.sidebar.info(
    "This app runs **two pipelines** on your audio:\n\n"
    "1. **Speaker Separation** – isolates each speaker into its own track.\n"
    "2. **Diarization + Transcription** – detects who spoke when and what they said."
)

# ==============================
# Tabs
# ==============================
tab1, tab2, tab3, tab4 = st.tabs(
    ["📤 Upload & Process", "🎧 Playback Dashboard", "📝 Diarization & Transcription", "📊 Analysis"]
)

# ==============================
# TAB 1 – Upload & process
# ==============================
with tab1:
    st.header("Upload Audio File")
    st.markdown("Supported formats: WAV, MP3, M4A, FLAC, OGG")

    uploaded_file = st.file_uploader(
        "Choose an audio file",
        type=["wav", "mp3", "m4a", "flac", "ogg"],
        help="Upload a multi-speaker audio recording",
    )

    if uploaded_file is not None:
        st.success(f"✅ File uploaded: {uploaded_file.name}")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.audio(uploaded_file)

        with col2:
            st.metric("File Size", f"{len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")

        if st.button("🚀 Run Separation + Diarization", type="primary", use_container_width=True):
            with st.spinner("Processing audio..."):
                try:
                    progress_bar = st.progress(0)
                    status_text = st.empty()

                    # --------- Save temp file for librosa + Whisper ----------
                    status_text.text("Saving temporary audio file...")
                    progress_bar.progress(10)

                    raw_bytes = uploaded_file.getvalue()
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
                        tmp_file.write(raw_bytes)
                        tmp_path = tmp_file.name

                    # --------- Load audio for separation pipeline ----------
                    status_text.text("Loading & preprocessing for separation...")
                    progress_bar.progress(25)

                    audio, sr = librosa.load(tmp_path, sr=None, mono=False)
                    if audio.ndim > 1:
                        audio = np.mean(audio, axis=0)

                    st.session_state.original_audio = audio
                    st.session_state.sample_rate = sr

                    processor = AudioProcessor(sr)
                    processed = processor.preprocess(
                        audio,
                        reduce_noise=noise_reduction,
                        normalize=normalize_audio,
                    )
                    st.session_state.processed_audio = processed

                    # --------- Speaker separation ----------
                    status_text.text("Running speaker separation...")
                    progress_bar.progress(45)

                    separator = SpeakerSeparator(sr, num_speakers=num_speakers)
                    separated_speakers = separator.separate_speakers(processed)
                    st.session_state.separated_speakers = separated_speakers

                    # Save separated session to saved_speaker/
                    ses_folder = create_speaker_session_folder()

                    # Save original audio
                    orig_path = os.path.join(ses_folder, "original_audio.wav")
                    sf.write(orig_path, st.session_state.original_audio, st.session_state.sample_rate)

                    # Save separated speakers
                    for i, (speaker_label, speaker_audio) in enumerate(
                        st.session_state.separated_speakers.items(), 1
                    ):
                        file_path = os.path.join(ses_folder, f"speaker_{i}.wav")
                        sf.write(file_path, speaker_audio, st.session_state.sample_rate)

                    # --------- Diarization + transcription ----------
                    status_text.text("Running diarization + transcription...")
                    progress_bar.progress(75)

                    dia_result = transcribe_streamlit(raw_bytes)
                    st.session_state.dia_turns = dia_result.get("transcription", [])
                    st.session_state.dia_script = dia_result.get("formatted_script", "")
                    st.session_state.dia_session_name = dia_result.get("session", "")

                    # Cleanup temp
                    os.unlink(tmp_path)

                    status_text.text("Finalizing...")
                    progress_bar.progress(100)
                    st.success("✅ Both separation and diarization completed!")
                    st.balloons()

                except Exception as e:
                    st.error(f"❌ Error processing audio: {e}")
                    st.exception(e)

# ==============================
# TAB 2 – Playback dashboard (speaker separation)
# ==============================
with tab2:
    st.header("Separated Speaker Tracks")

    if st.session_state.separated_speakers is not None and st.session_state.sample_rate is not None:
        speakers = st.session_state.separated_speakers
        sr = st.session_state.sample_rate

        st.markdown(f"**{len(speakers)} speakers detected by separation pipeline**")
        st.markdown("---")

        for idx, (speaker_label, speaker_audio) in enumerate(speakers.items(), 1):
            with st.expander(f"🎤 Speaker {idx}", expanded=True):
                col1, col2 = st.columns([3, 1])

                with col1:
                    audio_bytes = io.BytesIO()
                    sf.write(audio_bytes, speaker_audio, sr, format="WAV")
                    audio_bytes.seek(0)
                    st.audio(audio_bytes, format="audio/wav")

                with col2:
                    duration = len(speaker_audio) / sr
                    st.metric("Duration", f"{duration:.1f}s")
                    st.metric("Samples", f"{len(speaker_audio):,}")

                    wav_bytes = io.BytesIO()
                    sf.write(wav_bytes, speaker_audio, sr, format="WAV")
                    wav_bytes.seek(0)
                    st.download_button(
                        label="⬇️ Download WAV",
                        data=wav_bytes,
                        file_name=f"speaker_{idx}.wav",
                        mime="audio/wav",
                        use_container_width=True,
                    )

                fig, ax = plt.subplots(figsize=(10, 2))
                time = np.linspace(0, duration, len(speaker_audio))
                ax.plot(time, speaker_audio, linewidth=0.5)
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Amplitude")
                ax.set_title(f"Speaker {idx} Waveform")
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()

        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col2:
            if st.button("🗑️ Clear Results", use_container_width=True):
                for key in ["separated_speakers", "processed_audio", "original_audio", "dia_turns", "dia_script"]:
                    st.session_state[key] = None
                st.rerun()
    else:
        st.info("👆 Upload and process an audio file to see separated speaker tracks here.")

# ==============================
# TAB 3 – Diarization + transcription
# ==============================

# ==============================
# TAB 3 – Diarization + transcription
# ==============================
with tab3:
    st.header("Diarization & Transcription")

    turns = st.session_state.dia_turns
    script_text = st.session_state.dia_script
    session_name = st.session_state.dia_session_name

    if turns and script_text:
        st.markdown(f"**Session:** `{session_name}`")
        st.markdown("Each line below is a continuous turn from one speaker.\n\n---")

        for idx, t in enumerate(turns, start=1):

            # speaker tag
            st.markdown(
                f"<h4>🗣️ <b>{t['speaker']}</b>  |  {t['start']:.2f}s → {t['end']:.2f}s</h4>",
                unsafe_allow_html=True,
            )

            # text
            st.write(t["text"])

            # audio playback
            audio_path = t.get("audio_path")
            if audio_path and os.path.exists(audio_path):
                with open(audio_path, "rb") as f:
                    audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/wav")
            else:
                st.warning("Audio segment not found on disk.")

            st.markdown("---")

        # full script below all segments
        st.subheader("Full Script")
        st.text_area("Full diarized script", script_text, height=250)

    else:
        st.info("👆 Process an audio file to see diarization & transcription here.")



# ==============================
# TAB 4 – Analysis (original audio + timelines)
# ==============================
with tab4:
    st.header("Audio Analysis & Visualization")

    if st.session_state.original_audio is not None and st.session_state.sample_rate is not None:
        audio = st.session_state.original_audio
        sr = st.session_state.sample_rate

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Audio Waveform")
            fig, ax = plt.subplots(figsize=(10, 4))
            time = np.linspace(0, len(audio) / sr, len(audio))
            ax.plot(time, audio, linewidth=0.5)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.set_title("Full Audio Waveform")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()

        with col2:
            st.subheader("Spectrogram")
            fig, ax = plt.subplots(figsize=(10, 4))
            D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
            img = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="hz", ax=ax, cmap="viridis")
            ax.set_title("Spectrogram")
            fig.colorbar(img, ax=ax, format="%+2.0f dB")
            st.pyplot(fig)
            plt.close()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Sample Rate", f"{sr} Hz")
        with col2:
            st.metric("Duration", f"{len(audio) / sr:.2f}s")
        with col3:
            st.metric("Total Samples", f"{len(audio):,}")

        if st.session_state.separated_speakers is not None:
            st.markdown("---")
            st.subheader("Speaker Activity Timeline (Separation)")

            fig, ax = plt.subplots(figsize=(12, 6))

            for idx, (speaker_label, speaker_audio) in enumerate(st.session_state.separated_speakers.items(), 1):
                energy = np.abs(speaker_audio)
                smoothed = signal.medfilt(energy, kernel_size=2001)
                time = np.linspace(0, len(speaker_audio) / sr, len(speaker_audio))
                ax.plot(time, smoothed + (idx - 1) * 0.3, label=f"Speaker {idx}", linewidth=1)

            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Speaker Activity")
            ax.set_title("Speaker Activity Timeline (Separated)")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close()
    else:
        st.info("👆 Upload and process an audio file to see detailed analysis here.")

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666;'>"
    "SpeakerStudio v1.0 | Unified Separation + Diarization System"
    "</div>",
    unsafe_allow_html=True,
)
