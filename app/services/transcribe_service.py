from app.services.diarization_service import diarize_audio
from app.services.audio_utils import convert_to_wav_mono
from app.services.align_service import assign_speaker_to_words, group_words_to_turns
import whisper, torch, tempfile, os, datetime, subprocess
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)

SAVE_ROOT = Path("saved_sessions")
SAVE_ROOT.mkdir(exist_ok=True)

def extract_audio_segment(input_path, start, end, output_path):
    """Extract a specific audio segment (in seconds)."""
    command = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ss", str(start),
        "-to", str(end),
        "-c", "copy", output_path,
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

async def transcribe_audio(file):
    """Transcribe, diarize, and save each speaker's lines and audio."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    wav_path = convert_to_wav_mono(tmp_path)
    result = model.transcribe(wav_path, word_timestamps=True)

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            words.append({"word": w["word"], "start": w["start"], "end": w["end"]})

    speaker_segments = diarize_audio(wav_path)
    aligned = assign_speaker_to_words(words, speaker_segments)
    turns = group_words_to_turns(aligned)

    session_name = datetime.datetime.now().strftime("session_%Y-%m-%d_%H-%M-%S")
    session_dir = SAVE_ROOT / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    full_text_path = session_dir / "full_transcript.txt"
    with open(full_text_path, "w") as f:
        for t in turns:
            f.write(f"{t['speaker']}: {t['text']}\n\n")

    for idx, t in enumerate(turns, start=1):
        sp_dir = session_dir / t["speaker"]
        sp_dir.mkdir(exist_ok=True)

        txt_path = sp_dir / f"line_{idx:02d}.txt"
        with open(txt_path, "w") as f:
            f.write(t["text"])

        audio_path = sp_dir / f"line_{idx:02d}.wav"
        extract_audio_segment(wav_path, t["start"], t["end"], str(audio_path))
        t["audio_path"] = f"/sessions/{session_name}/{t['speaker']}/line_{idx:02d}.wav"

    script_lines = [f"{t['speaker']}: {t['text']}" for t in turns]
    drama_script = "\n\n".join(script_lines)

    os.remove(tmp_path)
    if os.path.exists(wav_path):
        os.remove(wav_path)

    return {
        "status": "success",
        "session": session_name,
        "transcription": turns,
        "formatted_script": drama_script,
    }
