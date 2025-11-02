from app.services.diarization_service import diarize_audio
from app.services.audio_utils import convert_to_wav_mono
from app.services.align_service import assign_speaker_to_words, group_words_to_turns
import whisper, torch, tempfile, os
from dotenv import load_dotenv

load_dotenv()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("base", device=device)

async def transcribe_audio(file):
    """
    Transcribes and diarizes an uploaded audio file.
    Returns both structured and formatted output.
    """
    # Save temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    # Convert to mono WAV for consistency
    wav_path = convert_to_wav_mono(tmp_path)

    # Whisper transcription with word timestamps
    result = model.transcribe(wav_path, word_timestamps=True)

    # Extract word-level timestamps
    words = []
    for segment in result["segments"]:
        for w in segment.get("words", []):
            words.append({
                "word": w["word"],
                "start": w["start"],
                "end": w["end"]
            })

    # Run diarization
    speaker_segments = diarize_audio(wav_path)

    # Align words with speakers
    aligned = assign_speaker_to_words(words, speaker_segments)
    turns = group_words_to_turns(aligned)

    # --- New: Generate readable script ---
    drama_script_lines = []
    for turn in turns:
        speaker = turn["speaker"]
        text = turn["text"].strip()
        # Capitalize first letter of sentence if needed
        if text and text[0].islower():
            text = text[0].upper() + text[1:]
        drama_script_lines.append(f"{speaker}: {text}")

    drama_script = "\n\n".join(drama_script_lines)

    # Cleanup temp files
    os.remove(tmp_path)
    if os.path.exists(wav_path):
        os.remove(wav_path)

    # Return both machine-readable and formatted text
    return {
        "status": "success",
        "transcription": turns,
        "formatted_script": drama_script
    }
