import os
import torch
from pyannote.audio import Pipeline
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get Hugging Face access token
HUGGINGFACE_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN")

# ✅ Define device properly as a torch.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ Initialize diarization pipeline safely
try:
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",  # Use the stable 3.1 model
        use_auth_token=HUGGINGFACE_TOKEN
    )

    # Move model to proper device
    pipeline.to(device)

except Exception as e:
    raise RuntimeError(
        f"Failed to load diarization pipeline. Please verify your Hugging Face token and model access.\nError: {e}"
    )

def diarize_audio(file_path: str):
    """
    Perform speaker diarization on the given audio file.

    Args:
        file_path (str): Path to the input audio file.

    Returns:
        list: List of segments with start, end, and speaker labels.
    """
    diarization = pipeline(file_path)

    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    return segments
