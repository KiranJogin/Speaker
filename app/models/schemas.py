from pydantic import BaseModel

class TranscriptionSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str

class TranscriptionResponse(BaseModel):
    status: str
    full_text: str
    transcription: list[TranscriptionSegment]
