import subprocess
import os

def convert_to_wav_mono(input_path):
    """
    Converts any audio file to mono WAV using ffmpeg
    """
    output_path = input_path.replace(".wav", "_mono.wav")
    command = [
        "ffmpeg",
        "-i", input_path,
        "-ac", "1",
        "-ar", "16000",
        output_path,
        "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path
