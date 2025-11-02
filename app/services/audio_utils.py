import subprocess

def convert_to_wav_mono(input_path):
    """Convert any audio file to mono WAV (16 kHz)."""
    output_path = input_path.replace(".wav", "_mono.wav")
    command = [
        "ffmpeg", "-i", input_path,
        "-ac", "1", "-ar", "16000",
        output_path, "-y"
    ]
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return output_path
