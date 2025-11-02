def assign_speaker_to_words(words, speaker_segments, margin=0.3):
    """
    Assign each transcribed word to the nearest diarized speaker segment.
    Handles small gaps and timing offsets.
    """
    result = []
    for word in words:
        w_start = word.get("start", 0)
        w_end = word.get("end", 0)
        w_text = word.get("word", "")

        speaker = "Unknown"
        for seg in speaker_segments:
            if (seg["start"] - margin <= w_start <= seg["end"] + margin) or \
               (seg["start"] - margin <= w_end <= seg["end"] + margin):
                speaker = seg["speaker"]
                break

        # Fallback: choose nearest segment by start time
        if speaker == "Unknown" and speaker_segments:
            closest = min(speaker_segments, key=lambda s: abs(s["start"] - w_start))
            speaker = closest["speaker"]

        result.append({
            "speaker": speaker,
            "start": w_start,
            "end": w_end,
            "text": w_text
        })
    return result


def group_words_to_turns(word_segments, merge_gap=0.5):
    """
    Merge consecutive words spoken by the same speaker into one sentence/turn.
    """
    turns = []
    if not word_segments:
        return turns

    current_speaker = word_segments[0]["speaker"]
    current_text = [word_segments[0]["text"]]
    start_time = word_segments[0]["start"]
    end_time = word_segments[0]["end"]

    for w in word_segments[1:]:
        if (
            w["speaker"] == current_speaker
            and w["start"] - end_time <= merge_gap
        ):
            current_text.append(w["text"])
            end_time = w["end"]
        else:
            turns.append({
                "speaker": current_speaker,
                "start": start_time,
                "end": end_time,
                "text": " ".join(current_text)
            })
            current_speaker = w["speaker"]
            current_text = [w["text"]]
            start_time = w["start"]
            end_time = w["end"]

    turns.append({
        "speaker": current_speaker,
        "start": start_time,
        "end": end_time,
        "text": " ".join(current_text)
    })
    return turns
