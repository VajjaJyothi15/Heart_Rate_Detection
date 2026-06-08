def smooth_bpm(raw_bpm: float) -> int:

    if raw_bpm < 50:
        return 50

    if raw_bpm > 100:
        return 100

    return int(raw_bpm)