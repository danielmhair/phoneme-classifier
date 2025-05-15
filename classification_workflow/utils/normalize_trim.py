def preprocess_audio(audio: np.ndarray, threshold=0.01, buffer=1000):
    nonzero_indices = np.where(np.abs(audio) > threshold)[0]
    if len(nonzero_indices) == 0:
        return None  # or return np.zeros(16000), depending on use case
    start = max(0, nonzero_indices[0] - buffer)
    end = min(len(audio), nonzero_indices[-1] + buffer)
    trimmed = audio[start:end]
    max_amp = np.max(np.abs(trimmed))
    return trimmed / max_amp if max_amp > 0 else trimmed
