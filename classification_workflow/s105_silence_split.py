from pydub import AudioSegment
import os

if not os.path.exists("recordings/all/sil"):
    os.makedirs("recordings/all/sil")

silence = AudioSegment.from_wav("silence.wav")
chunk_length_ms = 3000  # 3 seconds
for i, chunk in enumerate(silence[::chunk_length_ms]):
    chunk.export(f"recordings/all/sil/sil_{i:03}.wav", format="wav")
print(f"Exported {len(silence) // chunk_length_ms} silence chunks to 'recordings/all/sil/'")