from pydub import AudioSegment

silence = AudioSegment.from_wav("silence.wav")
chunk_length_ms = 3000  # 3 seconds
for i, chunk in enumerate(silence[::chunk_length_ms]):
    chunk.export(f"sil_chunks/sil_{i:03}.wav", format="wav")
print(f"Exported {len(silence) // chunk_length_ms} silence chunks to 'sil_chunks/'")