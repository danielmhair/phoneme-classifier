from pydub import AudioSegment
import os

if not os.path.exists("recordings/all/sil"):
    os.makedirs("recordings/all/sil")
if not os.path.exists("recordings_lower_quality_2/dan/sil"):
    os.makedirs("recordings_lower_quality_2/dan/sil")
if not os.path.exists("recordings_lower_quality/dan/sil"):
    os.makedirs("recordings_lower_quality/dan/sil")
if not os.path.exists("recordings_lowest_quality_1/dan/sil"):
    os.makedirs("recordings_lowest_quality_1/dan/sil")
# silence = AudioSegment.from_wav("recordings/silence.wav")
# chunk_length_ms = 3000  # 3 seconds
# for i, chunk in enumerate(silence[::chunk_length_ms]):
#     chunk.export(f"recordings/all/sil/sil_{i:03}.wav", format="wav")
# print(f"Exported {len(silence) // chunk_length_ms} silence chunks to 'recordings/all/sil/'")

# copy the silence chunks to the lower quality recordings directory, but only 5 random silence files from recordings/all/sil
import random
import shutil
silence_files = os.listdir("recordings/all/sil")
random_silence_files = random.sample(silence_files, 5)
for silence_file in random_silence_files:
    shutil.copy(os.path.join("recordings/all/sil", silence_file), "recordings_lower_quality_2/dan/sil" + os.sep + silence_file)
    print(f"Copied {silence_file} to recordings_lower_quality_2/sil")
    shutil.copy(os.path.join("recordings/all/sil", silence_file), "recordings_lower_quality/dan/sil" + os.sep + silence_file)
    print(f"Copied {silence_file} to recordings_lower_quality/sil")
    shutil.copy(os.path.join("recordings/all/sil", silence_file), "recordings_lowest_quality_1/dan/sil" + os.sep + silence_file)
    print(f"Copied {silence_file} to recordings_lowest_quality_1/sil")
print("✅ Silence chunks copied to lower quality recordings directories.")