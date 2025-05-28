import sounddevice as sd
import numpy as np
from scipy.io.wavfile import write

DURATION = 300  # seconds
SAMPLE_RATE = 16000

print("Recording 5 minutes of silence... stay quiet!")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()

write("silence.wav", SAMPLE_RATE, audio)
print("Saved silence.wav")
