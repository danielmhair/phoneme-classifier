from fastapi import FastAPI, UploadFile, File
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer
import torchaudio
import torch
import shutil
import soundfile as sf
app = FastAPI()

processor = None
model = None

def load_model():
    global processor, model
    if processor is None or model is None:
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-xlsr-53-espeak-cv-ft")

def predict_phonemes(file_path):
    load_model()
    

    speech_array, sampling_rate = sf.read(file_path)
    speech_array = torch.from_numpy(speech_array).float()
    resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
    speech = resampler(speech_array).squeeze()

    inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)

    with torch.no_grad():
        logits = model(**inputs).logits

    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)
    return transcription[0]

@app.post("/predict-phonemes")
async def predict_phoneme_endpoint(audio: UploadFile = File(...)):
    temp_audio_path = "temp_audio.wav"
    with open(temp_audio_path, "wb") as buffer:
        shutil.copyfileobj(audio.file, buffer)

    phoneme_result = predict_phonemes(temp_audio_path)
    return {"phonemes": phoneme_result}
