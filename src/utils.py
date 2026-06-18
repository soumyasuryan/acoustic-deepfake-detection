import numpy as np
import librosa
import torch

def preprocess_audio_gpu(audio_array, sampling_rate, target_duration=5, n_mels=128):
    target_length = target_duration * sampling_rate
    if len(audio_array) > target_length:
        audio_array = audio_array[:target_length]
    else:
        padding = target_length - len(audio_array)
        audio_array = np.pad(audio_array, (0, padding), mode='constant')
        
    mel_spec = librosa.feature.melspectrogram(
        y=audio_array, 
        sr=sampling_rate, 
        n_mels=n_mels, 
        n_fft=2048, 
        hop_length=512
    )
    return librosa.power_to_db(mel_spec, ref=np.max)

def get_text_embedding(text_string, tokenizer, model, device):
    inputs = tokenizer(text_string, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[0, 0, :]
