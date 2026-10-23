import torch
import numpy as np
from scipy.io import wavfile
from transformers import EncodecModel, AutoProcessor
from synth import make_chord_audio

print("Loading Encodec model...")
model_id = "facebook/encodec_32khz"
processor = AutoProcessor.from_pretrained(model_id)
codec = EncodecModel.from_pretrained(model_id)


sr = processor.sampling_rate
raw_audio = make_chord_audio(sr)
raw_audio = raw_audio.astype(np.float32)

print("Before codec")
print("Raw audio shape:", raw_audio.shape)

inputs = processor(raw_audio, sampling_rate=sr, return_tensors="pt").to("cuda")

with torch.no_grad():
    encoded = codec.encode(inputs.input_values, inputs.padding_mask)
    tokens = encoded.audio_codes

print("After codec")
print("Tokens tensor shape:", tokens.shape)
print("First 5 time steps across all 4 dictionnaries:", tokens[0, :5, :])

print("After decoding")
print("Decoded audio shape:", decoded_audio.shape)

with torch.no_grad():
    decoded = codec.decode(tokens, encoded.audio_scales).audio_values
    decoded_np = decoded_audio[0,0].cpu().numpy()
    
wavfile.write("original.wav", sr, raw_audio)
wavfile.write("decoded.wav", sr, decoded_np)

print("Done! Check original.wav and decoded.wav")    
    