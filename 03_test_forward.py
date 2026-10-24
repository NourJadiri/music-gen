import torch
import numpy as np
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from utils.synth import make_chord_audio

print("Loading MusicGen model...")
model_id = "facebook/musicgen-small"
processor = AutoProcessor.from_pretrained(model_id)
model = MusicgenForConditionalGeneration.from_pretrained(model_id).to("cuda")

model.config.decoder.decoder_start_token_id = 2048
model.config.decoder_start_token_id = 2048

sr = model.audio_encoder.config.sampling_rate
raw_audio = make_chord_audio(sr).astype(np.float32)
prompt = "A bright C major synth chord"


inputs = processor(
    text=[prompt], audio=[raw_audio], sampling_rate=sr, return_tensors="pt"
).to("cuda")


with torch.no_grad():
    encoded = model.audio_encoder.encode(inputs.input_values, inputs.padding_mask)
    audio_labels = encoded.audio_codes[0].transpose(1,2)

print("Target audio labels shape:", audio_labels.shape, "-> (batch, 50_steps, 4_dicts)")

with torch.no_grad():
    outputs = model(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, labels=audio_labels
    )
    
print("\n--- TRANSFORMER OUTPUT ---")
print("Predicted Logits shape:", outputs.logits.shape, "-> (4_dicts, 50_steps, 2048_vocab)")
print("Initial Loss:", outputs.loss.item())   