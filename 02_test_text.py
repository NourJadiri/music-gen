import torch
from transformers import AutoTokenizer, T5EncoderModel

print("Loading T5 model...")
model_id = "t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
text_encoder = T5EncoderModel.from_pretrained(model_id).to("cuda")

prompt = "A bright C major synth chord"

text_inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
print("STEP A: TEXT TOKEN IDS")
print("Word token ids:", text_inputs.input_ids)
print("Words broken down:", tokenizer.convert_ids_to_tokens(text_inputs.input_ids[0]))


with torch.no_grad():
    text_outputs = text_encoder(**text_inputs)
    text_embeddings = text_outputs.last_hidden_state

print("STEP B: TEXT EMBEDDINGS WORD BY WORD")
print("Text embeddings shape:", text_embeddings.shape, " (batch_size, seq_len, hidden_dim)")