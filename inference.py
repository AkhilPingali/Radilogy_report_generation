import torch
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from T5 import ModifiedT5  # Ensure T5.py is in the same directory or install as module

# === Setup ===
model_path = "/Users/nuthankishoremaddineni/Desktop/MTF/Models_Saved_two"
device = torch.device("cpu")

# === Load custom config ===
with open(f"{model_path}/custom_config.json", "r") as f:
    config = json.load(f)

# === Load tokenizer and model ===
tokenizer = T5Tokenizer.from_pretrained(model_path, legacy=False)
model = ModifiedT5.from_pretrained(
    model_path,
    image_feature_dim=config["image_feature_dim"],
    text_embedding_dim=config["text_embedding_dim"]
)
model.to(device)
model.eval()

# === Sample image feature ===
# Replace with a real image feature vector of size 1024
json_path = "/Users/nuthankishoremaddineni/Desktop/MTF/data/data_part2.json"
with open(json_path, "r") as f:
    data = json.load(f)
features = data[0]["features"]  # Should be a list of 1024 floats
image_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)  # [1, 1024]


# === Tokenize prompt ===
task_prefix = "Generate a Radiology report for the xray given image"
input_enc = tokenizer(task_prefix, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
input_ids = input_enc.input_ids.to(device)
attention_mask = input_enc.attention_mask.to(device)
#---------------------------
input_embeds = model.get_input_embeddings()(input_ids)

# Step 2: Fuse image features
image_embeddings = model.image_projection(image_tensor)  # [1, 768]
image_embeddings = image_embeddings.unsqueeze(1).expand(-1, input_embeds.size(1), -1)  # [1, 512, 768]
fused_embeds = model.final_projection(torch.cat([input_embeds, image_embeddings], dim=-1))  # [1, 512, 512]

fused_embeds = fused_embeds.to(device)
attention_mask = attention_mask.to(device)

print("attention_mask shape:", attention_mask.shape)
print("fused_embeds shape:", fused_embeds.shape)
print("Model's config:", model.config.hidden_size)  # Should match the last dimension of fused_embeds

# Step 3: Use original T5 generate method
output_ids = model.generate(
    inputs_embeds=fused_embeds,
    attention_mask=attention_mask,
    max_length=128,
    num_beams=4,
    early_stopping=True,
    use_cache=True,
    return_dict_in_generate=True,
    output_scores=True
)

report = tokenizer.decode(output_ids.sequences[0], skip_special_tokens=True)

# === Output ===
print("Generated Report:")
print(report)
