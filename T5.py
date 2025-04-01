from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import json
from T5_Evaluation import evaluate_model



# Preprocess data function
def preprocess_data(input_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    cleaned_data = []
    for item in data:
        cleaned_item = {
            'features': item['features'],
            'caption': item['caption'],
            'indication': item['indication'],
            'findings': item['findings'],
            'impression': item['impression']
        }
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# Dataset class to load data and prepare inputs for the model
class ChestXrayDataset(Dataset):
    def __init__(self, data, tokenizer, max_source_length=512,max_target_length=128):
        self.features = []
        self.input_ids = []
        self.attention_masks = []
        self.labels = []
        task_prefix = "Generate a Radiology report for the xray given image"
        # encode the targets
        for item in data:
            image_features = torch.tensor(
                [f for f in item['features'] if f is not None],
                dtype=torch.float32
            )
            target_text = (
                f"Caption: {item['caption']} "
                f"Indication: {item['indication']} "
                f"Findings: {item['findings']} "
                f"Impression: {item['impression']}"
            ).strip()

            input_encodings = tokenizer(
                task_prefix,
                max_length=max_source_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            target_encodings = tokenizer(
                target_text,
                max_length=max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            self.features.append(image_features)
            self.input_ids.append(input_encodings.input_ids.squeeze())
            self.attention_masks.append(input_encodings.attention_mask.squeeze())
            self.labels.append(target_encodings.input_ids.squeeze())

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            "labels": self.labels[idx]
        }

class ModifiedT5(T5ForConditionalGeneration):
    def __init__(self, config, image_feature_dim, text_embedding_dim):
        super().__init__(config)
        self.image_projection = nn.Linear(image_feature_dim, text_embedding_dim)
        self.final_projection = nn.Linear(text_embedding_dim + config.d_model, config.d_model)  # To match T5's expected embedding size
        nn.init.xavier_uniform_(self.image_projection.weight)
        nn.init.xavier_uniform_(self.final_projection.weight)
        nn.init.zeros_(self.image_projection.bias)
        nn.init.zeros_(self.final_projection.bias)
    def forward(self, input_ids=None, attention_mask=None, labels=None, image_features=None,inputs_embeds=None,**kwargs):
        # Get the input token embeddings from the model
        #input_embeds = self.get_input_embeddings()(input_ids)  # Use get_input_embeddings() for token embeddings
        if inputs_embeds is None: #inputs_embeds as these are the inputs to T5 encoder and we are changing that so we can think we are adding custome layers at the start of the network
            if input_ids is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided.")
            input_embeds = self.get_input_embeddings()(input_ids)
        else:
            input_embeds = inputs_embeds
            
        if image_features is not None:
            image_embeddings = self.image_projection(image_features)
            image_embeddings = image_embeddings.unsqueeze(1)  # [batch_size, 1, image_embedding_dim]
            image_embeddings = image_embeddings.expand(-1, input_embeds.size(1), -1)  # Expand to [batch_size, seq_length, image_embedding_dim]
            input_embeds = torch.cat([input_embeds, image_embeddings], dim=-1)  # Concatenate along embedding dimension
            input_embeds = self.final_projection(input_embeds)  # Project back to the size expected by T5
        
        outputs = super().forward(inputs_embeds=input_embeds, attention_mask=attention_mask, labels=labels,*kwargs)
        return outputs

# Fine-tuning function
def fine_tune_t5(dataloader, tokenizer, device, epochs=3):
    model = ModifiedT5.from_pretrained('t5-small', image_feature_dim=1024, text_embedding_dim=768)
    model.to(device)  # Move model to the CPU
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
        for batch in dataloader:
            optimizer.zero_grad()

            image_features = batch['features'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, 
                            image_features=image_features, labels=labels)

            loss = outputs.loss
            loss.backward()
            optimizer.step()
            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            
    save_path = '/Users/nuthankishoremaddineni/Desktop/MTF/Models_Saved_two'
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    # Save custom configuration
    config_dict = {
        'image_feature_dim': 1024,
        'text_embedding_dim': 768
    }
    with open(f'{save_path}/custom_config.json', 'w') as f:
        json.dump(config_dict, f)
    
    return model

# Tokenizer and DataLoader preparation
def prepare_fine_tuning(input_file,test_size=0.2):
    cleaned_data = preprocess_data(input_file)
    train_data, val_data = train_test_split(cleaned_data, test_size=test_size, random_state=42)
    tokenizer = T5Tokenizer.from_pretrained('t5-small', legacy=False)

    train_dataset = ChestXrayDataset(train_data, tokenizer)
    val_dataset = ChestXrayDataset(val_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    return train_dataloader, val_dataloader, tokenizer

# Main function to run the training and evaluation
def main(input_file, device='cpu'):  # Ensure 'cpu' is passed as default
    # Set device to CPU
    device = torch.device('cpu')  # Always run on CPU
    train_dataloader, val_dataloader, tokenizer = prepare_fine_tuning(input_file)
    
    # Fine-tune the model
    model=fine_tune_t5(train_dataloader, tokenizer, device)

    # Evaluate the model
    generated_reports, true_reports, avg_metrics = evaluate_model(model, val_dataloader, tokenizer, device)
    print("\nEvaluation Metrics:")
    for metric, score in avg_metrics.items():
        print(f"{metric}: {score:.4f}")
    # Print some generated reports for inspection
    for report in generated_reports[:5]:
        print(report)
    for report in true_reports[:5]:
        print(report)

if __name__ == '__main__':
    input_file = '/Users/nuthankishoremaddineni/Desktop/MTF/data/data_part1.json'  # Adjust this path to your JSON file
    main(input_file)