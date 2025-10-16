from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from tqdm import tqdm
print(torch.backends.mps.is_available())
print(torch.backends.mps.is_built())

# Load model & tokenizer
model = AutoModel.from_pretrained('Simonlee711/Clinical_ModernBERT')
tokenizer = AutoTokenizer.from_pretrained('Simonlee711/Clinical_ModernBERT')


# Move model to GPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

model.to(device)
model.eval()

# Load data
df = pd.read_csv('data/TCGA_Reports.csv')

embeddings = []
batch_size = 2  # try 16/32 if GPU memory allows

with torch.no_grad():
    for i in tqdm(range(0, len(df), batch_size)):
        batch_sentences = df['text'].iloc[i:i+batch_size].tolist()
        
        # Batch tokenization (pad to max length in batch)
        inputs = tokenizer(
            batch_sentences,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=8192
        ).to(device)
        
        # Forward pass on GPU
        outputs = model(**inputs)
        
        # CLS embeddings
        cls_embeddings = outputs.last_hidden_state[:, 0, :]

        # Move back to CPU and store
        embeddings.extend(cls_embeddings.cpu())

# Build dataframe
df_embeddings = pd.DataFrame({
    'patient': df['patient_filename'],
    'embedding': [emb.tolist() for emb in embeddings]
})

# Save as CSV (flatten vectors into strings)
df_embeddings['embedding'] = df_embeddings['embedding'].apply(lambda x: ','.join(map(str, x)))
df_embeddings.to_csv('out/embeddings.csv', index=False)