import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# --- Configuration ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
CSV_FILE = "/root/TCGA-Classification/jina-fine-tune/data.csv"
OUTPUT_DIR = "jina-v3-finetuned-classification"
TASK = "classification"
NUM_EPOCHS = 3          # Recommended starting point for LoRA fine-tuning
BATCH_SIZE = 2         # Adjust based on your GPU memory (8GB VRAM is typically sufficient for this model)
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# --- Step 1: Load, Preprocess, and Split Data ---
print(f"Loading data from {CSV_FILE}...")
df = pd.read_csv(CSV_FILE)

# Ensure required columns are present
print(df.head())
if 'text' not in df.columns or 'OS' not in df.columns:
    raise ValueError("CSV must contain 'text' and 'OS' columns.")

# Convert 'OS' (Operating System/Category) strings to numerical labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['OS'])

# Split the data into training and evaluation sets
train_df, eval_df = train_test_split(df, test_size=0.1, stratify=df['label'], random_state=42)

# Convert split DataFrames into InputExample objects
train_examples = [
    InputExample(texts=[row['text']], label=float(row['label']))
    for _, row in train_df.iterrows()
]

# The SentenceLabelDataset wraps the examples and automatically creates positive pairs
# (texts with the same label) for the CoSENTLoss.
train_dataset = SentenceLabelDataset(train_examples)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=4)

# --- Step 2: Load Model and Define Training Objective ---

# Load the model, specifying the 'classification' task for LoRA tuning
print(f"Loading model: {MODEL_NAME} with default_task='{TASK}'...")
model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    model_kwargs={
        'default_task': TASK,
        # Set this to True to fine-tune the ENTIRE model, not just LoRA adapters (requires much more VRAM)
        # 'lora_main_params_trainable': False
    }
)

# Use CoSENTLoss, which is highly effective for contrastive learning and semantic similarity
train_loss = losses.CoSENTLoss(model=model)

# Calculate warmup steps
total_steps = len(train_dataloader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)

# --- Step 3: Fine-Tune the Model ---

print("Starting fine-tuning...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Using model.fit() for simplicity with the custom DataLoader
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=NUM_EPOCHS,
    warmup_steps=warmup_steps,
    scheduler='warmupcosine',
    optimizer_params={'lr': LEARNING_RATE},
    output_path=OUTPUT_DIR,
    show_progress_bar=True,
    save_best_model=True,
)

print(f"\n✅ Fine-tuning complete. Model saved to {OUTPUT_DIR}")

# --- Optional: Verification/Inference ---
print("\nExample Inference (after fine-tuning):")
test_texts = eval_df['text'].sample(2).tolist()
embeddings = model.encode(test_texts, task=TASK, convert_to_tensor=False)
print(f"Encoded {len(embeddings)} texts to shape: {embeddings[0].shape}")