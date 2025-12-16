import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
# Import the necessary training argument class
from sentence_transformers import SentenceTransformerTrainingArguments 

# --- Configuration ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
CSV_FILE = "/root/TCGA-Classification/jina-fine-tune/data.csv"
OUTPUT_DIR = "jina-v3-finetuned-classification"
TASK = "classification"
NUM_EPOCHS = 3
BATCH_SIZE = 2       # Smallest batch size to prevent OOM
# Accumulate 8 steps to achieve an effective batch size of 16 (2 * 8)
GRADIENT_ACCUMULATION_STEPS = 8 
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1

# ... (Step 1: Load, Preprocess, and Split Data - unchanged)
# ... (Lines 30-44: df loading, label encoding, train/eval split)

# Convert split DataFrames into InputExample objects
train_examples = [
    InputExample(texts=[row['text']], label=float(row['label']))
    for _, row in train_df.iterrows()
]

# The SentenceLabelDataset wraps the examples and automatically creates positive pairs
# (texts with the same label) for the CoSENTLoss.
train_dataset = SentenceLabelDataset(train_examples)
# FIX 1: Set num_workers=0 to suppress the DataLoader UserWarnings
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0) 

# --- Step 2: Load Model and Define Training Objective ---

# Load the model, specifying the 'classification' task for LoRA tuning
print(f"Loading model: {MODEL_NAME} with default_task='{TASK}'...")
model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    max_seq_length=512, # <-- CRITICAL FIX for OOM on long texts
    model_kwargs={
        'default_task': TASK,
        # 'lora_main_params_trainable': False
    }
)

# Use CoSENTLoss, which is highly effective for contrastive learning and semantic similarity
train_loss = losses.CoSENTLoss(model=model)

# --- Step 3: Fine-Tune the Model ---

print("Starting fine-tuning...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FIX 2: Define Training Arguments explicitly for Gradient Accumulation
training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS, # <-- NEW LINE for stability/speed
    learning_rate=LEARNING_RATE,
    evaluation_strategy="no",
    save_strategy="epoch",
    warmup_ratio=WARMUP_RATIO,
    fp16=True, # Recommended for GPU memory efficiency
    logging_steps=50,
)

# Using model.fit() with training arguments
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    args=training_args, # <-- Pass arguments here
    warmup_steps=int(len(train_dataloader) * WARMUP_RATIO),
    output_path=OUTPUT_DIR,
    show_progress_bar=True,
    save_best_model=True,
)

print(f"\n✅ Fine-tuning complete. Model saved to {OUTPUT_DIR}")

# ... (Optional: Verification/Inference - unchanged)