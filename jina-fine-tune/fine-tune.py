import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from sentence_transformers.datasets import SentenceLabelDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
from sentence_transformers import SentenceTransformerTrainingArguments 

# --- Configuration (A100 Optimized) ---
MODEL_NAME = "jinaai/jina-embeddings-v3"
CSV_FILE = "/root/TCGA-Classification/jina-fine-tune/data.csv"
OUTPUT_DIR = "jina-v3-finetuned-classification"
TASK = "classification"
NUM_EPOCHS = 3          
BATCH_SIZE = 32         # OPTIMIZED: High batch size for 80GB VRAM
LEARNING_RATE = 2e-5
WARMUP_RATIO = 0.1
# NOTE: GRADIENT_ACCUMULATION_STEPS is implicitly 1 (not needed)

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

# The SentenceLabelDataset wraps the examples
train_dataset = SentenceLabelDataset(train_examples)
# ESSENTIAL FIX: num_workers=0 to prevent indexing/empty batch issues with IterableDataset
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0)

# --- Step 2: Load Model and Define Training Objective ---

print(f"Loading model: {MODEL_NAME} with default_task='{TASK}'...")
model = SentenceTransformer(
    MODEL_NAME,
    trust_remote_code=True,
    max_seq_length=512, # ESSENTIAL FIX: Truncate long documents to prevent OOM spikes
    model_kwargs={
        'default_task': TASK,
    }
)

# Use CoSENTLoss
train_loss = losses.CoSENTLoss(model=model)

# --- Step 3: Fine-Tune the Model ---

print("Starting fine-tuning...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define Training Arguments
training_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=1, # Set to 1 because BATCH_SIZE is high enough
    learning_rate=LEARNING_RATE,
    evaluation_strategy="no",
    save_strategy="epoch",
    warmup_ratio=WARMUP_RATIO,
    fp16=True, # Keep FP16 for efficiency
    logging_steps=50,
)

# Calculate warmup steps
total_steps = len(train_dataloader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)


# Using model.fit() with training arguments
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    args=training_args,
    warmup_steps=warmup_steps,
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