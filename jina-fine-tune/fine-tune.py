# fine-tune.py

import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import os

# -----------------------------
# CONFIG
# -----------------------------
MODEL = "jinaai/jina-embeddings-v3"
DATA_CSV = "data.csv"           # your CSV file with 'sentence' and 'label' columns
OUTPUT_DIR = "jina-finetuned-lora"
BATCH_SIZE = 16
EPOCHS = 3
WARMUP_STEPS = 100

# -----------------------------
# Load data
# -----------------------------
df = pd.read_csv(DATA_CSV)
examples = [
    InputExample(texts=[row["sentence"]], label=float(row["label"]))
    for _, row in df.iterrows()
]

dataloader = DataLoader(examples, batch_size=BATCH_SIZE, shuffle=True)

# -----------------------------
# Load model
# -----------------------------
model = SentenceTransformer(
    MODEL,
    trust_remote_code=True,
    model_kwargs={"default_task": "classification"}
)

# -----------------------------
# Fine-tuning
# -----------------------------
train_loss = losses.CosineSimilarityLoss(model)

print("Starting fine-tuning...")
model.fit(
    train_objectives=[(dataloader, train_loss)],
    epochs=EPOCHS,
    warmup_steps=WARMUP_STEPS
)

# -----------------------------
# Save model
# -----------------------------
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save(OUTPUT_DIR)
print(f"Fine-tuned model saved to {OUTPUT_DIR}")

# -----------------------------
# Optional evaluation (AUC)
# -----------------------------
print("Evaluating model AUC on training set...")
sentences = df['sentence'].tolist()
labels = df['label'].tolist()

embeddings = model.encode(sentences)
clf = LogisticRegression(max_iter=1000).fit(embeddings, labels)
preds = clf.predict_proba(embeddings)[:, 1]

auc = roc_auc_score(labels, preds)
print("Training set AUC:", auc)
