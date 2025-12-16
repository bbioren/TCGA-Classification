#!/usr/bin/env python3
"""
fine_tune.py

Fine-tune jina-embeddings-v3 LoRA adapter for binary classification (OS column).
Mac-optimized: uses MPS if available, otherwise CPU.

Usage:
    python3 fine_tune.py --data /path/to/data.csv --text_col text --label_col OS

Dependencies:
    pip install -U torch transformers scikit-learn pandas numpy einops tqdm
"""

import os
import argparse
import math
from typing import List, Dict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, required=True, help="Path to CSV with text + label")
parser.add_argument("--text_col", type=str, default="text", help="Column name for text")
parser.add_argument("--label_col", type=str, default="OS", help="Column name for binary label (0/1)")
parser.add_argument("--model_name", type=str, default="jinaai/jina-embeddings-v3", help="HuggingFace model")
parser.add_argument("--max_length", type=int, default=256, help="Tokenizer max length")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size (keep small on Mac)")
parser.add_argument("--epochs", type=int, default=2, help="Number of training epochs (small by default)")
parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
parser.add_argument("--output_dir", type=str, default="fine_tuned_outputs", help="Where to save adapters/embeddings")
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

# ----------------------------
# Reproducibility + device
# ----------------------------
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")

os.makedirs(args.output_dir, exist_ok=True)

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv(args.data)
if args.text_col not in df.columns or args.label_col not in df.columns:
    raise ValueError(f"Columns {args.text_col} and/or {args.label_col} not found in CSV")

# Keep only rows with non-null text & labels
df = df[[args.text_col, args.label_col]].dropna()
# Ensure binary labels (0/1)
df[args.label_col] = df[args.label_col].astype(int)
if not set(df[args.label_col].unique()).issubset({0, 1}):
    raise ValueError("Label column must be binary 0/1")

train_df, val_df = train_test_split(df, test_size=0.15, random_state=args.seed, stratify=df[args.label_col])

print(f"#train = {len(train_df)}, #val = {len(val_df)}")

# ----------------------------
# Tokenizer + Model (trust_remote_code)
# ----------------------------
print("Loading tokenizer and model (trust_remote_code=True). This may download custom HF code...")
tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
base_model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
# base_model is the model with Jina custom code (LoRA adapters inside)

# ----------------------------
# Find/print adaptation map (which task id corresponds to 'classification')
# ----------------------------
adapt_map = None
# Try several possible attribute locations
for attr in ["_adaptation_map", "adaptation_map", "config"]:
    if hasattr(base_model, attr):
        candidate = getattr(base_model, attr)
        if isinstance(candidate, dict) and "classification" in candidate:
            adapt_map = candidate
            break
# fallback: sometimes it's under base_model.base_model or base_model.model
if adapt_map is None:
    for name, module in base_model.named_modules():
        if hasattr(module, "_adaptation_map"):
            candidate = getattr(module, "_adaptation_map")
            if isinstance(candidate, dict) and "classification" in candidate:
                adapt_map = candidate
                break

print("Adaptation map (if found):", adapt_map)
if adapt_map is not None and "classification" in adapt_map:
    classification_task_id = int(adapt_map["classification"])
    print("classification task id:", classification_task_id)
else:
    # As a safe fallback, default to 0 and warn the user
    classification_task_id = 0
    print("WARNING: Could not find adaptation map mapping. Defaulting classification_task_id=0. "
          "If results look wrong, inspect base_model._adaptation_map manually.")

# ----------------------------
# Dataset + collate function
# ----------------------------
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int]):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return {"text": self.texts[idx], "label": int(self.labels[idx])}

def collate_fn(batch: List[Dict]):
    texts = [b["text"] for b in batch]
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.float32)
    enc = tokenizer(texts, truncation=True, padding="longest", max_length=args.max_length, return_tensors="pt")
    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "labels": labels
    }

train_dataset = TextDataset(train_df[args.text_col].tolist(), train_df[args.label_col].tolist())
val_dataset   = TextDataset(val_df[args.text_col].tolist(), val_df[args.label_col].tolist())

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

# ----------------------------
# Wrapper classifier that uses adapter_mask when calling base_model
# ----------------------------
class JinaV3Classifier(nn.Module):
    def __init__(self, encoder: AutoModel, out_dim:int=None):
        super().__init__()
        self.encoder = encoder
        # get hidden size from config if available
        hidden = getattr(getattr(encoder, "config", None), "hidden_size", None)
        if hidden is None:
            # fallback - try common attr name
            hidden = getattr(encoder, "config", {}).get("hidden_size", 1024)
        out_dim = 1 if out_dim is None else out_dim
        self.classifier = nn.Linear(hidden, out_dim)

    def forward(self, input_ids, attention_mask, adapter_mask=None):
        # Many Jina models accept adapter_mask argument; pass it if provided.
        # Use kwargs approach so the call doesn't fail if the model signature differs.
        call_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if adapter_mask is not None:
            call_kwargs["adapter_mask"] = adapter_mask

        outputs = self.encoder(**call_kwargs)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        summed = torch.sum(last_hidden * mask, dim=1)
        counts = torch.clamp(torch.sum(mask, dim=1), min=1e-9)
        pooled = summed / counts  # (batch, hidden)
        logits = self.classifier(pooled)
        return logits.view(-1), pooled  # logits shape (batch,), pooled (batch,hidden)

# ----------------------------
# Freeze all params except LoRA / adapter params
# ----------------------------
# Heuristic: enable grad only for params containing 'lora' or 'adapter' or 'lora_' or 'adapter_' in their name
def unfreeze_lora_params(model):
    trainable = []
    for n, p in model.named_parameters():
        name = n.lower()
        if ("lora" in name) or ("adapter" in name) or ("lora_" in name) or ("adapter_" in name):
            p.requires_grad = True
            trainable.append(n)
        else:
            p.requires_grad = False
    return trainable

# Wrap base_model into classifier
classifier_model = JinaV3Classifier(base_model).to(device)

# Unfreeze LoRA-style params in the underlying encoder
trainable_param_names = unfreeze_lora_params(classifier_model.encoder)
# Also ensure classifier head is trainable
for n,p in classifier_model.classifier.named_parameters():
    p.requires_grad = True
    trainable_param_names.append("classifier." + n)

print(f"# trainable parameters (names sample) [{len(trainable_param_names)}]:")
for i, n in enumerate(trainable_param_names[:50]):
    print(" ", i+1, n)
# Show total trainable count
trainable_count = sum(p.numel() for p in classifier_model.parameters() if p.requires_grad)
total_params = sum(p.numel() for p in classifier_model.parameters())
print(f"Trainable params: {trainable_count:,} / Total params: {total_params:,}")

# ----------------------------
# Optimizer + Loss
# ----------------------------
optimizer = torch.optim.AdamW([p for p in classifier_model.parameters() if p.requires_grad], lr=args.lr)
criterion = nn.BCEWithLogitsLoss()

# ----------------------------
# Helper: compute AUC on a dataloader
# ----------------------------
def evaluate_auc(model, loader, task_id):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            adapter_mask = torch.full((input_ids.size(0),), task_id, dtype=torch.int32, device=device)
            logits, pooled = model(input_ids, attention_mask, adapter_mask=adapter_mask)
            probs = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            all_preds.extend(probs.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    try:
        auc = roc_auc_score(all_labels, all_preds) if len(set(all_labels)) > 1 else float("nan")
    except Exception:
        auc = float("nan")
    return auc

# ----------------------------
# Training loop
# ----------------------------
print("Starting training...")
best_val_auc = -math.inf
for epoch in range(args.epochs):
    classifier_model.train()
    running_loss = 0.0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    for batch in pbar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        adapter_mask = torch.full((input_ids.size(0),), classification_task_id, dtype=torch.int32, device=device)
        optimizer.zero_grad()
        logits, pooled = classifier_model(input_ids, attention_mask, adapter_mask=adapter_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        pbar.set_postfix({"loss": f"{running_loss/(pbar.n+1):.4f}"})

    # end epoch
    train_auc = evaluate_auc(classifier_model, train_loader, classification_task_id)
    val_auc = evaluate_auc(classifier_model, val_loader, classification_task_id)
    print(f"Epoch {epoch+1}: train AUC = {train_auc:.4f}, val AUC = {val_auc:.4f}")

    # Save best adapter weights (trainable params only)
    if val_auc is not None and not math.isnan(val_auc) and val_auc > best_val_auc:
        best_val_auc = val_auc
        adapter_state = {n: p.cpu().clone() for n, p in classifier_model.state_dict().items() if any(substr in n.lower() for substr in ['lora', 'adapter', 'classifier'])}
        save_path = os.path.join(args.output_dir, "best_adapter.pt")
        torch.save(adapter_state, save_path)
        print(f"Saved best adapter state to {save_path} (val_auc={val_auc:.4f})")

# ----------------------------
# After training: extract embeddings for all data and save
# ----------------------------
print("Extracting embeddings for full dataset...")

def encode_texts(texts: List[str], batch_size=8):
    classifier_model.eval()
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, truncation=True, padding=True, max_length=args.max_length, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        adapter_mask = torch.full((input_ids.size(0),), classification_task_id, dtype=torch.int32, device=device)
        with torch.no_grad():
            _, pooled = classifier_model(input_ids, attention_mask, adapter_mask=adapter_mask)
        embeddings.append(pooled.cpu().numpy())
    embeddings = np.vstack(embeddings)
    return embeddings

all_texts = df[args.text_col].tolist()
embs = encode_texts(all_texts, batch_size= max(1, args.batch_size))
print("Embeddings shape:", embs.shape)

# Save embeddings and mapping
emb_path = os.path.join(args.output_dir, "embeddings.npy")
np.save(emb_path, embs)
meta_path = os.path.join(args.output_dir, "meta.csv")
df.reset_index(drop=True).to_csv(meta_path, index=False)
print(f"Saved embeddings to {emb_path} and full dataframe to {meta_path}")

print("Done.")
