import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
from tqdm import tqdm
import random
from collections import defaultdict

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        features = F.normalize(features, dim=1)
        logits = torch.matmul(features, features.T) / self.temperature

        logits_mask = torch.ones_like(mask) - torch.eye(mask.size(0), device=device)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
        return -mean_log_prob_pos.mean()

class CSVDataset(Dataset):
    def __init__(self, path):
        self.df = pd.read_csv(path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df.iloc[idx]["sentence"], int(self.df.iloc[idx]["label"])

class BalancedBatchSampler:
    def __init__(self, labels, samples_per_class):
        self.indexes = defaultdict(list)
        for i, y in enumerate(labels):
            self.indexes[y].append(i)
        self.labels = list(self.indexes.keys())
        self.samples_per_class = samples_per_class

    def __iter__(self):
        while True:
            batch = []
            for y in self.labels:
                batch.extend(random.sample(self.indexes[y], self.samples_per_class))
            random.shuffle(batch)
            yield batch

def mean_pooling(hidden_states, attention_mask):
    mask = attention_mask.unsqueeze(-1).float()
    return (hidden_states * mask).sum(1) / mask.sum(1)

MODEL = "jinaai/jina-embeddings-v3"

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL, trust_remote_code=True)

model.gradient_checkpointing_enable()

for p in model.parameters():
    p.requires_grad = False

# unfreeze top 4 layers
for layer in model.encoder.layer[-4:]:
    for p in layer.parameters():
        p.requires_grad = True

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = CSVDataset("data.csv")
labels = dataset.df["label"].tolist()

sampler = BalancedBatchSampler(labels, samples_per_class=16)

loader = DataLoader(
    dataset,
    batch_sampler=sampler,
    collate_fn=lambda batch: ([x[0] for x in batch],
                              torch.tensor([x[1] for x in batch]))
)

criterion = SupConLoss()
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=2e-5)

model.train()
steps = 1500

for step, (sentences, y) in zip(range(steps), loader):
    optimizer.zero_grad()

    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(device)

    outputs = model(**inputs)
    embeddings = mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])

    loss = criterion(embeddings, y.to(device))
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")

model.save_pretrained("jina-finetuned")
tokenizer.save_pretrained("jina-finetuned")
