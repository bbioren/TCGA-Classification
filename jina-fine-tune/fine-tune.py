import pandas as pd
from jina import DocumentArray
from jina.types.arrays.memmap import DocumentArrayMemmap
from jina import Flow
from jina.types.document import Document
from jina.excepts import BadClient

# Import InputExample depending on the version of Jina embeddings v2
from jina import InputExample

# ---------- 1. Load CSV safely ----------
df = pd.read_csv(
    "data.csv",
    quotechar='"',       # ensures text with commas/newlines is treated as one cell
    escapechar='\\',     # optional, in case of escaped quotes
    dtype={'text': str, 'OS': float}  # ensure correct types
)

# Strip whitespace from column names
df.columns = df.columns.str.strip()
print("Columns detected:", df.columns)

# ---------- 2. Prepare InputExamples ----------
examples = [
    InputExample(texts=[row["text"]], label=float(row["OS"]))
    for _, row in df.iterrows()
]

print(f"Prepared {len(examples)} examples.")

# ---------- 3. Save as memmap for Jina (optional but recommended for large datasets) ----------
da = DocumentArray([Document(text=ex.texts[0], tags={'label': ex.label}) for ex in examples])
DocumentArrayMemmap(da, 'dataset_mmap')

print("Dataset saved to 'dataset_mmap'.")

# ---------- 4. Fine-tuning setup ----------
# Example: assuming you are using Jina's Transformer-based embeddings
from jina import Client

f = Flow().add(
    uses='jinahub://TransformerTorchEncoder/v2'  # replace with the embeddings model you want
)

with f:
    # Index all documents into your flow
    f.post(on='/index', inputs=da)

print("Fine-tuning complete.")
