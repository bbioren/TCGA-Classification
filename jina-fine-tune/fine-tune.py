# fine-tune.py

import pandas as pd
from docarray import Document, DocumentArray

# Path to your CSV
csv_file = 'cancer_reports.csv'

# Load CSV
df = pd.read_csv(csv_file)

# Check columns
if 'text' not in df.columns or 'OS' not in df.columns:
    raise ValueError("CSV must have 'text' and 'OS' columns")

# Create DocumentArray
docs = DocumentArray()
for _, row in df.iterrows():
    docs.append(
        Document(text=str(row['text']), tags={'OS': float(row['OS'])})
    )

print(f'Total documents: {len(docs)}')

# Save memory-mapped version for large datasets
docs_memmap = DocumentArray.memmap('./docs_memmap', docs)
print("Memory-mapped DocumentArray saved at ./docs_memmap")
