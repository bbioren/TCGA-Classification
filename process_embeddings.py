import pandas as pd
import numpy as np

data = pd.read_csv("out/embeddings.csv")

# take intersect with pathology reports
filter = pd.read_csv("out/data.csv")
filter = filter["cases.submitter_id"]
mask = data["patient"].apply(
    lambda x: any(partial_id in x for partial_id in filter)
)
data = data[mask].copy()

data.reset_index(drop=True, inplace=True)


data.to_csv('out/embeddings_proccessed.csv', index=False)
