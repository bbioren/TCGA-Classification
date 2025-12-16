import pandas as pd
import numpy as np

sentences = pd.read_csv('data/TCGA_Reports.csv')
sentences["patient_filename"] = sentences["patient_filename"].str.split(".").str[0]

structured = pd.read_csv('out/data.csv')

## Remove all of the sentences not in the data
sentences = sentences[sentences["patient_filename"].isin(structured["cases.submitter_id"])]

## add the OS label
merged = sentences.merge(
    structured[['cases.submitter_id', 'OS']], 
    left_on='patient_filename', 
    right_on='cases.submitter_id', 
    how='left' # Keep all sentence rows, fill matching OS info
)

print(merged.columns.tolist())

## drop irrelevant
merged = merged[['text', 'OS']]

## write to out
merged.to_csv('out/fine_tune_sentence_label_tuple.csv', index=False)
print(merged.columns.tolist())
 