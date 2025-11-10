import pandas as pd
import numpy as np

CUTOFF = 730

# FILES TO READ FROM
clinical = pd.concat([pd.read_csv("data/luad_clincal.tsv", sep="\t", na_values="'--"),
                      pd.read_csv("data/lusc_clinical.tsv", sep="\t", na_values="'--")],
                      ignore_index=True)

exposure = pd.concat([pd.read_csv("data/luad_exposure.tsv", sep="\t", na_values="'--"),
                      pd.read_csv("data/lusc_exposure.tsv", sep="\t", na_values="'--")],
                      ignore_index=True)

# only consider primary disease
clinical = clinical[clinical["diagnoses.diagnosis_is_primary_disease"] == True]

# coalesce primary disease cases
def custom_agg(series):
    """
    Returns the single unique non-NaN value if all non-NaN values are the same,
    otherwise returns a sorted list of all unique values (converted to strings).
    """
    # Get unique values, explicitly excluding NaNs first for the condition check
    unique_non_na = series.dropna().unique()

    if len(unique_non_na) <= 1:
        # Case 1: All non-NaN values are identical or the group is all NaN.
        if len(unique_non_na) == 1:
            # Safely return the single unique non-NaN value
            return unique_non_na[0]
        else:
            # Group was all NaN/missing
            return np.nan
    else:
        # Case 2: Multiple unique non-NaN values exist.
        
        # Get ALL unique values (not including nan)
        
        # Convert all unique elements to strings before returning and sorting.
        # This prevents the TypeError ('<' not supported between 'str' and 'float').
        unique_str_values = [str(x) for x in unique_non_na]
        
        return (unique_str_values)
    
# The GroupBy and Aggregation Step
clinical = clinical.groupby("cases.submitter_id", as_index=False).agg(custom_agg)

## MORE PREPROCESSING
# combine exposure and clinical sets
data = pd.concat(
    [clinical.set_index('cases.submitter_id'), exposure.set_index('cases.submitter_id')],
    axis=1
).reset_index()

## OS ASSIGNMENT LOGIC
data["OS"] = -1  # 1 --> survive, 0 --> death

# build element-wise conditions
cond_alive = (
    ((data["demographic.vital_status"] == "Alive")
    & data["diagnoses.days_to_last_follow_up"].notna()
    & (data["diagnoses.days_to_last_follow_up"] >= CUTOFF))
    or ((data["demographic.vital_status"] == "Dead")
    & data["diagnoses.days_to_last_follow_up"].notna()
    & (data["diagnoses.days_to_last_follow_up"] >= CUTOFF))
)

cond_dead = (
    (data["demographic.vital_status"] == "Dead")
    & data["demographic.days_to_death"].notna()
    & (data["demographic.days_to_death"] < CUTOFF) # cutoff days
)

data["OS"] = np.select([cond_alive, cond_dead], [1, 0], default=-1)

data.to_csv('out/data_full.csv', index=False)

print("OS value counts:\n", data["OS"].value_counts(dropna=False))

# drop all invalid examples
data = data[data["OS"] != -1]

# drop unrelated/sparese features
data.dropna(axis='columns', how='all', inplace=True)  # empty columns
data = data.drop(labels=["treatments.treatment_id", "treatments.submitter_id",
                                 "diagnoses.year_of_diagnosis", "diagnoses.submitter_id",
                                 "diagnoses.diagnosis_id", "diagnoses.classification_of_tumor",
                                 "demographic.submitter_id", "demographic.demographic_id",
                                 "demographic.age_is_obfuscated", "cases.case_id",
                                 "cases.consent_type", "cases.case_id",
                                 "project.project_id"], axis='columns') # project.project_id


embeddings = pd.read_csv("out/embeddings.csv")
embeddings["cases.submitter_id"] = embeddings["patient"].str.split(".").str[0]
embeddings.drop(labels="patient", axis="columns", inplace=True)
embeddings = embeddings[['cases.submitter_id', 'embedding']]

embeddings = embeddings[embeddings["cases.submitter_id"].isin(data["cases.submitter_id"])]
data = data[data["cases.submitter_id"].isin(embeddings["cases.submitter_id"])]

print("OS value counts:\n", data["OS"].value_counts(dropna=False))

print("Examples: ", data.shape[0])
print("Features: ", data.shape[1])

embeddings.to_csv('out/embeddings_processed.csv', index=False)
data.to_csv('out/data.csv', index=False)
