import pandas as pd
import numpy as np

CUTOFF = 365

# check filename spelling
luad_clinical = pd.read_csv("data/luad_clincal.tsv", sep="\t", na_values="'--")
luad_exposure = pd.read_csv("data/lusc_exposure.tsv", sep="\t", na_values="'--")

data = pd.concat([luad_clinical, luad_exposure], axis=1)
data = data.loc[:, ~data.columns.duplicated()]


# Convert the specific columns that should be numeric to numeric dtype
num_cols = ["diagnoses.days_to_last_follow_up", "demographic.days_to_death"]
for col in num_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# initialize OS
data["OS"] = -1  # 1 --> survive, 0 --> death

# build element-wise conditions
cond_alive = (
    (data["demographic.vital_status"] == "Alive")
    & data["diagnoses.days_to_last_follow_up"].notna()
    & (data["diagnoses.days_to_last_follow_up"] >= CUTOFF)
)

cond_dead = (
    (data["demographic.vital_status"] == "Dead")
    & data["demographic.days_to_death"].notna()
    & (data["demographic.days_to_death"] < CUTOFF) # cutoff days
)

data["OS"] = np.select([cond_alive, cond_dead], [1, 0], default=-1)

# quick checks
print(data[["demographic.vital_status",
                     "diagnoses.days_to_last_follow_up",
                     "demographic.days_to_death",
                     "OS"]].head())

print("dtypes:\n", data[["diagnoses.days_to_last_follow_up",
                                 "demographic.days_to_death"]].dtypes)
print("OS value counts:\n", data["OS"].value_counts(dropna=False))

data = data[data["OS"] != -1] # drop invalid patients

data = data.dropna(axis=1, how="all") # drop columns of all nan

data = data[data["diagnoses.diagnosis_is_primary_disease"] == True] # drop non-primary cases

# function to concatenate fields if different
def combine_group(group):
    def combine_column(x):
        vals = x.dropna().unique()
        if len(vals) == 0:
            return None  # all NaN
        elif len(vals) == 1:
            return vals[0]
        else:
            # convert all to string before joining
            return '; '.join(sorted(map(str, vals)))
    return group.apply(combine_column)

# remove repeats
data = data.groupby("cases.submitter_id", as_index=False).apply(combine_group)

# remove indexing
data.reset_index(drop=True, inplace=True)

# take intersect with pathology reports
filter = pd.read_csv("data/TCGA_Reports.csv")
filter = filter["patient_filename"]
mask = data["cases.submitter_id"].apply(
    lambda x: any(x in full_id for full_id in filter)
)
data = data[mask].copy()


# drop irrelevant features
data = data.drop(columns=[
    "cases.case_id",
    "demographic.demographic_id",
    "demographic.demographic_id",
    "diagnoses.diagnosis_id",
    "treatments.treatment_id",
    "demographic.submitter_id",
    "diagnoses.submitter_id",
    "treatments.submitter_id",
    "demographic.vital_status",
    "diagnoses.days_to_last_follow_up",
    "demographic.days_to_death",
    "cases.consent_type",
    "cases.lost_to_followup",
    "cases.days_to_consent"
])


# save to csv
data.to_csv('out/data.csv', index=False)

print("OS value counts:\n", data["OS"].value_counts(dropna=False))
