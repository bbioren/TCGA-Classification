# %%
# === IMPORTS ===
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# %%
# === LOAD DATA ===
embeddings_df = pd.read_csv("out/embeddings_processed.csv")
structured_df = pd.read_csv("out/data.csv")

# Split embedding strings into columns
embeddings_df = embeddings_df['embedding'].str.split(',', expand=True)
embeddings_df = embeddings_df.astype(float)
embeddings_df.columns = [f'emb_{i}' for i in range(embeddings_df.shape[1])]

# Combine structured and embedding data
data = pd.concat([embeddings_df, structured_df], axis=1)
data = data.drop(columns=["cases.submitter_id"])

# %%
# === DATA CLEANING ===
# Convert objects to string
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].astype(str)

# Separate numeric and categorical columns
num_cols = data.select_dtypes(include=[np.number]).columns
cat_cols = data.select_dtypes(include=['object']).columns

# Impute numeric columns
num_imputer = SimpleImputer(strategy='median')
for col in num_cols:
    data[col] = num_imputer.fit_transform(data[[col]])

# Impute categorical columns
if len(cat_cols) > 0:
    cat_imputer = SimpleImputer(strategy='most_frequent')
    data[cat_cols] = pd.DataFrame(
        cat_imputer.fit_transform(data[cat_cols]),
        columns=cat_cols,
        index=data.index
    )

# Encode categorical columns
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le

# %%
# === DEFINE FEATURES AND TARGET ===
X = data.drop('OS', axis=1)
y = data['OS']

# Identify feature groups
embedding_cols = [col for col in X.columns if col.startswith('emb_')]
data_cols = [col for col in X.columns if not col.startswith('emb_')]

print(f"Number of embedding features: {len(embedding_cols)}")
print(f"Number of structured features: {len(data_cols)}")

# %%
# === UNIVARIATE FEATURE SELECTION (Structured Features Only) ===
# Keep top 10 features for example (you can tune k)
univariate_selector = SelectKBest(score_func=f_classif, k=10)
X_structured_selected = univariate_selector.fit_transform(X[data_cols], y)

selected_structured_cols = np.array(data_cols)[univariate_selector.get_support()].tolist()
print(f"Selected structured features (univariate): {selected_structured_cols}")

# %%
# === PCA FOR EMBEDDINGS ===
pca_n_components = 5  # You can tune this based on explained variance
pca_pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=pca_n_components))
])
X_embeddings_pca = pca_pipeline.fit_transform(X[embedding_cols])

# %%
# === COMBINE PCA EMBEDDINGS AND SELECTED STRUCTURED FEATURES ===
X_combined = np.hstack([X_embeddings_pca, X_structured_selected])
all_feature_names = [f'PCA_{i+1}' for i in range(pca_n_components)] + selected_structured_cols
print(f"Total combined features: {len(all_feature_names)}")

# %%
# === CROSS-VALIDATION WITH RANDOM FOREST ===
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_combined, y, cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"Mean CV ROC AUC: {cv_scores.mean():.4f}")
print(f"Std CV ROC AUC: {cv_scores.std():.4f}")

# %%
# === TRAIN FINAL MODEL ON FULL DATA ===
model.fit(X_combined, y)
importances = model.feature_importances_
feature_importances_df = pd.DataFrame({
    'Feature': all_feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot top 20 features
top_n = min(20, len(feature_importances_df))
plt.figure(figsize=(12, 7))
plt.barh(feature_importances_df['Feature'].head(top_n), feature_importances_df['Importance'].head(top_n))
plt.xlabel('Feature Importance')
plt.title(f'Top {top_n} Feature Importances')
plt.gca().invert_yaxis()
plt.show()

print("\nTop 10 Features:")
print(feature_importances_df.head(10))

# %%
# === OPTIONAL: GRID SEARCH TUNING (Structured Only) ===
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5],
    'max_features': ['sqrt', 'log2']
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_grid=param_grid,
    scoring='roc_auc',
    cv=skf,
    verbose=2,
    n_jobs=-1
)
grid_search.fit(X_structured_selected, y)
print("\nBest parameters (structured only):", grid_search.best_params_)
print("Best ROC AUC (structured only):", grid_search.best_score_)

# Scores from SelectKBest
scores = univariate_selector.scores_
selected = univariate_selector.get_support()
feature_scores_df = pd.DataFrame({
    'Feature': data_cols,
    'F-score': scores,
    'Selected': selected
}).sort_values('F-score', ascending=True)

plt.figure(figsize=(10, 8))
plt.barh(feature_scores_df['Feature'], feature_scores_df['F-score'],
         color=['green' if s else 'gray' for s in feature_scores_df['Selected']])
plt.xlabel('ANOVA F-Score')
plt.ylabel('Structured Feature')
plt.title('Univariate Feature Selection (Green = Selected)')
plt.show()


# Univariate F-scores
scores = univariate_selector.scores_
selected = univariate_selector.get_support()

feature_scores_df = pd.DataFrame({
    'Feature': data_cols,
    'F-score': scores,
    'Selected': selected
}).sort_values('F-score', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(feature_scores_df['Feature'], feature_scores_df['F-score'],
         color=['green' if s else 'gray' for s in feature_scores_df['Selected']])
plt.xlabel('ANOVA F-Score')
plt.ylabel('Structured Feature')
plt.title('Univariate Feature Selection (Green = Selected)')
plt.show()

# Already computed feature_importances_df
feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=True)

plt.figure(figsize=(12, 8))
plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'], color='skyblue')
plt.xlabel('Feature Importance (Random Forest)')
plt.ylabel('Feature')
plt.title('Feature Importances from Random Forest')
plt.show()


# Prepare combined DataFrame
combined_df = pd.DataFrame({
    'Feature': feature_scores_df['Feature'].tolist() + [f'PCA_{i+1}' for i in range(pca_n_components)],
    'Univariate_F': list(feature_scores_df['F-score']) + [0]*pca_n_components,
    'RF_Importance': list(feature_importances_df['Importance'])[:len(feature_scores_df)] + list(feature_importances_df['Importance'])[:pca_n_components]
})

combined_df = combined_df.sort_values('RF_Importance', ascending=True)

plt.figure(figsize=(14, 8))
plt.barh(combined_df['Feature'], combined_df['Univariate_F'], color='gray', alpha=0.5, label='Univariate F-Score')
plt.barh(combined_df['Feature'], combined_df['RF_Importance'], color='blue', alpha=0.7, label='RF Importance')
plt.xlabel('Score / Importance')
plt.ylabel('Feature')
plt.title('Comparison: Univariate F-Score vs Random Forest Importance')
plt.legend()
plt.show()
