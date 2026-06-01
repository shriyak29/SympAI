import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

print("📂 Loading datasets...")

df      = pd.read_csv('data/dataset.csv')
desc_df = pd.read_csv('data/symptom_Description.csv')
prec_df = pd.read_csv('data/symptom_precaution.csv')
sev_df  = pd.read_csv('data/Symptom-severity.csv')

df.columns      = df.columns.str.strip()
desc_df.columns = desc_df.columns.str.strip()
prec_df.columns = prec_df.columns.str.strip()
sev_df.columns  = sev_df.columns.str.strip()

sev_df['Symptom'] = sev_df['Symptom'].str.strip().str.replace(' ', '_')

severity_dict = dict(zip(sev_df['Symptom'], sev_df['weight']))

df.fillna(0, inplace=True)
symptom_cols = [col for col in df.columns if col != 'Disease']

all_symptoms = []
for col in symptom_cols:
    all_symptoms.extend(df[col].unique())

all_symptoms = list(set([
    str(s).strip().replace(' ', '_').lower()
    for s in all_symptoms
    if s != 0 and str(s).strip() != ''
]))
all_symptoms.sort()

print(f"✅ Symptoms  : {len(all_symptoms)}")
print(f"✅ Diseases  : {df['Disease'].nunique()}")

def encode_row(row):
    present = [
        str(row[col]).strip().replace(' ', '_').lower()
        for col in symptom_cols
        if row[col] != 0 and str(row[col]).strip() != ''
    ]
    return [1 if s in present else 0 for s in all_symptoms]

print("⚙️  Encoding rows...")
X = df.apply(encode_row, axis=1, result_type='expand')
X.columns = all_symptoms

le = LabelEncoder()
y  = le.fit_transform(df['Disease'].str.strip())

print("🤖 Training...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=500,
    max_depth=None,
    min_samples_leaf=1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

acc = accuracy_score(y_test, model.predict(X_test))
print(f"✅ Accuracy  : {round(acc * 100, 2)}%")

os.makedirs('model', exist_ok=True)
with open('model/disease_model.pkl', 'wb') as f:
    pickle.dump({
        'model'        : model,
        'label_encoder': le,
        'all_symptoms' : all_symptoms,
        'severity_dict': severity_dict,
    }, f)

print("✅ Model saved!")
print("🎉 Training complete!")