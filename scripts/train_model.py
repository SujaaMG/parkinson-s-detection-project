# scripts/train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.utils import resample
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import XGBoost
from xgboost import XGBClassifier

# ------------------- Load Dataset -------------------
df = pd.read_csv("data/features/features.csv")
print(f"📂 Loaded dataset with {df.shape[0]} samples and {df.shape[1]} features")

# Drop non-feature columns
X = df.drop(columns=["label", "file_name"], errors="ignore")
y = df["label"]

# ------------------- Balance Classes -------------------
print("⚖️ Balancing dataset (equal PD & HC samples)...")
df_majority = df[df.label == 0]
df_minority = df[df.label == 1]

df_minority_upsampled = resample(df_minority,
                                 replace=True,
                                 n_samples=len(df_majority),
                                 random_state=42)

df_balanced = pd.concat([df_majority, df_minority_upsampled])
X = df_balanced.drop(columns=["label", "file_name"], errors="ignore")
y = df_balanced["label"]
print(f"✅ After balancing: {y.value_counts().to_dict()}")

# ------------------- Split Dataset -------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ------------------- Scaling -------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------- XGBoost Classifier (GPU) -------------------
xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    tree_method='hist',  # use 'hist' instead of 'gpu_hist'
    random_state=42
)

# ------------------- Hyperparameter Grid -------------------
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1],
    'reg_lambda': [1, 1.5, 2]
}


grid_search = GridSearchCV(
    estimator=xgb,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

# ------------------- Cross-validation & Grid Search -------------------
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_
print(f"🏆 Best hyperparameters: {grid_search.best_params_}")

cv_scores = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"🔁 Cross-validation accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# ------------------- Train Final Model -------------------
best_model.fit(X_train_scaled, y_train)
y_pred = best_model.predict(X_test_scaled)

# ------------------- Evaluation -------------------
print("\n🎯 FINAL MODEL EVALUATION")
print("✅ Test Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['HC', 'PD'], yticklabels=['HC', 'PD'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# ------------------- Feature Importance -------------------
importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\n🌲 Top 10 Important Features:")
print(importances.head(10))

plt.figure(figsize=(8,5))
importances.head(15).plot(kind='bar')
plt.title("Top 15 Feature Importances (XGBoost)")
plt.ylabel("Importance Score")
plt.tight_layout()
plt.show()

# ------------------- Save Model -------------------
os.makedirs("models", exist_ok=True)
joblib.dump((best_model, scaler), "models/model.pkl")
print("\n💾 Model and scaler saved to models/model.pkl")

print("\n✅ Training complete with XGBoost + GPU acceleration + hyperparameter tuning.")
