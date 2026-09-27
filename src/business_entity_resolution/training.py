"""
Phase 5 & 6: training.py
Model training, hard negative mining, and validation loop.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_logreg(X_train, y_train, balanced=False):
    class_weight = 'balanced' if balanced else None
    model = LogisticRegression(class_weight=class_weight, random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def train_rf(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

import joblib

def save_model(model, filepath):
    joblib.dump(model, filepath)

def load_model(filepath):
    return joblib.load(filepath)
