"""Model Training and Evaluation Pipeline.

Trains, evaluates, and exports:
1. Category Classification Model (NLP: TF-IDF Vectorizer + Logistic Regression / Linear Classifier).
2. Candidate Screening & Recommendation Model (Tabular + Text ColumnTransformer Pipeline).

Outputs performance metrics, cross-validation results, and saves models to `models/`.
"""

import os
import sys
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Ensure src is on path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_text


def find_data_file(filename: str) -> str:
    """Finds dataset in data/ or root directory."""
    candidates = [
        os.path.join(BASE_DIR, "data", filename),
        os.path.join(BASE_DIR, filename)
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not locate dataset: {filename}")


def train_category_model(data_path: str, models_dir: str):
    """Trains the Job Category Classification Model."""
    print("\n" + "="*60)
    print(" 1. TRAINING JOB CATEGORY CLASSIFIER")
    print("="*60)
    
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path} (Shape: {df.shape})")

    # Clean text
    df["clean_resume"] = df["Resume"].apply(clean_text)
    print(f"Dataset samples: {df.shape[0]} across {df['Category'].nunique()} job categories.")

    X = df["clean_resume"]
    y = df["Category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            stop_words="english",
            max_features=5000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            random_state=42
        ))
    ])

    # 5-fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
    print(f"5-Fold CV Accuracy : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # Fit and evaluate
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Test Accuracy      : {acc*100:.2f}%")
    print(f"Test F1-Score      : {f1*100:.2f}%")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "category_model.pkl")
    joblib.dump(pipeline, out_path)
    # Also save to root for backwards compatibility
    joblib.dump(pipeline, os.path.join(BASE_DIR, "category_model.pkl"))
    print(f"-> Model successfully saved to: {out_path}")
    return pipeline


def train_screening_model(data_path: str, models_dir: str):
    """Trains the Recruiter Recommendation & Screening Model."""
    print("\n" + "="*60)
    print(" 2. TRAINING CANDIDATE SCREENING / RECOMMENDATION MODEL")
    print("="*60)

    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {data_path} (Shape: {df.shape})")

    df["Certifications"] = df["Certifications"].fillna("None")

    features = [
        "Skills",
        "Experience (Years)",
        "Education",
        "Certifications",
        "Job Role",
        "Projects Count"
    ]
    target = "Recruiter Decision"

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    text_col = "Skills"
    cat_cols = ["Education", "Certifications", "Job Role"]
    num_cols = ["Experience (Years)", "Projects Count"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("skills_text", TfidfVectorizer(stop_words="english", max_features=100), text_col),
            ("categories", Pipeline([
                ("fill_missing", SimpleImputer(strategy="most_frequent")),
                ("one_hot", OneHotEncoder(handle_unknown="ignore"))
            ]), cat_cols),
            ("numbers", SimpleImputer(strategy="median"), num_cols)
        ]
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))
    ])

    # 5-fold cross validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_weighted")
    print(f"5-Fold CV F1-Score : {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

    # Fit and evaluate
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    print(f"Test Accuracy      : {acc*100:.2f}%")
    print(f"Test F1-Score      : {f1*100:.2f}%")
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred))

    # Save model
    os.makedirs(models_dir, exist_ok=True)
    out_path = os.path.join(models_dir, "screening_model.pkl")
    joblib.dump(pipeline, out_path)
    # Also save to root for backwards compatibility
    joblib.dump(pipeline, os.path.join(BASE_DIR, "screening_model.pkl"))
    print(f"-> Model successfully saved to: {out_path}")
    return pipeline


def main():
    cat_data = find_data_file("UpdatedResumeDataSet.csv")
    screen_data = find_data_file("AI_Resume_Screening.csv")
    models_dir = os.path.join(BASE_DIR, "models")

    train_category_model(cat_data, models_dir)
    train_screening_model(screen_data, models_dir)
    print("\n" + "="*60)
    print(" ALL MODELS SUCCESSFULLY TRAINED & EXPORTED!")
    print("="*60)


if __name__ == "__main__":
    main()
