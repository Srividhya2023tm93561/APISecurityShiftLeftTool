import os
import json
import yaml
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import pandas as pd


def load_custom_dataset(manifest_file="custom_dataset_150.json"):
    """Load API YAML data and labels for ML training."""
    dataset_path = os.path.join(os.getcwd(), manifest_file)
    print(f"Loading dataset from: {dataset_path}")

    with open(dataset_path, "r") as f:
        data = json.load(f)

    entries = data.get("api_specs", [])
    print(f"Total entries found: {len(entries)}")
    if not entries:
        return [], []

    texts, labels = [], []
    for entry in entries:
        try:
            spec_file = os.path.join(os.getcwd(), "dataset", entry["filename"])
            with open(spec_file, "r") as f:
                content = yaml.safe_load(f)
                texts.append(json.dumps(content))  # Convert YAML to string
                labels.append(int(entry["vulnerable"]))
        except Exception as e:
            print(f"Skipping file {spec_file} due to error: {e}")

    print(f"Successfully loaded {len(texts)} API specs.\n")
    return texts, labels


def evaluate_model(model_name, model, X, y):
    """Train and evaluate model using 5-Fold cross-validation."""
    print(f"\n=== Evaluating {model_name} (Pipeline) with 5-Fold CV ===")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_pred = cross_val_predict(model, X, y, cv=cv)

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n--- {model_name} Results (CV aggregated) ---")
    print(f"Accuracy:  {acc:.3f}")
    print(f"Precision: {prec:.3f}")
    print(f"Recall:    {rec:.3f}")
    print(f"F1-Score:  {f1:.3f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred, digits=3))

    return model_name, acc, prec, rec, f1


def main():
    texts, labels = load_custom_dataset()
    if not texts:
        print("Dataset not found or empty. Aborting.")
        return

    # --- Enhanced TF-IDF Vectorizer ---
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 3),
        max_features=3000,
        token_pattern=r"(?u)\b[a-zA-Z_][a-zA-Z0-9_]+\b",
        stop_words='english',
        sublinear_tf=True
    )

    # --- Models with Tuned Parameters ---
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, C=2.0, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_split=4, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_split=4,
                                                class_weight='balanced', random_state=42)
    }

    results = []

    for name, clf in models.items():
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", clf)])
        result = evaluate_model(name, pipeline, texts, labels)
        results.append(result)

    # --- Save Comparison Summary ---
    df = pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1-Score"])
    print("\n==============================")
    print("Model Comparison Summary (5-Fold CV, pipeline, noise)")
    print("==============================")
    print(df)
    df.to_csv("model_comparison_results_pipeline_cv.csv", index=False)
    print("\nResults saved to model_comparison_results_pipeline_cv.csv")


if __name__ == "__main__":
    main()
