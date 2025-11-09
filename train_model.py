import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score
import joblib

from custom_data_loader import load_custom_dataset

def train_and_evaluate_model():
    """Loads custom dataset, trains a Logistic Regression model, and saves it."""
    print("Starting model training on custom API dataset...")
    texts, labels = load_custom_dataset()

    if not texts:
        print("Training data could not be loaded. Aborting.")
        return

    # Vectorize text using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(texts)
    y = np.array(labels)

    # Use stratified train-test split to ensure class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train classifier
    clf = LogisticRegression(max_iter=500)
    clf.fit(X_train, y_train)

    # Predict on test set
    y_pred = clf.predict(X_test)

    # Evaluate on test set
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n=== Test Set Evaluation ===")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save model and vectorizer
    joblib.dump(clf, 'logreg_model_labeled.pkl')
    joblib.dump(vectorizer, 'vectorizer_labeled.pkl')
    print("\nLabeled model and vectorizer saved.")

if __name__ == "__main__":
    train_and_evaluate_model()