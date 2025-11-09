import json
import yaml
from faiss_index import FAISSIndex
import joblib
from rag_module import generate_recommendation
from custom_data_loader import flatten_api_spec
import os

# Use the FAISS index for RAG recommendations
faiss_index = FAISSIndex('kb_labelled.json')

def analyze_api_request(request_text):
    """
    Analyzes a flattened API spec using both ML prediction and RAG.
    """
    # Load the trained model and vectorizer
    try:
        clf = joblib.load('logreg_model_labeled.pkl')
        vectorizer = joblib.load('vectorizer_labeled.pkl')
    except FileNotFoundError:
        print("Error: The trained model files (logreg_model_labeled.pkl or vectorizer_labeled.pkl) were not found.")
        print("Please run train_model.py first to train the model and save the files.")
        return

    # --- RAG Recommendation ---
    print("\n--- Compliance & Recommendations ---")

    # Retrieve top 3 most relevant security guidelines from the KB
    results = faiss_index.search(request_text, top_k=3)
    recommendation_context = "Relevant Security Guidelines:\n"
    for text, _ in results:
        recommendation_context += f"- {text}\n"

    # Generate a final, context-aware recommendation using the local LLM
    final_recommendation = generate_recommendation(request_text, recommendation_context)
    print(final_recommendation)

    # --- Logistic Regression Prediction ---
    print("\n--- Logistic Regression Prediction ---")
    x_vec = vectorizer.transform([request_text])
    pred = clf.predict(x_vec)[0]

    risk_score = clf.predict_proba(x_vec)[0]
    vulnerable_prob = risk_score[1]

    print("Looks Vulnerable" if pred == 1 else "Looks Safe")
    print(f"Confidence Score: {vulnerable_prob:.2f}")

if __name__ == "__main__":
    file_path = input("Enter path to API spec (YAML file): ")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        exit(1)

    try:
        with open(file_path, "r") as f:
            api_spec = yaml.safe_load(f)
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML file: {exc}")
        exit(1)

    flattened_text = flatten_api_spec(api_spec)
    analyze_api_request(flattened_text)