import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from custom_data_loader import load_custom_dataset

# Make output folder
os.makedirs("figures", exist_ok=True)

# ================================
# BAR CHART COMPARISON
# ================================
def plot_model_comparison():
    df = pd.read_csv("model_comparison_results_pipeline_cv.csv")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    plt.figure(figsize=(10, 6))
    bar_width = 0.2
    positions = np.arange(len(df["Model"]))

    for i, metric in enumerate(metrics):
        plt.bar(positions + i * bar_width, df[metric], width=bar_width, label=metric)

    plt.xticks(positions + bar_width * 1.5, df["Model"], fontsize=12)
    plt.ylim(0, 1.05)
    plt.ylabel("Score", fontsize=12)
    plt.title("Model Performance Comparison", fontsize=14, weight='bold')
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/1_model_comparison_bar_chart.png", dpi=300)
    plt.close()
    print("Saved: figures/1_model_comparison_bar_chart.png")

# ================================
# RADAR / SPIDER CHART
# ================================
def plot_radar_chart():
    df = pd.read_csv("model_comparison_results_pipeline_cv.csv")
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)

    for i, row in df.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row["Model"], linewidth=2)
        ax.fill(angles, values, alpha=0.2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_title("Performance Radar Chart", fontsize=14, weight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.savefig("figures/2_model_comparison_radar_chart.png", dpi=300)
    plt.close()
    print("Saved: figures/2_model_comparison_radar_chart.png")

# ================================
# CONFUSION MATRICES
# ================================
def plot_confusion_matrices():
    print("🔍 Generating confusion matrices...")

    texts, labels = load_custom_dataset()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, stratify=labels, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8)
    }

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=3000)

    for model_name, model in models.items():
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=["Safe", "Vulnerable"])
        disp.plot(cmap="Blues", colorbar=False)
        plt.title(f"{model_name} - Confusion Matrix", fontsize=12, weight='bold')
        plt.tight_layout()
        plt.savefig(f"figures/3_{model_name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=300)
        plt.close()

    print("Saved confusion matrices in 'figures/'")

# ================================
# ROC CURVES
# ================================
def plot_roc_curves():
    print("🔍 Generating ROC curves...")
    texts, labels = load_custom_dataset()
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, stratify=labels, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=200),
        "Decision Tree": DecisionTreeClassifier(max_depth=5),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=8)
    }

    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=3000)

    plt.figure(figsize=(8, 6))
    for name, model in models.items():
        pipeline = Pipeline([("tfidf", vectorizer), ("clf", model)])
        pipeline.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc_val = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.2f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve Comparison", fontsize=14, weight='bold')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/4_roc_curve_comparison.png", dpi=300)
    plt.close()
    print("Saved: figures/4_roc_curve_comparison.png")

# ================================
# 5️⃣ METRIC HEATMAP
# ================================
def plot_heatmap():
    df = pd.read_csv("model_comparison_results_pipeline_cv.csv")
    df.set_index("Model", inplace=True)

    plt.figure(figsize=(7, 5))
    sns.heatmap(df.iloc[:, 1:], annot=True, cmap="viridis", fmt=".2f")
    plt.title("Performance Metric Heatmap", fontsize=14, weight='bold')
    plt.tight_layout()
    plt.savefig("figures/5_model_performance_heatmap.png", dpi=300)
    plt.close()
    print("Saved: figures/5_model_performance_heatmap.png")

# ================================
# Run All Visuals
# ================================
if __name__ == "__main__":
    print("Generating full visualization suite...")
    plot_model_comparison()
    plot_radar_chart()
    plot_confusion_matrices()
    plot_roc_curves()
    plot_heatmap()
    print("All visualizations saved in 'figures/' folder.")
