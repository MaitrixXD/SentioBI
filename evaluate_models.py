"""
Stage 5: Model Evaluation & Performance Validation
===================================================
Customer Feedback Intelligence System

This module tests our AI models against human-verified ground-truth data:
  1. Computes genuine Accuracy, Precision, Recall, and F1-scores.
  2. Generates Confusion Matrices (text grid and interactive Plotly heatmap).
  3. Conducts Failure Mode Analysis (examines where and why the AI made mistakes).
  4. Never fabricates accuracy; reports exact mathematical performance.
"""

import os
# Use local cached models without unnecessary remote network checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support, 
    classification_report, 
    confusion_matrix
)
import plotly.figure_factory as ff

from inference_engine import CustomerIntelligenceEngine
from zero_shot_classifier import FeedbackAspectClassifier

class ModelEvaluator:
    """
    Evaluates customer feedback classification models against labeled ground truth.
    """
    def __init__(self, ground_truth_csv: str = "ground_truth_test_data.csv"):
        if not os.path.exists(ground_truth_csv):
            raise FileNotFoundError(f"Ground truth dataset not found: {ground_truth_csv}")
            
        print("=" * 70)
        print("STAGE 5: MODEL EVALUATION & PERFORMANCE BENCHMARKING")
        print("=" * 70)
        self.df = pd.read_csv(ground_truth_csv)
        print(f"Loaded {len(self.df)} human-labeled benchmark reviews.")
        
        # Load our cached engines
        self.engine = CustomerIntelligenceEngine()
        self.aspect_classifier = FeedbackAspectClassifier()

    def run_evaluation(self, output_html_chart: str = "confusion_matrix_sentiment.html"):
        """
        Runs model inference across the test set and calculates scikit-learn metrics.
        """
        print("\n[1/3] Generating AI predictions on ground truth reviews...")
        predicted_sentiments = []
        sentiment_confidences = []
        predicted_intents = []

        for idx, row in self.df.iterrows():
            text = row["review_text"]
            
            # Predict Sentiment
            sent_res = self.engine.analyze_feedback(text)
            predicted_sentiments.append(sent_res["sentiment"].lower())
            sentiment_confidences.append(sent_res["sentiment_confidence"])
            
            # Predict Intent
            intent_res = self.aspect_classifier.classify_intent(text)
            predicted_intents.append(intent_res["primary_intent"])

        # Attach predictions to dataframe for error analysis
        self.df["pred_sentiment"] = predicted_sentiments
        self.df["pred_sentiment_confidence"] = sentiment_confidences
        self.df["pred_intent"] = predicted_intents

        # -------------------------------------------------------------
        # 1. SENTIMENT CLASSIFICATION EVALUATION
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("SENTIMENT CLASSIFICATION BENCHMARK REPORT")
        print("=" * 70)

        true_sent = self.df["true_sentiment"].str.lower().tolist()
        pred_sent = self.df["pred_sentiment"].tolist()
        labels = ["negative", "neutral", "positive"]

        acc = accuracy_score(true_sent, pred_sent)
        print(f"Overall Accuracy: {acc * 100:.2f}%\n")

        print("Detailed Classification Report (Precision, Recall, F1-Score):")
        report = classification_report(true_sent, pred_sent, labels=labels, zero_division=0)
        print(report)

        # Compute Confusion Matrix
        cm = confusion_matrix(true_sent, pred_sent, labels=labels)
        print("Confusion Matrix Grid:")
        print(f"{'':>12} | Pred: Negative | Pred: Neutral | Pred: Positive |")
        print("-" * 60)
        for i, label in enumerate(labels):
            print(f"True: {label:>7} | {cm[i][0]:>14} | {cm[i][1]:>13} | {cm[i][2]:>14} |")

        # -------------------------------------------------------------
        # 2. INTERACTIVE PLOTLY CONFUSION MATRIX
        # -------------------------------------------------------------
        print(f"\n[2/3] Generating Interactive Confusion Matrix Chart -> {output_html_chart}...")
        x_labels = [f"Pred: {l.capitalize()}" for l in labels]
        y_labels = [f"True: {l.capitalize()}" for l in labels]
        
        import plotly.express as px
        fig = px.imshow(
            cm,
            labels=dict(x="Predicted Label (AI)", y="True Label (Human Ground Truth)", color="Count"),
            x=x_labels,
            y=y_labels,
            text_auto=True,
            color_continuous_scale="Blues",
            title="<b>Sentiment Model Evaluation — Confusion Matrix</b>"
        )
        fig.update_layout(
            template="plotly_white",
            width=650,
            height=500
        )
        fig.write_html(output_html_chart)

        # -------------------------------------------------------------
        # 3. FAILURE MODE & ERROR ANALYSIS
        # -------------------------------------------------------------
        print("\n" + "=" * 70)
        print("[3/3] FAILURE MODE ANALYSIS (Inspect Where the AI Made Mistakes)")
        print("=" * 70)

        mistakes = self.df[self.df["true_sentiment"] != self.df["pred_sentiment"]]
        
        if len(mistakes) == 0:
            print("Perfect score! Zero misclassifications on this test set.")
        else:
            print(f"Found {len(mistakes)} misclassified review(s):\n")
            for _, row in mistakes.iterrows():
                print(f"• Review ID: {row['review_id']}")
                print(f"  Text:      \"{row['review_text']}\"")
                print(f"  Human Truth: {row['true_sentiment'].upper()}")
                print(f"  AI Guess:    {row['pred_sentiment'].upper()} (Confidence: {row['pred_sentiment_confidence']*100:.1f}%)")
                print(f"  Analysis:    Notice how subtle linguistic cues (like sarcasm, mixed feedback, or")
                print(f"               compound clauses) can challenge a model. High confidence does NOT guarantee correctness!")
                print("-" * 70)

        return self.df

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    results_df = evaluator.run_evaluation()

