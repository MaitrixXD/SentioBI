"""
Stage 3: Zero-Shot Multi-Category Engine (Aspects & Intent)
===========================================================
Customer Feedback Intelligence System

This module uses Natural Language Inference (NLI) to perform zero-shot
classification on customer reviews without needing labeled training data.

Capabilities:
  1. Aspect-Based Classification (Multilabel):
     Identifies which business areas are discussed (Pricing, Support, Delivery, Quality, UI).
  2. Customer Intent Discovery (Multilabel / Multiclass):
     Identifies the customer's goal (Complaint, Praise, Suggestion, Inquiry, Refund).
  3. Aspect-Level Sentiment Pairing:
     Pairs identified aspects with the review's overall sentiment.

Model:
  valhalla/distilbart-mnli-12-3 (DistilBART fine-tuned on MNLI)
  Fast, lightweight zero-shot classification via premise-hypothesis entailment.
"""

from typing import List, Dict, Any, Optional
from transformers import pipeline

class FeedbackAspectClassifier:
    """
    Zero-shot classifier that categorizes customer reviews into 
    business aspects and customer intents on the fly.
    """
    def __init__(self, model_name: str = "valhalla/distilbart-mnli-12-3", device: int = -1):
        print(f"Loading Zero-Shot Classification Model ({model_name})...")
        self.model_name = model_name
        self.classifier = pipeline(
            "zero-shot-classification",
            model=self.model_name,
            device=device
        )
        print("Zero-Shot Classification Model ready!\n")

        # Standard business aspect taxonomy
        self.default_aspects = [
            "Customer Support",
            "Pricing & Billing",
            "Delivery & Shipping",
            "Product Quality & Reliability",
            "User Interface & Ease of Use"
        ]

        # Standard customer intent taxonomy
        self.default_intents = [
            "Complaint",
            "Praise & Appreciation",
            "Feature Suggestion",
            "Refund Request",
            "General Inquiry"
        ]

    def classify_aspects(
        self, 
        text: str, 
        candidate_aspects: Optional[List[str]] = None, 
        threshold: float = 0.50
    ) -> Dict[str, Any]:
        """
        Multilabel classification: A review can discuss multiple aspects simultaneously.
        """
        if not text or not str(text).strip():
            return {"primary_aspect": "General", "detected_aspects": [], "all_aspect_scores": {}}

        aspects_to_test = candidate_aspects or self.default_aspects
        
        # In NLI zero-shot, the hypothesis template helps the model understand the context
        template = "This customer review discusses {}."
        
        # multi_label=True evaluates each label independently via Sigmoid (not Softmax)
        result = self.classifier(
            text,
            candidate_labels=aspects_to_test,
            hypothesis_template=template,
            multi_label=True
        )

        all_scores = {label: round(score, 4) for label, score in zip(result["labels"], result["scores"])}
        
        # Filter aspects that pass the confidence threshold
        detected = [label for label, score in all_scores.items() if score >= threshold]
        
        # The primary aspect is the highest confidence label, or 'General' if none pass threshold
        primary = result["labels"][0] if result["scores"][0] >= 0.35 else "General Feedback"

        return {
            "primary_aspect": primary,
            "detected_aspects": detected if detected else [primary],
            "all_aspect_scores": all_scores
        }

    def classify_intent(
        self, 
        text: str, 
        candidate_intents: Optional[List[str]] = None,
        threshold: float = 0.45
    ) -> Dict[str, Any]:
        """
        Detects customer intents (e.g., complaint, refund request, feature suggestion).
        """
        if not text or not str(text).strip():
            return {"primary_intent": "General", "detected_intents": [], "all_intent_scores": {}}

        intents_to_test = candidate_intents or self.default_intents
        template = "The intent of this customer is to make a {}."

        result = self.classifier(
            text,
            candidate_labels=intents_to_test,
            hypothesis_template=template,
            multi_label=True
        )

        all_scores = {label: round(score, 4) for label, score in zip(result["labels"], result["scores"])}
        detected = [label for label, score in all_scores.items() if score >= threshold]
        primary = result["labels"][0]

        return {
            "primary_intent": primary,
            "detected_intents": detected if detected else [primary],
            "all_intent_scores": all_scores
        }

    def analyze_batch(
        self, 
        texts: List[str], 
        sentiments: Optional[List[str]] = None, 
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Batch-optimized Zero-Shot classification for aspects & customer intents.
        Leverages vectorized tensor evaluation across multiple CPU cores.
        """
        if not texts:
            return []

        if sentiments is None:
            sentiments = ["neutral"] * len(texts)

        # Batch Aspect Evaluation
        aspect_results = self.classifier(
            texts,
            candidate_labels=self.default_aspects,
            hypothesis_template="This customer review discusses {}.",
            multi_label=True,
            batch_size=batch_size,
            truncation=True
        )

        # Batch Intent Evaluation
        intent_results = self.classifier(
            texts,
            candidate_labels=self.default_intents,
            hypothesis_template="The intent of this customer is to make a {}.",
            multi_label=True,
            batch_size=batch_size,
            truncation=True
        )

        outputs = []
        for text, sent, a_res, i_res in zip(texts, sentiments, aspect_results, intent_results):
            a_scores = {label: round(score, 4) for label, score in zip(a_res["labels"], a_res["scores"])}
            detected_a = [label for label, score in a_scores.items() if score >= 0.50]
            primary_a = a_res["labels"][0] if a_res["scores"][0] >= 0.35 else "General Feedback"

            i_scores = {label: round(score, 4) for label, score in zip(i_res["labels"], i_res["scores"])}
            detected_i = [label for label, score in i_scores.items() if score >= 0.45]
            primary_i = i_res["labels"][0]

            outputs.append({
                "text": text,
                "sentiment": sent,
                "primary_aspect": primary_a,
                "detected_aspects": detected_a if detected_a else [primary_a],
                "all_aspect_scores": a_scores,
                "primary_intent": primary_i,
                "detected_intents": detected_i if detected_i else [primary_i],
                "all_intent_scores": i_scores,
                "aspect_sentiment_pairs": [
                    {"aspect": asp, "sentiment": sent}
                    for asp in (detected_a if detected_a else [primary_a])
                ]
            })
        return outputs

    def analyze_review_categories(self, text: str, sentiment: str = "neutral") -> Dict[str, Any]:
        """
        Combines aspect classification, intent discovery, and sentiment pairing.
        """
        aspect_info = self.classify_aspects(text)
        intent_info = self.classify_intent(text)

        # Aspect-Sentiment Pairing: links the overall sentiment to the detected aspects
        aspect_sentiment_pairs = [
            {"aspect": aspect, "sentiment": sentiment}
            for aspect in aspect_info["detected_aspects"]
        ]

        return {
            "text": text,
            "sentiment": sentiment,
            "primary_aspect": aspect_info["primary_aspect"],
            "detected_aspects": aspect_info["detected_aspects"],
            "all_aspect_scores": aspect_info["all_aspect_scores"],
            "primary_intent": intent_info["primary_intent"],
            "detected_intents": intent_info["detected_intents"],
            "all_intent_scores": intent_info["all_intent_scores"],
            "aspect_sentiment_pairs": aspect_sentiment_pairs
        }

if __name__ == "__main__":
    classifier = FeedbackAspectClassifier()

    test_reviews = [
        {
            "text": "The onboarding was seamless, but your customer support took 48 hours to reply to my ticket.",
            "sentiment": "positive"
        },
        {
            "text": "Your new pricing plans are absurd! I want a full refund immediately.",
            "sentiment": "negative"
        },
        {
            "text": "Could you please add dark mode support and keyboard shortcuts for power users?",
            "sentiment": "neutral"
        },
        {
            "text": "The delivery arrived 3 days late, the box was damaged, and the product was completely scratched.",
            "sentiment": "negative"
        }
    ]

    print("=" * 80)
    print("STAGE 3 TEST RUN: ZERO-SHOT ASPECT & INTENT INTELLIGENCE")
    print("=" * 80)

    for i, item in enumerate(test_reviews, 1):
        res = classifier.analyze_review_categories(item["text"], sentiment=item["sentiment"])
        print(f"\n[Review #{i}]: \"{res['text']}\"")
        print(f"  • Sentiment:         {res['sentiment'].upper()}")
        print(f"  • Primary Aspect:    {res['primary_aspect']}")
        print(f"  • Detected Aspects:  {res['detected_aspects']}")
        print(f"  • Top Aspect Scores: {dict(list(res['all_aspect_scores'].items())[:3])}")
        print(f"  • Primary Intent:    {res['primary_intent']}")
        print(f"  • Detected Intents:  {res['detected_intents']}")
        print(f"  • Top Intent Scores: {dict(list(res['all_intent_scores'].items())[:3])}")
        print(f"  • Aspect-Sentiment Pairs: {res['aspect_sentiment_pairs']}")
    print("=" * 80)

