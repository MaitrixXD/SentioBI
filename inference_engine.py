"""
Stage 2: Multi-Model Inference Engine
=====================================
Customer Feedback Intelligence System

This module coordinates multiple specialized pretrained Hugging Face models:
  1. Sentiment Analysis: cardiffnlp/twitter-roberta-base-sentiment-latest
     (Negative, Neutral, Positive)
  2. Emotion Detection: j-hartmann/emotion-english-distilroberta-base
     (Anger, Disgust, Fear, Joy, Neutral, Sadness, Surprise)
  3. Toxicity & Escalation: martin-ha/toxic-comment-model
     (Non-toxic, Toxic - used to flag urgent/abusive customer tickets)

Purpose:
  Provides a clean, unified API (`analyze_feedback()` and `analyze_batch()`)
  that converts raw customer text into structured business intelligence.
"""

from typing import Dict, List, Any
from transformers import pipeline

class CustomerIntelligenceEngine:
    """
    Unified inference engine coordinating sentiment, emotion, 
    and urgency/toxicity detection models.
    """
    def __init__(self, device: int = -1):
        """
        Initializes and caches the specialized Hugging Face pipelines.
        device: -1 for CPU, 0 for GPU / MPS where available.
        """
        print("Initializing Customer Intelligence Engine...")
        
        # 1. Sentiment Model (3-class: negative, neutral, positive)
        self.sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        print(f"Loading Sentiment Pipeline ({self.sentiment_model_name})...")
        self.sentiment_pipe = pipeline(
            "sentiment-analysis",
            model=self.sentiment_model_name,
            device=device,
            top_k=None  # Returns probabilities for all classes (for distribution inspection)
        )
        
        # 2. Emotion Model (7-class Ekman emotions + neutral)
        self.emotion_model_name = "j-hartmann/emotion-english-distilroberta-base"
        print(f"Loading Emotion Pipeline ({self.emotion_model_name})...")
        self.emotion_pipe = pipeline(
            "text-classification",
            model=self.emotion_model_name,
            device=device,
            top_k=None
        )
        
        # 3. Toxicity Model (Flags aggressive/toxic complaints for escalation)
        self.toxicity_model_name = "martin-ha/toxic-comment-model"
        print(f"Loading Toxicity Pipeline ({self.toxicity_model_name})...")
        self.toxicity_pipe = pipeline(
            "text-classification",
            model=self.toxicity_model_name,
            device=device,
            top_k=None
        )
        print("All models successfully loaded and ready for inference!\n")

    def analyze_feedback(self, text: str) -> Dict[str, Any]:
        """
        Performs multi-dimensional analysis on a single customer review.
        """
        if not text or not str(text).strip():
            return {
                "text": text,
                "sentiment": "neutral",
                "sentiment_confidence": 0.0,
                "emotion": "neutral",
                "emotion_confidence": 0.0,
                "is_toxic": False,
                "toxicity_confidence": 0.0,
                "is_urgent": False,
                "all_sentiment_scores": {},
                "all_emotion_scores": {}
            }

        # Run inference across models
        sentiment_res = self.sentiment_pipe(text)[0]
        emotion_res = self.emotion_pipe(text)[0]
        toxicity_res = self.toxicity_pipe(text)[0]

        # Extract top sentiment & dictionary of all scores
        sentiment_scores = {item["label"].lower(): round(item["score"], 4) for item in sentiment_res}
        top_sentiment = max(sentiment_scores, key=sentiment_scores.get)
        top_sentiment_score = sentiment_scores[top_sentiment]

        # Extract top emotion & dictionary of all scores
        emotion_scores = {item["label"].lower(): round(item["score"], 4) for item in emotion_res}
        top_emotion = max(emotion_scores, key=emotion_scores.get)
        top_emotion_score = emotion_scores[top_emotion]

        # Extract toxicity
        toxic_score = next((item["score"] for item in toxicity_res if item["label"].lower() in ["toxic", "label_1"]), 0.0)
        is_toxic = toxic_score > 0.5

        # Business Logic: Urgent Escalation Flag
        # Flag if review is toxic OR contains strong anger (> 0.70)
        is_urgent = is_toxic or (top_emotion == "anger" and top_emotion_score >= 0.70)

        return {
            "text": text,
            "sentiment": top_sentiment,
            "sentiment_confidence": top_sentiment_score,
            "emotion": top_emotion,
            "emotion_confidence": top_emotion_score,
            "is_toxic": is_toxic,
            "toxicity_confidence": round(toxic_score, 4),
            "is_urgent": is_urgent,
            "all_sentiment_scores": sentiment_scores,
            "all_emotion_scores": emotion_scores
        }

    def analyze_batch(self, texts: List[str], batch_size: int = 16) -> List[Dict[str, Any]]:
        """
        Analyzes a batch of customer reviews using vectorized pipeline batches.
        Dramatically faster on multi-core CPUs.
        """
        if not texts:
            return []

        # Vectorized Batched Pipelines with truncation
        sent_res_list = self.sentiment_pipe(texts, batch_size=batch_size, truncation=True)
        emotion_res_list = self.emotion_pipe(texts, batch_size=batch_size, truncation=True)
        toxicity_res_list = self.toxicity_pipe(texts, batch_size=batch_size, truncation=True)

        results = []
        for text, sent_res, emotion_res, toxicity_res in zip(texts, sent_res_list, emotion_res_list, toxicity_res_list):
            sentiment_scores = {item["label"].lower(): round(item["score"], 4) for item in sent_res}
            top_sentiment = max(sentiment_scores, key=sentiment_scores.get)
            top_sentiment_score = sentiment_scores[top_sentiment]

            emotion_scores = {item["label"].lower(): round(item["score"], 4) for item in emotion_res}
            top_emotion = max(emotion_scores, key=emotion_scores.get)
            top_emotion_score = emotion_scores[top_emotion]

            toxic_score = next((item["score"] for item in toxicity_res if item["label"].lower() in ["toxic", "label_1"]), 0.0)
            is_toxic = toxic_score > 0.5
            is_urgent = is_toxic or (top_emotion == "anger" and top_emotion_score >= 0.70)

            results.append({
                "text": text,
                "sentiment": top_sentiment,
                "sentiment_confidence": top_sentiment_score,
                "emotion": top_emotion,
                "emotion_confidence": top_emotion_score,
                "is_toxic": is_toxic,
                "toxicity_confidence": round(toxic_score, 4),
                "is_urgent": is_urgent,
                "all_sentiment_scores": sentiment_scores,
                "all_emotion_scores": emotion_scores
            })
        return results

if __name__ == "__main__":
    # Test suite demonstrating diverse customer feedback scenarios
    engine = CustomerIntelligenceEngine()

    test_reviews = [
        "The onboarding experience was seamless, but customer support took 48 hours to respond.",
        "Your new pricing is an absolute scam! You charged my card twice and your support reps hung up on me!",
        "The software works fine. It does what it says on the box, nothing more, nothing less.",
        "I absolutely adore the intuitive UI update! It cut our team's reporting time in half. Kudos!"
    ]

    print("=" * 80)
    print("STAGE 2 TEST RUN: MULTI-MODEL CUSTOMER FEEDBACK INTELLIGENCE")
    print("=" * 80)

    results = engine.analyze_batch(test_reviews)

    for i, res in enumerate(results, 1):
        print(f"\n[Review #{i}]: \"{res['text']}\"")
        print(f"  • Sentiment: {res['sentiment'].upper()} (Confidence: {res['sentiment_confidence']*100:.1f}%)")
        print(f"  • Emotion:   {res['emotion'].upper()} (Confidence: {res['emotion_confidence']*100:.1f}%)")
        print(f"  • Toxic:     {'YES (High Alert)' if res['is_toxic'] else 'No'} ({res['toxicity_confidence']*100:.1f}%)")
        print(f"  • Urgent Escalation Flag: {'[!] ACTION REQUIRED' if res['is_urgent'] else 'Normal Workflow'}")
        print(f"  • Sentiment Distribution: {res['all_sentiment_scores']}")
        print(f"  • Top 3 Emotions: {sorted(res['all_emotion_scores'].items(), key=lambda x: x[1], reverse=True)[:3]}")
    print("=" * 80)

