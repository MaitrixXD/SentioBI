"""
Customer Intelligence Module (Category A)
=========================================
AI-Powered Customer & Business Intelligence Platform

Implements:
  A1. Sentiment Analysis & Net Sentiment Score (NSS)
  A2. Emotion Analysis & Customer Segmentation
  A3. Customer Churn Risk Signal Detection
  A4. Customer Loyalty & Advocacy Analysis
  A5. Customer Satisfaction Drivers
"""

import re
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

class CustomerIntelligenceEngine:
    """
    Analyzes customer sentiment, emotional pulse, churn-risk signals,
    brand advocacy, and key satisfaction drivers from enriched review data.
    """
    def __init__(self):
        # Churn signal patterns
        self.high_churn_patterns = [
            r"\b(cancel(l?ed|ing)?\s+(my\s+)?(subscription|account|service|order))\b",
            r"\b(switch(ed|ing)?\s+to\s+[a-z0-9]+)\b",
            r"\b(never\s+(buy(ing)?|us(e|ing)|order(ing)?)\s+again)\b",
            r"\b(last\s+time\s+i\s+(buy|order|use))\b",
            r"\b(taking\s+my\s+business\s+elsewhere)\b",
            r"\b(done\s+with\s+(this|your|you))\b",
            r"\b(fraud|scam|lawyer|sue|illegal)\b"
        ]
        self.med_churn_patterns = [
            r"\b(disappoint(ed|ing)?|frustrat(ed|ing)?)\b",
            r"\b(waste\s+of\s+(money|time))\b",
            r"\b(unacceptable|terrible|horrible|awful)\b",
            r"\b(refund|money\s+back)\b",
            r"\b(unresolved|no\s+response|ignored)\b"
        ]
        
        # Advocacy patterns
        self.advocacy_patterns = [
            r"\b(high(ly)?\s+recommend)\b",
            r"\b(best\s+[a-z0-9]+\s+(i('ve)?\s+ever|on\s+the\s+market))\b",
            r"\b(customer\s+for\s+life)\b",
            r"\b(repeat\s+(buyer|customer|order))\b",
            r"\b(tell(ing)?\s+(all\s+my\s+friends|everyone))\b",
            r"\b(love\s+(this|your)\s+[a-z0-9]+)\b",
            r"\b(game\s+changer|kudos|exceptional|flawless)\b"
        ]

    def analyze_sentiment_overview(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        A1. Computes sentiment distribution, percentages, and Net Sentiment Score (NSS).
        NSS = (% Positive Reviews) - (% Negative Reviews) [-100 to +100]
        """
        total = len(df)
        if total == 0:
            return {"total": 0, "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0, "nss": 0}

        counts = df["ai_sentiment"].str.lower().value_counts().to_dict()
        pos = counts.get("positive", 0)
        neg = counts.get("negative", 0)
        neu = counts.get("neutral", 0)

        pos_pct = round((pos / total) * 100, 1)
        neg_pct = round((neg / total) * 100, 1)
        neu_pct = round((neu / total) * 100, 1)
        nss = round(pos_pct - neg_pct, 1)

        return {
            "total_reviews": total,
            "positive_count": pos,
            "negative_count": neg,
            "neutral_count": neu,
            "positive_percentage": pos_pct,
            "negative_percentage": neg_pct,
            "neutral_percentage": neu_pct,
            "net_sentiment_score": nss
        }

    def analyze_emotion_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        A2. Analyzes granular emotional pulse and generates empathetic action guidance.
        """
        if len(df) == 0 or "ai_emotion" not in df.columns:
            return {"distribution": {}, "actionable_groups": {}}

        dist = df["ai_emotion"].str.lower().value_counts().to_dict()
        total = len(df)
        dist_pct = {k: round((v / total) * 100, 1) for k, v in dist.items()}

        # Segment by business reaction needed
        angry_count = dist.get("anger", 0)
        sad_fear_count = dist.get("sadness", 0) + dist.get("fear", 0)
        joy_count = dist.get("joy", 0)

        return {
            "emotion_counts": dist,
            "emotion_percentages": dist_pct,
            "action_segments": {
                "priority_escalation_angry": {
                    "count": angry_count,
                    "percentage": round((angry_count / total) * 100, 1),
                    "action": "Immediate de-escalation by senior support; investigate systemic failures."
                },
                "empathetic_reassurance_sad_fear": {
                    "count": sad_fear_count,
                    "percentage": round((sad_fear_count / total) * 100, 1),
                    "action": "Warm, empathetic communication resolving customer anxiety and disappointment."
                },
                "advocacy_testimonials_joy": {
                    "count": joy_count,
                    "percentage": round((joy_count / total) * 100, 1),
                    "action": "Engage for public reviews, referral bonuses, and marketing case studies."
                }
            }
        }

    def detect_churn_risk_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        A3. Identifies reviews indicating customer attrition risk.
        Note: Flags review-inferred signals, not subscription-data predictions.
        """
        high_risk_reviews = []
        med_risk_reviews = []

        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rev_id = row.get("review_id", "N/A")
            sentiment = row.get("ai_sentiment", "neutral")
            emotion = row.get("ai_emotion", "neutral")

            # Check high churn patterns
            matched_high = [p for p in self.high_churn_patterns if re.search(p, text, re.IGNORECASE)]
            if matched_high or (sentiment == "negative" and emotion in ["anger", "disgust"] and row.get("is_urgent", False)):
                high_risk_reviews.append({
                    "review_id": rev_id,
                    "text": text,
                    "signal_level": "HIGH",
                    "reason": "Expressed cancellation intent, switching threats, or severe anger/toxicity."
                })
                continue

            # Check medium churn patterns
            matched_med = [p for p in self.med_churn_patterns if re.search(p, text, re.IGNORECASE)]
            if matched_med or sentiment == "negative":
                med_risk_reviews.append({
                    "review_id": rev_id,
                    "text": text,
                    "signal_level": "MEDIUM",
                    "reason": "Repeated frustration, unresolved complaint, or refund demand."
                })

        total = len(df)
        high_cnt = len(high_risk_reviews)
        med_cnt = len(med_risk_reviews)
        low_cnt = max(0, total - (high_cnt + med_cnt))

        return {
            "summary": {
                "high_risk_count": high_cnt,
                "medium_risk_count": med_cnt,
                "low_risk_count": low_cnt,
                "high_risk_percentage": round((high_cnt / total) * 100, 1) if total > 0 else 0
            },
            "high_risk_evidence": high_risk_reviews[:5],
            "recommended_retention_actions": [
                "Proactively contact customers in the HIGH risk tier with personalized resolution offers.",
                "Fast-track pending refund and replacement tickets before customers trigger chargebacks.",
                "Review SLA bottlenecks on repeated complaint categories."
            ],
            "data_limitation_notice": (
                "These risk levels represent sentiment and keyword signals extracted from review text. "
                "Verified predictive churn models require historical subscription, login, and transaction data."
            )
        }

    def analyze_loyalty_and_advocacy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        A4. Identifies loyal customers and organic advocacy mentions.
        """
        advocates = []
        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rev_id = row.get("review_id", "N/A")
            sentiment = row.get("ai_sentiment", "neutral")
            emotion = row.get("ai_emotion", "neutral")

            matched = [p for p in self.advocacy_patterns if re.search(p, text, re.IGNORECASE)]
            if matched or (sentiment == "positive" and emotion == "joy" and row.get("ai_sentiment_confidence", 0) > 0.90):
                advocates.append({
                    "review_id": rev_id,
                    "text": text,
                    "advocacy_strength": "STRONG" if len(matched) > 0 else "MODERATE"
                })

        total = len(df)
        return {
            "advocate_count": len(advocates),
            "advocacy_percentage": round((len(advocates) / total) * 100, 1) if total > 0 else 0,
            "advocate_evidence": advocates[:5],
            "opportunities": [
                "Invite identified strong advocates to leave Trustpilot / App Store reviews.",
                "Feature enthusiastic quotes in marketing testimonials and case studies.",
                "Offer loyalty referral discount codes to encourage word-of-mouth growth."
            ],
            "data_limitation_notice": (
                "Advocacy signals reflect public praise in customer reviews and should not be confused "
                "with a formally calculated Net Promoter Score (NPS) survey."
            )
        }

    def analyze_satisfaction_drivers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        A5. Determines which business aspects drive the highest positive sentiment.
        """
        if "ai_primary_aspect" not in df.columns:
            return {"top_drivers": [], "satisfaction_gaps": []}

        aspect_groups = df.groupby("ai_primary_aspect")
        drivers = []
        gaps = []

        for aspect, group in aspect_groups:
            total_aspect = len(group)
            pos_count = (group["ai_sentiment"].str.lower() == "positive").sum()
            neg_count = (group["ai_sentiment"].str.lower() == "negative").sum()
            pos_rate = round((pos_count / total_aspect) * 100, 1) if total_aspect > 0 else 0

            entry = {
                "aspect": aspect,
                "total_mentions": total_aspect,
                "positive_mentions": int(pos_count),
                "negative_mentions": int(neg_count),
                "satisfaction_rate": pos_rate
            }

            if pos_rate >= 60.0:
                drivers.append(entry)
            else:
                gaps.append(entry)

        # Sort drivers by satisfaction rate
        drivers.sort(key=lambda x: x["satisfaction_rate"], reverse=True)
        gaps.sort(key=lambda x: x["satisfaction_rate"])

        return {
            "top_satisfaction_drivers": drivers,
            "aspects_needing_improvement": gaps,
            "causation_notice": (
                "Observed satisfaction drivers indicate strong correlation with positive customer reviews. "
                "They represent customer priorities rather than mathematically proven isolated causation."
            )
        }

    def generate_full_customer_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs complete Category A suite and compiles unified intelligence payload.
        """
        return {
            "sentiment_overview": self.analyze_sentiment_overview(df),
            "emotion_analysis": self.analyze_emotion_distribution(df),
            "churn_risk_detection": self.detect_churn_risk_signals(df),
            "loyalty_and_advocacy": self.analyze_loyalty_and_advocacy(df),
            "satisfaction_drivers": self.analyze_satisfaction_drivers(df)
        }

