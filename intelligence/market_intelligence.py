"""
Market Intelligence Module (Category D)
=======================================
AI-Powered Customer & Business Intelligence Platform

Implements:
  D1. Competitor Intelligence & Comparative Mentions
  D2. Emerging Trend & Issue Velocity Detection
  D3. Market Positioning & Customer Expectations
"""

import re
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

class MarketIntelligenceEngine:
    """
    Analyzes competitor mentions, emerging complaint velocity,
    and market positioning signals from customer reviews.
    """
    def __init__(self):
        # Comparative competitor phrasing patterns
        self.competitor_patterns = [
            r"\b(switch(ed|ing)?\s+(from|to)\s+[A-Za-z0-9_-]+)\b",
            r"\b(better\s+than\s+[A-Za-z0-9_-]+)\b",
            r"\b(worse\s+than\s+[A-Za-z0-9_-]+)\b",
            r"\b(competitor|alternative|rival|other\s+brand)\b"
        ]

        # Common expectation patterns
        self.expectation_patterns = [
            r"\b(expect(ed|ing)?|supposed\s+to|advertised\s+as|promised)\b",
            r"\b(standard|norm|benchmark)\b"
        ]

    def extract_competitor_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        D1. Identifies competitor mentions and extracts comparative advantages vs weaknesses.
        """
        competitor_mentions = []
        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rev_id = row.get("review_id", "N/A")
            sentiment = str(row.get("ai_sentiment", "neutral")).lower()

            matched = [p for p in self.competitor_patterns if re.search(p, text, re.IGNORECASE)]
            if matched:
                is_advantage = sentiment == "positive" or any(w in text.lower() for w in ["better than", "switched from"])
                competitor_mentions.append({
                    "review_id": rev_id,
                    "text": text,
                    "sentiment": sentiment,
                    "comparison_type": "Perceived Brand Advantage" if is_advantage else "Perceived Competitive Weakness"
                })

        advantages = [m for m in competitor_mentions if m["comparison_type"] == "Perceived Brand Advantage"]
        weaknesses = [m for m in competitor_mentions if m["comparison_type"] == "Perceived Competitive Weakness"]

        return {
            "total_competitor_mentions": len(competitor_mentions),
            "brand_advantages_count": len(advantages),
            "competitive_weaknesses_count": len(weaknesses),
            "evidence": competitor_mentions[:5],
            "data_limitation_notice": (
                "Competitor intelligence reflects subjective customer comparisons inside review text. "
                "Verified competitor benchmarking requires external competitor review datasets."
            )
        }

    def detect_emerging_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        D2. Compares review frequency and complaint volume over time to identify emerging issues.
        """
        date_col = None
        for col in ["review_date", "date", "created_at", "timestamp"]:
            if col in df.columns:
                date_col = col
                break

        if not date_col:
            return {
                "available": False,
                "message": "Date column not found. Trend velocity analysis requires review timestamps."
            }

        df_copy = df.copy()
        try:
            df_copy["parsed_date"] = pd.to_datetime(df_copy[date_col], errors="coerce")
            df_copy = df_copy.dropna(subset=["parsed_date"]).sort_values("parsed_date")
        except Exception:
            return {"available": False, "message": "Could not parse date column into valid datetime objects."}

        total_dated = len(df_copy)
        if total_dated < 4:
            return {
                "available": False,
                "message": f"Sample size too small ({total_dated} dated reviews) for reliable statistical trend detection."
            }

        # Split into earlier half vs recent half to detect velocity
        halfway = total_dated // 2
        earlier_df = df_copy.iloc[:halfway]
        recent_df = df_copy.iloc[halfway:]

        earlier_neg_pct = round(((earlier_df["ai_sentiment"] == "negative").sum() / len(earlier_df)) * 100, 1)
        recent_neg_pct = round(((recent_df["ai_sentiment"] == "negative").sum() / len(recent_df)) * 100, 1)
        velocity_delta = round(recent_neg_pct - earlier_neg_pct, 1)

        trend_status = "DETERIORATING" if velocity_delta >= 10.0 else ("IMPROVING" if velocity_delta <= -10.0 else "STABLE")

        return {
            "available": True,
            "earlier_period_negative_rate": f"{earlier_neg_pct}%",
            "recent_period_negative_rate": f"{recent_neg_pct}%",
            "negative_velocity_change": f"{velocity_delta:+0.1f}%",
            "trend_status": trend_status,
            "sample_size_flag": "Adequate" if total_dated >= 30 else f"Small Sample ({total_dated} reviews) - Interpret as observation, not definitive forecast.",
            "earliest_date": str(df_copy["parsed_date"].min().date()),
            "latest_date": str(df_copy["parsed_date"].max().date())
        }

    def analyze_market_positioning(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        D3. Evaluates customer expectations against delivered experience.
        """
        unmet_expectations = []
        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rev_id = row.get("review_id", "N/A")
            sentiment = str(row.get("ai_sentiment", "")).lower()

            if any(re.search(p, text, re.IGNORECASE) for p in self.expectation_patterns) and sentiment == "negative":
                unmet_expectations.append({
                    "review_id": rev_id,
                    "text": text,
                    "gap": "Customer cited an unmet explicit product promise or expectation."
                })

        return {
            "unmet_expectations_count": len(unmet_expectations),
            "unmet_expectations_evidence": unmet_expectations[:3],
            "positioning_recommendation": (
                "Review marketing copy and product onboarding promises to ensure customer "
                "expectations align accurately with current operational capabilities."
            )
        }

    def generate_full_market_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compiles complete Category D intelligence payload.
        """
        return {
            "competitor_intelligence": self.extract_competitor_intelligence(df),
            "emerging_trends": self.detect_emerging_trends(df),
            "market_positioning": self.analyze_market_positioning(df)
        }

