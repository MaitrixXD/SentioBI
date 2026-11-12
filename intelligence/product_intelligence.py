"""
Product Intelligence Module (Category B)
========================================
AI-Powered Customer & Business Intelligence Platform

Implements:
  B1. Root Cause Diagnosis (Aspect-Based Sentiment & Hypotheses)
  B2. Product & Category Benchmarking
  B3. Hidden Revenue & Upselling Opportunity Discovery
  B4. Feature Request & Innovation Taxonomy
  B5. Price Sensitivity & Value-for-Money Analysis
"""

import re
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

class ProductIntelligenceEngine:
    """
    Analyzes aspect-level root causes, category benchmarks, hidden revenue
    opportunities, feature innovation demand, and pricing perceptions.
    """
    def __init__(self):
        # Feature request keywords
        self.feature_request_patterns = [
            r"\b(add|introduce|support|include|implement|build)\s+([a-z0-9\s]+)\b",
            r"\b(would\s+(really\s+)?(like|love|appreciate)\s+if\s+you\s+(could\s+)?add)\b",
            r"\b(please\s+(add|support|bring\s+back))\b",
            r"\b(missing\s+(feature|option|capability|export))\b",
            r"\b(wish\s+(there\s+was|it\s+had))\b"
        ]
        
        # Upselling & monetization opportunity keywords
        self.revenue_opportunity_patterns = [
            r"\b(would\s+pay\s+(more|extra)\s+for)\b",
            r"\b(willing\s+to\s+pay\s+for)\b",
            r"\b(need\s+(more|extra|larger)\s+(storage|capacity|seats|licenses))\b",
            r"\b(enterprise|premium|pro\s+version|dedicated\s+support)\b",
            r"\b(integration\s+with\s+[a-z0-9]+)\b",
            r"\b(api\s+(access|support|rate\s+limit))\b"
        ]

        # Price perception keywords
        self.price_negative_patterns = [
            r"\b(too\s+expensive|overpriced|pricey|not\s+worth\s+(it|the\s+money))\b",
            r"\b(price\s+(hike|increase|gouging|scam))\b",
            r"\b(rip-?off|exorbitant|cost\s+too\s+much)\b",
            r"\b(poor\s+value|bad\s+deal)\b"
        ]
        self.price_positive_patterns = [
            r"\b(great\s+value|worth\s+(every\s+penny|it))\b",
            r"\b(affordable|reasonably\s+priced|inexpensive|bargain)\b",
            r"\b(good\s+price|fair\s+price)\b"
        ]

    def diagnose_root_causes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        B1. Evaluates aspect-level success vs. failure and generates root-cause hypotheses.
        """
        if "ai_primary_aspect" not in df.columns:
            return {"aspect_diagnosis": [], "critical_aspect_failure": None}

        aspect_diagnosis = []
        for aspect, group in df.groupby("ai_primary_aspect"):
            total = len(group)
            pos = int((group["ai_sentiment"].str.lower() == "positive").sum())
            neg = int((group["ai_sentiment"].str.lower() == "negative").sum())
            neu = int((group["ai_sentiment"].str.lower() == "neutral").sum())

            neg_rate = round((neg / total) * 100, 1) if total > 0 else 0
            pos_rate = round((pos / total) * 100, 1) if total > 0 else 0

            # Gather sample negative evidence
            neg_reviews = group[group["ai_sentiment"].str.lower() == "negative"]
            sample_evidence = neg_reviews["review_text"].head(3).tolist()

            aspect_diagnosis.append({
                "aspect": aspect,
                "total_mentions": total,
                "positive_count": pos,
                "negative_count": neg,
                "neutral_count": neu,
                "positive_percentage": pos_rate,
                "negative_percentage": neg_rate,
                "negative_evidence": sample_evidence,
                "status": "FAILING" if neg_rate >= 50.0 else ("AT RISK" if neg_rate >= 30.0 else "HEALTHY")
            })

        # Sort by negative percentage descending
        aspect_diagnosis.sort(key=lambda x: x["negative_percentage"], reverse=True)
        most_problematic = aspect_diagnosis[0] if len(aspect_diagnosis) > 0 and aspect_diagnosis[0]["negative_percentage"] > 30 else None

        # Formulate root-cause hypothesis
        hypothesis = None
        if most_problematic:
            hypothesis = {
                "aspect": most_problematic["aspect"],
                "failure_rate": f"{most_problematic['negative_percentage']}% negative",
                "hypothesis": f"Customer dissatisfaction is heavily concentrated in '{most_problematic['aspect']}'. "
                              f"Evidence indicates systemic operational or design issues rather than isolated user errors.",
                "evidence": most_problematic["negative_evidence"]
            }

        return {
            "aspect_breakdown": aspect_diagnosis,
            "most_problematic_aspect": most_problematic,
            "root_cause_hypothesis": hypothesis
        }

    def benchmark_products_and_categories(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        B2. Compares performance across products or categories if available.
        """
        # Look for category or product columns
        bench_col = None
        for col in ["product_category", "category", "product_id", "product_name", "product"]:
            if col in df.columns:
                bench_col = col
                break

        if not bench_col:
            return {
                "available": False,
                "message": "Product/Category benchmark column not present in dataset. Benchmarking unavailable."
            }

        benchmarks = []
        for cat, group in df.groupby(bench_col):
            total = len(group)
            pos = int((group["ai_sentiment"].str.lower() == "positive").sum())
            neg = int((group["ai_sentiment"].str.lower() == "negative").sum())
            pos_rate = round((pos / total) * 100, 1) if total > 0 else 0
            neg_rate = round((neg / total) * 100, 1) if total > 0 else 0

            benchmarks.append({
                "group_name": str(cat),
                "review_count": total,
                "positive_percentage": pos_rate,
                "negative_percentage": neg_rate,
                "primary_complaint_aspect": group[group["ai_sentiment"].str.lower() == "negative"]["ai_primary_aspect"].mode()[0] if neg > 0 else "None"
            })

        benchmarks.sort(key=lambda x: x["positive_percentage"], reverse=True)
        top_perf = benchmarks[0] if benchmarks else None
        bottom_perf = benchmarks[-1] if benchmarks else None

        return {
            "available": True,
            "benchmark_column": bench_col,
            "rankings": benchmarks,
            "top_performing_category": top_perf,
            "lowest_performing_category": bottom_perf
        }

    def discover_hidden_revenue_opportunities(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        B3. Identifies customer demand for premium features, expansions, or new paid tiers.
        """
        opportunities = []
        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rev_id = row.get("review_id", "N/A")

            matched = [p for p in self.revenue_opportunity_patterns if re.search(p, text, re.IGNORECASE)]
            if matched or (row.get("ai_primary_intent") == "Feature Suggestion" and "pro" in text.lower()):
                opportunities.append({
                    "review_id": rev_id,
                    "text": text,
                    "demand_signal": "Customer requested advanced / premium capability"
                })

        return {
            "opportunity_count": len(opportunities),
            "customer_demand_evidence": opportunities[:5],
            "monetization_validation_strategy": [
                "Conduct customer discovery interviews with users who requested premium features.",
                "Build a lightweight landing page or in-app poll to test willingness to pay before writing code.",
                "Explore an enterprise or add-on pricing tier rather than bundling into basic plans."
            ],
            "data_limitation_notice": (
                "Revenue opportunities are based on customer interest signals in review text. "
                "Actual revenue projections require pricing models and willingness-to-pay validation surveys."
            )
        }

    def analyze_feature_requests(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        B4. Groups and ranks customer innovation and feature requests.
        """
        feature_reviews = []
        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rev_id = row.get("review_id", "N/A")
            intent = str(row.get("ai_primary_intent", "")).lower()

            matched = [p for p in self.feature_request_patterns if re.search(p, text, re.IGNORECASE)]
            if matched or "feature suggestion" in intent:
                # Classify subtype
                subtype = "New Feature"
                if any(w in text.lower() for w in ["dark mode", "shortcut", "ui", "button", "layout"]):
                    subtype = "UI / Usability Improvement"
                elif any(w in text.lower() for w in ["api", "sync", "export", "connect", "integrate"]):
                    subtype = "Integration / Export"

                feature_reviews.append({
                    "review_id": rev_id,
                    "text": text,
                    "subtype": subtype
                })

        # Group by subtype
        subtypes_count = {}
        for r in feature_reviews:
            st = r["subtype"]
            subtypes_count[st] = subtypes_count.get(st, 0) + 1

        return {
            "total_feature_requests": len(feature_reviews),
            "requests_by_subtype": subtypes_count,
            "feature_request_evidence": feature_reviews[:5],
            "suggested_roadmap_action": (
                "Pass feature requests to Product Management for backlog prioritization. "
                "Group requests into sprint epics based on frequency."
            )
        }

    def analyze_price_sensitivity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        B5. Analyzes customer perceptions of pricing and value for money.
        """
        too_expensive = []
        good_value = []

        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rev_id = row.get("review_id", "N/A")
            aspect = str(row.get("ai_primary_aspect", ""))

            # Check negative pricing
            if any(re.search(p, text, re.IGNORECASE) for p in self.price_negative_patterns) or \
               ("pricing" in aspect.lower() and row.get("ai_sentiment") == "negative"):
                too_expensive.append({"review_id": rev_id, "text": text})

            # Check positive pricing
            elif any(re.search(p, text, re.IGNORECASE) for p in self.price_positive_patterns) or \
                 ("pricing" in aspect.lower() and row.get("ai_sentiment") == "positive"):
                good_value.append({"review_id": rev_id, "text": text})

        total_price_feedback = len(too_expensive) + len(good_value)
        value_perception = "FAVORABLE" if len(good_value) > len(too_expensive) else ("UNFAVORABLE" if len(too_expensive) > len(good_value) else "NEUTRAL / BALANCED")

        return {
            "total_price_mentions": total_price_feedback,
            "expensive_complaint_count": len(too_expensive),
            "good_value_praise_count": len(good_value),
            "overall_value_perception": value_perception,
            "pricing_objections_evidence": too_expensive[:3],
            "pricing_praise_evidence": good_value[:3],
            "recommendation": (
                "Do not lower list prices impulsively. Run controlled A/B price experiments or offer "
                "annual billing discounts if pricing complaints exceed 25% of negative feedback."
            )
        }

    def generate_full_product_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compiles complete Category B intelligence payload.
        """
        return {
            "root_cause_diagnosis": self.diagnose_root_causes(df),
            "product_category_benchmarks": self.benchmark_products_and_categories(df),
            "hidden_revenue_opportunities": self.discover_hidden_revenue_opportunities(df),
            "feature_requests": self.analyze_feature_requests(df),
            "price_sensitivity": self.analyze_price_sensitivity(df)
        }

