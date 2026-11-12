"""
Stage 4: Production Pandas Data Processing Pipeline
====================================================
Customer Feedback Intelligence System

This module provides end-to-end data processing for customer feedback CSVs:
  1. Accepts CSV files with minimal columns (e.g. just `review_id` and `review_text`).
  2. Auto-detects the feedback/review text column dynamically.
  3. Data Validation & Cleaning (handles nulls, empty strings, duplicates).
  4. Batch AI Processing (coordinates Sentiment, Emotion, Aspects & Intent).
  5. Preserves original columns (`review_id`, `review_text`) exactly as given.
  6. Exports enriched tabular datasets with AI metadata columns for dashboard consumption.
"""

import os
import torch
# Enable multi-core parallelization across CPU cores
torch.set_num_threads(min(8, os.cpu_count() or 4))
# Use local cached models without unnecessary remote network checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from typing import Optional, List, Dict, Any
import pandas as pd

from inference_engine import CustomerIntelligenceEngine
from zero_shot_classifier import FeedbackAspectClassifier
from intelligence import (
    CustomerIntelligenceEngine as CustomerBI,
    ProductIntelligenceEngine as ProductBI,
    OperationalIntelligenceEngine as OperationalBI,
    MarketIntelligenceEngine as MarketBI,
    StrategicIntelligenceEngine as StrategicBI
)

class FeedbackDataPipeline:
    """
    Robust data pipeline that ingests raw customer review CSV files,
    cleans the data, applies multi-model AI enrichment, and exports
    the enriched dataset while preserving all original columns.
    """
    def __init__(self, device: int = -1):
        print("=" * 70)
        print("INITIALIZING CUSTOMER FEEDBACK DATA PIPELINE")
        print("=" * 70)
        # Load and reuse our cached inference engines from Stages 2 & 3
        self.sentiment_emotion_engine = CustomerIntelligenceEngine(device=device)
        self.aspect_classifier = FeedbackAspectClassifier(device=device)
        
        # Load Category A - E Business Intelligence engines
        self.customer_bi = CustomerBI()
        self.product_bi = ProductBI()
        self.operational_bi = OperationalBI()
        self.market_bi = MarketBI()
        self.strategic_bi = StrategicBI()
        print("Data Pipeline & 5 Intelligence Engines ready for batch processing!\n")

    def auto_detect_text_column(self, df: pd.DataFrame) -> str:
        """
        Intelligently identifies the review text column by searching for common
        names, or fallback to the column with the highest average string length.
        """
        candidate_names = [
            "review_text", "review", "text", "feedback", "comment", 
            "comments", "description", "customer_feedback", "body", "message"
        ]
        
        # Check for matching column names (case-insensitive)
        for col in df.columns:
            if str(col).lower().strip() in candidate_names:
                return col
                
        # Fallback: check all object/string columns and pick the one with longest avg character count
        string_cols = df.select_dtypes(include=["object", "string"]).columns
        if len(string_cols) == 0:
            raise ValueError("No string or text columns found in the CSV!")
            
        longest_col = max(string_cols, key=lambda c: df[c].astype(str).str.len().mean())
        return longest_col

    def clean_dataframe(
        self, 
        df: pd.DataFrame, 
        text_column: str,
        drop_duplicates: bool = True
    ) -> pd.DataFrame:
        """
        Cleans messy customer review data:
        - Drops NaN / Null reviews.
        - Strips whitespace and filters out empty strings.
        - Optionally removes duplicate reviews to avoid redundant computation.
        """
        initial_count = len(df)
        clean_df = df.copy()

        # 1. Drop rows where text_column is completely NaN
        clean_df = clean_df.dropna(subset=[text_column])

        # 2. Strip whitespace and filter out empty / short garbage strings (< 3 chars)
        clean_df[text_column] = clean_df[text_column].astype(str).str.strip()
        clean_df = clean_df[clean_df[text_column].str.len() >= 3]

        # 3. Handle duplicates
        if drop_duplicates:
            clean_df = clean_df.drop_duplicates(subset=[text_column])

        dropped_count = initial_count - len(clean_df)
        print(f"Data Cleaning Summary:")
        print(f"  • Initial records:     {initial_count}")
        print(f"  • Valid records kept:  {len(clean_df)}")
        print(f"  • Messy/Duplicate dropped: {dropped_count}")

        return clean_df.reset_index(drop=True)

    def process_dataframe(
        self,
        df: pd.DataFrame,
        output_csv_path: Optional[str] = None,
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        In-memory pipeline execution on an existing DataFrame:
        Cleaning -> Batch AI Inference -> Department Routing -> Escalation -> Export (optional).
        """
        # Detect or validate text column
        target_col = text_column or self.auto_detect_text_column(df)
        print(f"[1/3] Selected Review Text Column: '{target_col}'")

        # Clean data
        clean_df = self.clean_dataframe(df, target_col)
        reviews_list = clean_df[target_col].tolist()

        # Batch AI Processing (Optimized multi-threaded vectorized batch execution)
        print(f"[2/3] Running Optimized Batched AI Analysis on {len(reviews_list)} reviews...")
        
        # Step A: Batched Sentiment, Emotion & Toxicity
        ai_sentiment_results = self.sentiment_emotion_engine.analyze_batch(reviews_list, batch_size=16)
        sentiments = [r["sentiment"] for r in ai_sentiment_results]

        # Step B: Batched Zero-Shot Aspects & Intent
        ai_aspect_results = self.aspect_classifier.analyze_batch(reviews_list, sentiments=sentiments, batch_size=16)
        print(f"   All {len(reviews_list)} rows successfully analyzed in high-speed batch!")

        # Append new AI columns while preserving all original columns (review_id, review_text)
        clean_df["ai_sentiment"] = [r["sentiment"] for r in ai_sentiment_results]
        clean_df["ai_sentiment_confidence"] = [r["sentiment_confidence"] for r in ai_sentiment_results]
        clean_df["ai_emotion"] = [r["emotion"] for r in ai_sentiment_results]
        clean_df["ai_emotion_confidence"] = [r["emotion_confidence"] for r in ai_sentiment_results]
        clean_df["ai_primary_aspect"] = [r["primary_aspect"] for r in ai_aspect_results]
        clean_df["ai_detected_aspects"] = [", ".join(r["detected_aspects"]) for r in ai_aspect_results]
        clean_df["ai_primary_intent"] = [r["primary_intent"] for r in ai_aspect_results]
        clean_df["is_urgent"] = [r["is_urgent"] for r in ai_sentiment_results]

        # Department routing & severity escalation columns
        routing_info = self.operational_bi.route_departments(clean_df)
        clean_df["assigned_department"] = routing_info["assigned_series"]

        escalation_info = self.operational_bi.detect_urgent_escalations(clean_df)
        sev_map = {e["review_id"]: e["severity"] for e in escalation_info["urgent_queue"]}
        clean_df["escalation_severity"] = [sev_map.get(row.get("review_id"), "LOW") for _, row in clean_df.iterrows()]

        # Export enriched CSV if path provided
        if output_csv_path:
            print(f"[3/3] Exporting Enriched Dataset to: {output_csv_path}")
            clean_df.to_csv(output_csv_path, index=False)
            print("Pipeline execution completed successfully!\n")

        return clean_df

    def process_csv(
        self, 
        input_csv_path: str, 
        output_csv_path: Optional[str] = None,
        text_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Full pipeline execution: Ingestion -> Cleaning -> Batch AI Inference -> Export.
        """
        if not os.path.exists(input_csv_path):
            raise FileNotFoundError(f"Input file not found: {input_csv_path}")

        print(f"\n[1/4] Reading CSV: {input_csv_path}")
        df = pd.read_csv(input_csv_path)
        out_path = output_csv_path or "enriched_customer_feedback.csv"
        return self.process_dataframe(df, output_csv_path=out_path, text_column=text_column)

    def generate_complete_intelligence(self, enriched_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Runs all 5 Business Intelligence engines across the enriched dataset.
        Returns a structured master dictionary ready for Streamlit dashboard rendering.
        """
        cust_intel = self.customer_bi.generate_full_customer_intelligence(enriched_df)
        prod_intel = self.product_bi.generate_full_product_intelligence(enriched_df)
        ops_intel = self.operational_bi.generate_full_operational_intelligence(enriched_df)
        mkt_intel = self.market_bi.generate_full_market_intelligence(enriched_df)
        strat_intel = self.strategic_bi.generate_full_strategic_intelligence(
            enriched_df, ops_intel, prod_intel
        )

        return {
            "customer_intelligence": cust_intel,
            "product_intelligence": prod_intel,
            "operational_intelligence": ops_intel,
            "market_intelligence": mkt_intel,
            "strategic_intelligence": strat_intel
        }

if __name__ == "__main__":
    input_file = "sample_customer_reviews.csv"
    output_file = "enriched_customer_feedback.csv"

    pipeline = FeedbackDataPipeline()
    enriched_df = pipeline.process_csv(input_file, output_file)

    print("=" * 80)
    print("ENRICHED DATASET PREVIEW (First 5 Rows):")
    print("=" * 80)
    preview_cols = [
        "review_id", "review_text", 
        "ai_sentiment", "ai_emotion", "ai_primary_aspect", "ai_primary_intent", "is_urgent"
    ]
    print(enriched_df[preview_cols].head(5).to_string(index=False))

    print("\n" + "=" * 80)
    print("BUSINESS AGGREGATION SUMMARY:")
    print("=" * 80)
    print(f"Total Enriched Feedback: {len(enriched_df)}")
    print("\nSentiment Distribution:")
    print(enriched_df["ai_sentiment"].value_counts().to_string())
    print("\nTop Discussed Aspects:")
    print(enriched_df["ai_primary_aspect"].value_counts().to_string())
    print("\nCustomer Intents:")
    print(enriched_df["ai_primary_intent"].value_counts().to_string())
    print(f"\nUrgent Action Required (Alerts): {enriched_df['is_urgent'].sum()}")
    print("=" * 80)
