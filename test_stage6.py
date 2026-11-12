"""
Stage 6 Verification Test: Customer & Product Intelligence
"""
import pandas as pd
from intelligence.customer_intelligence import CustomerIntelligenceEngine
from intelligence.product_intelligence import ProductIntelligenceEngine

def test_stage_6():
    df = pd.read_csv("enriched_customer_feedback.csv")
    print(f"Loaded {len(df)} rows from enriched_customer_feedback.csv\n")

    print("=" * 70)
    print("TESTING CATEGORY A: CUSTOMER INTELLIGENCE")
    print("=" * 70)
    cust_engine = CustomerIntelligenceEngine()
    cust_intel = cust_engine.generate_full_customer_intelligence(df)
    
    print(f"Net Sentiment Score (NSS): {cust_intel['sentiment_overview']['net_sentiment_score']} "
          f"({cust_intel['sentiment_overview']['positive_percentage']}% Pos vs {cust_intel['sentiment_overview']['negative_percentage']}% Neg)")
    
    print("\nEmotion Action Segments:")
    for seg, data in cust_intel['emotion_analysis']['action_segments'].items():
        print(f"  • {seg}: {data['count']} reviews ({data['percentage']}%) -> {data['action']}")

    print(f"\nChurn Risk Signals:")
    print(f"  • High Risk: {cust_intel['churn_risk_detection']['summary']['high_risk_count']}")
    print(f"  • Medium Risk: {cust_intel['churn_risk_detection']['summary']['medium_risk_count']}")
    if cust_intel['churn_risk_detection']['high_risk_evidence']:
        print(f"  • Sample High-Risk Review: \"{cust_intel['churn_risk_detection']['high_risk_evidence'][0]['text']}\"")

    print(f"\nBrand Advocates & Loyalty:")
    print(f"  • Total Advocates Detected: {cust_intel['loyalty_and_advocacy']['advocate_count']}")

    print("\n" + "=" * 70)
    print("TESTING CATEGORY B: PRODUCT INTELLIGENCE")
    print("=" * 70)
    prod_engine = ProductIntelligenceEngine()
    prod_intel = prod_engine.generate_full_product_intelligence(df)

    print("Root Cause Aspect Breakdown:")
    for asp in prod_intel['root_cause_diagnosis']['aspect_breakdown']:
        print(f"  • {asp['aspect']:<30}: {asp['status']:<8} ({asp['negative_percentage']}% Negative | {asp['total_mentions']} mentions)")

    if prod_intel['root_cause_diagnosis']['root_cause_hypothesis']:
        print(f"\nRoot Cause Hypothesis:")
        print(f"  • {prod_intel['root_cause_diagnosis']['root_cause_hypothesis']['hypothesis']}")

    print(f"\nFeature Requests Detected: {prod_intel['feature_requests']['total_feature_requests']}")
    for req in prod_intel['feature_requests']['feature_request_evidence']:
        print(f"  • [{req['subtype']}]: \"{req['text']}\"")

    print(f"\nPrice Sensitivity Perception: {prod_intel['price_sensitivity']['overall_value_perception']}")
    print(f"  • Complaints: {prod_intel['price_sensitivity']['expensive_complaint_count']} | Praise: {prod_intel['price_sensitivity']['good_value_praise_count']}")
    print("=" * 70)
    print("STAGE 6 VERIFICATION COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    test_stage_6()

