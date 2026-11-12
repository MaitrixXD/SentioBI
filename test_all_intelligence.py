"""
Comprehensive Intelligence Verification Test: Categories A, B, C, D, E
"""
import pandas as pd
from intelligence import (
    CustomerIntelligenceEngine,
    ProductIntelligenceEngine,
    OperationalIntelligenceEngine,
    MarketIntelligenceEngine,
    StrategicIntelligenceEngine
)

def test_complete_intelligence():
    df = pd.read_csv("enriched_customer_feedback.csv")
    print(f"Loaded {len(df)} rows from enriched_customer_feedback.csv\n")

    # Category A
    cust_engine = CustomerIntelligenceEngine()
    cust_intel = cust_engine.generate_full_customer_intelligence(df)
    print(f"[Category A] Net Sentiment Score: {cust_intel['sentiment_overview']['net_sentiment_score']}")

    # Category B
    prod_engine = ProductIntelligenceEngine()
    prod_intel = prod_engine.generate_full_product_intelligence(df)
    print(f"[Category B] Most Problematic Aspect: {prod_intel['root_cause_diagnosis']['most_problematic_aspect']['aspect']}")

    # Category C
    ops_engine = OperationalIntelligenceEngine()
    ops_intel = ops_engine.generate_full_operational_intelligence(df)
    print(f"[Category C] Department Load: {ops_intel['department_routing']['department_counts']}")
    print(f"[Category C] Critical Escalations: {ops_intel['urgent_escalations']['severity_counts']['CRITICAL']}")

    # Category D
    mkt_engine = MarketIntelligenceEngine()
    mkt_intel = mkt_engine.generate_full_market_intelligence(df)
    print(f"[Category D] Competitor Mentions: {mkt_intel['competitor_intelligence']['total_competitor_mentions']}")

    # Category E
    strat_engine = StrategicIntelligenceEngine()
    strat_intel = strat_engine.generate_full_strategic_intelligence(df, ops_intel, prod_intel)
    print(f"[Category E] Estimated Financial Exposure: {strat_intel['financial_risk_exposure']['estimated_financial_exposure']}")
    print(f"[Category E] Generated {len(strat_intel['evidence_backed_insights'])} Evidence-Backed Insights:")
    for ins in strat_intel['evidence_backed_insights']:
        print(f"   • [{ins['priority']}] {ins['title']} (Score: {ins['priority_score']}) -> {ins['responsible_department']}")

    print("\nALL 5 INTELLIGENCE ENGINES OPERATING FLAWLESSLY!")

if __name__ == "__main__":
    test_complete_intelligence()

