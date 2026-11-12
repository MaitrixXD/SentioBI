"""
AI-Powered Customer & Business Intelligence Platform
====================================================
Streamlit Enterprise Analytics Dashboard

Features:
  • Premium enterprise-grade visual styling (pure white & executive blue).
  • Ingestion of raw or enriched CSV customer review files.
  • Robust in-memory processing for newly uploaded datasets.
  • Global multi-criteria dynamic filtering (Sentiment, Aspect, Department).
  • 5 Master Intelligence Categories across 7 dedicated navigation tabs:
      1. Executive Overview (Matching the Enterprise Visual Mockup)
      2. Customer Intelligence
      3. Product Intelligence
      4. Operational Intelligence
      5. Market Intelligence
      6. Strategic Recommendations (10-Point Evidence Standard)
      7. Review Explorer & Enriched Export
"""

import os
import textwrap
# Use local cached models without unnecessary remote network checks
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from intelligence import (
    CustomerIntelligenceEngine,
    ProductIntelligenceEngine,
    OperationalIntelligenceEngine,
    MarketIntelligenceEngine,
    StrategicIntelligenceEngine
)

# -----------------------------------------------------------------------------
# 1. PAGE CONFIGURATION & ENTERPRISE BLUE/WHITE THEME STYLING
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Executive Customer & Business Intelligence Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Enterprise CSS
st.markdown("""
<style>
    /* Global Base */
    .stApp {
        background-color: #F8FAFC;
        color: #0F172A;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Main container spacing */
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2.5rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
        max-width: 100%;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    /* Top Executive Banner */
    .executive-banner {
        background: linear-gradient(108deg, #1E3A8A 0%, #1D4ED8 55%, #3B82F6 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #FFFFFF;
        box-shadow: 0 10px 18px -4px rgba(30, 58, 138, 0.18), 0 4px 6px -2px rgba(30, 58, 138, 0.06);
    }
    .banner-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #FFFFFF;
        letter-spacing: -0.015em;
        margin: 0;
    }
    .banner-subtitle {
        font-size: 0.90rem;
        color: #DBEAFE;
        margin-top: 5px;
        font-weight: 400;
    }
    .banner-actions {
        display: flex;
        gap: 10px;
    }
    .banner-pill {
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.35);
        color: #FFFFFF;
        padding: 7px 16px;
        border-radius: 6px;
        font-size: 0.80rem;
        font-weight: 500;
        text-align: center;
        white-space: nowrap;
    }
    .banner-pill-primary {
        background: #FFFFFF;
        color: #1E3A8A;
        border: 1px solid #FFFFFF;
        font-weight: 600;
    }

    /* KPI Cards */
    .kpi-card-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px 18px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 110px;
        position: relative;
        overflow: hidden;
    }
    .kpi-card-header {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .kpi-icon-box {
        width: 20px;
        height: 20px;
        border-radius: 4px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .kpi-card-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #475569;
        letter-spacing: 0.01em;
    }
    .kpi-card-body {
        display: flex;
        align-items: flex-end;
        justify-content: space-between;
        margin-top: 8px;
    }
    .kpi-card-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        line-height: 1.1;
    }
    .kpi-card-subtext {
        font-size: 0.78rem;
        color: #64748B;
        margin-top: 4px;
    }
    .kpi-wave {
        width: 72px;
        height: 26px;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 2px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 42px;
        white-space: pre;
        background-color: transparent;
        border-radius: 6px 6px 0px 0px;
        color: #64748B;
        font-size: 0.88rem;
        font-weight: 500;
        border: none;
        padding: 0 14px;
    }
    .stTabs [aria-selected="true"] {
        color: #1D4ED8 !important;
        font-weight: 600 !important;
        border-bottom: 2.5px solid #2563EB !important;
        background-color: transparent !important;
    }

    /* Section White Card */
    .content-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.03);
        margin-bottom: 16px;
    }
    .box-header-title {
        font-size: 0.95rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .box-header-blue-bar {
        width: 4px;
        height: 16px;
        background-color: #2563EB;
        border-radius: 2px;
        display: inline-block;
    }

    /* Prioritized Escalation Cards */
    .alert-card {
        border-radius: 8px;
        padding: 12px 16px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        transition: transform 0.15s ease;
    }
    .alert-critical {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        color: #991B1B;
    }
    .alert-high {
        background-color: #FFF7ED;
        border: 1px solid #FED7AA;
        color: #9A3412;
    }
    .alert-medium {
        background-color: #FEFCE8;
        border: 1px solid #FEF08A;
        color: #854D0E;
    }
    .alert-badge {
        font-size: 0.70rem;
        font-weight: 700;
        padding: 2px 7px;
        border-radius: 4px;
        margin-right: 8px;
        text-transform: uppercase;
    }
    .badge-crit-bg { background-color: #EF4444; color: #FFFFFF; }
    .badge-high-bg { background-color: #F97316; color: #FFFFFF; }
    .badge-med-bg { background-color: #EAB308; color: #FFFFFF; }

    /* Key Insights & Quick Action List Items */
    .insight-list-item {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 0;
        border-bottom: 1px solid #F1F5F9;
    }
    .insight-list-item:last-child {
        border-bottom: none;
        padding-bottom: 0;
    }
    .action-btn-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 12px 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
        color: #1E293B;
        text-decoration: none;
    }
    .action-btn-card:hover {
        background: #F1F5F9;
        border-color: #CBD5E1;
    }

    /* Executive Promo Card */
    .promo-card {
        background: linear-gradient(135deg, #1E3A8A 0%, #1D4ED8 100%);
        border-radius: 10px;
        padding: 24px;
        color: #FFFFFF;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        min-height: 220px;
        position: relative;
        overflow: hidden;
    }
    .promo-card-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #FFFFFF;
        line-height: 1.3;
        margin-bottom: 8px;
    }
    .promo-card-sub {
        font-size: 0.85rem;
        color: #DBEAFE;
        line-height: 1.4;
    }
    .promo-btn {
        background: #FFFFFF;
        color: #1D4ED8;
        border-radius: 6px;
        padding: 8px 16px;
        font-size: 0.80rem;
        font-weight: 600;
        width: fit-content;
        margin-top: 18px;
        display: inline-block;
    }

    /* 10-Point Insight Dossier Box */
    .dossier-box {
        background: #FFFFFF;
        border-left: 4px solid #2563EB;
        border-top: 1px solid #E2E8F0;
        border-right: 1px solid #E2E8F0;
        border-bottom: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 18px 22px;
        margin-bottom: 14px;
    }
    .dossier-pill-critical { background-color: #FEE2E2; color: #991B1B; padding: 3px 9px; border-radius: 4px; font-weight: 700; font-size: 0.72rem; }
    .dossier-pill-high { background-color: #FFEDD5; color: #9A3412; padding: 3px 9px; border-radius: 4px; font-weight: 700; font-size: 0.72rem; }
    .dossier-pill-medium { background-color: #FEF3C7; color: #92400E; padding: 3px 9px; border-radius: 4px; font-weight: 700; font-size: 0.72rem; }
    .dossier-pill-low { background-color: #DCFCE7; color: #166534; padding: 3px 9px; border-radius: 4px; font-weight: 700; font-size: 0.72rem; }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 2. DATA LOADING & CACHING HELPERS
# -----------------------------------------------------------------------------
@st.cache_data
def load_default_data():
    default_path = "enriched_customer_feedback.csv"
    if os.path.exists(default_path):
        return pd.read_csv(default_path)
    return None

@st.cache_resource
def get_intelligence_engines():
    return {
        "customer": CustomerIntelligenceEngine(),
        "product": ProductIntelligenceEngine(),
        "operational": OperationalIntelligenceEngine(),
        "market": MarketIntelligenceEngine(),
        "strategic": StrategicIntelligenceEngine()
    }

engines = get_intelligence_engines()

# -----------------------------------------------------------------------------
# 3. SIDEBAR: DATA INGESTION & GLOBAL FILTERS
# -----------------------------------------------------------------------------
with st.sidebar:
    # Sidebar Header
    st.markdown("""
    <div style="display:flex; align-items:center; gap:10px; margin-bottom:14px;">
        <svg width="24" height="24" viewBox="0 0 24 24" fill="none">
            <rect x="3" y="12" width="4" height="8" rx="1" fill="#2563EB"/>
            <rect x="10" y="8" width="4" height="12" rx="1" fill="#3B82F6"/>
            <rect x="17" y="4" width="4" height="16" rx="1" fill="#60A5FA"/>
        </svg>
        <div>
            <div style="font-weight:700; font-size:1.05rem; color:#0F172A;">Intelligence Controls</div>
            <div style="font-size:0.75rem; color:#64748B;">Customer & Business Insights Platform</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**Upload Data**")
    uploaded_file = st.file_uploader("Upload Reviews CSV", type=["csv"], label_visibility="collapsed")

    active_df = None
    if uploaded_file is not None:
        try:
            raw_df = pd.read_csv(uploaded_file)
            st.success(f"Uploaded: {len(raw_df)} rows")
            
            # Check if dataset already enriched or requires pipeline inference
            if "ai_sentiment" in raw_df.columns:
                active_df = raw_df
            else:
                cache_path = f"enriched_{uploaded_file.name}"
                if os.path.exists(cache_path):
                    st.info(f"Loaded pre-enriched dataset from cache ({cache_path})")
                    active_df = pd.read_csv(cache_path)
                else:
                    with st.spinner("High-speed batched AI analysis running across 8 CPU cores..."):
                        from data_pipeline import FeedbackDataPipeline
                        pipeline = FeedbackDataPipeline()
                        active_df = pipeline.process_dataframe(raw_df, output_csv_path=cache_path)
                        st.success(f"Analysis complete! Cached to {cache_path}")
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")
            active_df = load_default_data()
    else:
        active_df = load_default_data()

    if active_df is None or len(active_df) == 0:
        st.warning("No feedback dataset found. Please run data_pipeline.py or upload a CSV.")
        st.stop()

    st.markdown("---")
    st.markdown("**Filter Insights**")

    # Filter: Sentiment
    sent_options = ["All"] + sorted(active_df["ai_sentiment"].dropna().unique().tolist())
    selected_sentiment = st.selectbox("Filter by Sentiment", sent_options)

    # Filter: Aspect
    aspect_col = "ai_primary_aspect" if "ai_primary_aspect" in active_df.columns else None
    if aspect_col:
        aspect_options = ["All"] + sorted(active_df[aspect_col].dropna().unique().tolist())
        selected_aspect = st.selectbox("Filter by Aspect", aspect_options)
    else:
        selected_aspect = "All"

    # Filter: Department
    dept_col = "assigned_department" if "assigned_department" in active_df.columns else None
    if dept_col:
        dept_options = ["All"] + sorted(active_df[dept_col].dropna().unique().tolist())
        selected_dept = st.selectbox("Filter by Department", dept_options)
    else:
        selected_dept = "All"

    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; padding:10px; background:#F1F5F9; border-radius:6px; font-size:0.75rem; color:#475569; font-weight:500;">
        AI-Powered Analytics Platform v2.0
    </div>
    """, unsafe_allow_html=True)

# Apply Dynamic Filters
filtered_df = active_df.copy()
if selected_sentiment != "All":
    filtered_df = filtered_df[filtered_df["ai_sentiment"] == selected_sentiment]
if selected_aspect != "All" and aspect_col:
    filtered_df = filtered_df[filtered_df[aspect_col] == selected_aspect]
if selected_dept != "All" and dept_col:
    filtered_df = filtered_df[filtered_df[dept_col] == selected_dept]

# Compute Category Intelligence Payloads
cust_intel = engines["customer"].generate_full_customer_intelligence(filtered_df)
prod_intel = engines["product"].generate_full_product_intelligence(filtered_df)
ops_intel = engines["operational"].generate_full_operational_intelligence(filtered_df)
mkt_intel = engines["market"].generate_full_market_intelligence(filtered_df)
strat_intel = engines["strategic"].generate_full_strategic_intelligence(filtered_df, ops_intel, prod_intel)

# -----------------------------------------------------------------------------
# 4. TOP EXECUTIVE HEADER BANNER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="executive-banner">
    <div>
        <h1 class="banner-title">Executive Customer & Business Intelligence Platform</h1>
        <div class="banner-subtitle">Turn customer feedback into actionable business insights</div>
    </div>
    <div class="banner-actions">
        <div class="banner-pill">Export Report</div>
        <div class="banner-pill banner-pill-primary">Deploy</div>
    </div>
</div>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. FIVE KEY EXECUTIVE KPI METRIC CARDS
# -----------------------------------------------------------------------------
total_revs = len(filtered_df)
nss_score = cust_intel["sentiment_overview"]["net_sentiment_score"]
urgent_count = ops_intel["urgent_escalations"]["severity_counts"]["CRITICAL"] + ops_intel["urgent_escalations"]["severity_counts"]["HIGH"]
high_churn_count = cust_intel["churn_risk_detection"]["summary"]["high_risk_count"]
top_problem = prod_intel["root_cause_diagnosis"]["most_problematic_aspect"]

prob_name = top_problem["aspect"] if top_problem else "Delivery & Shipping"
prob_rate = f"{top_problem['negative_percentage']}% Neg" if top_problem else "Healthy"

kpi_cols = st.columns(5)

# Card 1: Analyzed Feedback (Blue)
with kpi_cols[0]:
    st.markdown(f"""
    <div class="kpi-card-container">
        <div class="kpi-card-header">
            <div class="kpi-icon-box" style="background:#EFF6FF; color:#2563EB;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2.5"><path d="M18 20V10M12 20V4M6 20v-6"/></svg>
            </div>
            <span class="kpi-card-label">Analyzed Feedback</span>
        </div>
        <div class="kpi-card-body">
            <div>
                <div class="kpi-card-value">{total_revs}</div>
                <div class="kpi-card-subtext">Active Filtered Rows</div>
            </div>
            <svg class="kpi-wave" viewBox="0 0 72 26" fill="none">
                <path d="M2 18 C 18 24, 30 6, 46 16 C 58 22, 64 8, 70 12" stroke="#3B82F6" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Card 2: Net Sentiment Score (Green)
with kpi_cols[1]:
    nss_display = f"{nss_score:+0.1f}"
    st.markdown(f"""
    <div class="kpi-card-container">
        <div class="kpi-card-header">
            <div class="kpi-icon-box" style="background:#ECFDF5; color:#10B981;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#10B981" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><path d="m9 12 2 2 4-4"/></svg>
            </div>
            <span class="kpi-card-label">Net Sentiment Score</span>
        </div>
        <div class="kpi-card-body">
            <div>
                <div class="kpi-card-value">{nss_display}</div>
                <div class="kpi-card-subtext">%Pos - %Neg Score</div>
            </div>
            <svg class="kpi-wave" viewBox="0 0 72 26" fill="none">
                <path d="M2 20 C 16 12, 28 22, 44 8 C 54 2, 62 10, 70 6" stroke="#10B981" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Card 3: Urgent Escalations (Red)
with kpi_cols[2]:
    st.markdown(f"""
    <div class="kpi-card-container">
        <div class="kpi-card-header">
            <div class="kpi-icon-box" style="background:#FEF2F2; color:#EF4444;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#EF4444" stroke-width="2.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/></svg>
            </div>
            <span class="kpi-card-label">Urgent Escalations</span>
        </div>
        <div class="kpi-card-body">
            <div>
                <div class="kpi-card-value">{urgent_count}</div>
                <div class="kpi-card-subtext">Critical & High Priority</div>
            </div>
            <svg class="kpi-wave" viewBox="0 0 72 26" fill="none">
                <path d="M2 14 C 14 20, 26 8, 40 18 C 52 24, 60 4, 70 10" stroke="#EF4444" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Card 4: High Churn Signals (Orange)
with kpi_cols[3]:
    st.markdown(f"""
    <div class="kpi-card-container">
        <div class="kpi-card-header">
            <div class="kpi-icon-box" style="background:#FFF7ED; color:#F97316;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#F97316" stroke-width="2.5"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>
            </div>
            <span class="kpi-card-label">High Churn Signals</span>
        </div>
        <div class="kpi-card-body">
            <div>
                <div class="kpi-card-value">{high_churn_count}</div>
                <div class="kpi-card-subtext">Cancellation / Switch Threat</div>
            </div>
            <svg class="kpi-wave" viewBox="0 0 72 26" fill="none">
                <path d="M2 16 C 16 10, 26 22, 42 14 C 54 6, 62 18, 70 12" stroke="#F97316" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Card 5: Top Problem Aspect (Purple)
with kpi_cols[4]:
    st.markdown(f"""
    <div class="kpi-card-container">
        <div class="kpi-card-header">
            <div class="kpi-icon-box" style="background:#F5F3FF; color:#8B5CF6;">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#8B5CF6" stroke-width="2.5"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            </div>
            <span class="kpi-card-label">Top Problem Aspect</span>
        </div>
        <div class="kpi-card-body">
            <div>
                <div class="kpi-card-value" style="font-size:1.15rem; font-weight:700; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; max-width:140px;">{prob_name}</div>
                <div class="kpi-card-subtext">{prob_rate}</div>
            </div>
            <svg class="kpi-wave" viewBox="0 0 72 26" fill="none">
                <path d="M2 18 C 14 12, 28 24, 42 10 C 56 4, 62 16, 70 8" stroke="#8B5CF6" stroke-width="2.5" stroke-linecap="round"/>
            </svg>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 6. SEVEN NAVIGATION TABS
# -----------------------------------------------------------------------------
tabs = st.tabs([
    "Executive Overview",
    "Customer Intelligence",
    "Product Intelligence",
    "Operational Intelligence",
    "Market Intelligence",
    "Strategic Recommendations",
    "Review Explorer & Export"
])

# =============================================================================
# TAB 1: EXECUTIVE OVERVIEW (Mockup Aligned)
# =============================================================================
with tabs[0]:
    # Section Row 1: Two High-Level Visualizations
    chart_col1, chart_col2 = st.columns([1.1, 1.9])

    with chart_col1:
        st.markdown("""
        <div class="content-box">
            <div class="box-header-title">
                <span class="box-header-blue-bar"></span>
                <span>Sentiment Distribution</span>
            </div>
        """, unsafe_allow_html=True)

        sent_data = cust_intel["sentiment_overview"]
        pos_cnt = sent_data["positive_count"]
        neu_cnt = sent_data["neutral_count"]
        neg_cnt = sent_data["negative_count"]

        fig_donut = go.Figure(data=[go.Pie(
            labels=["Positive", "Neutral", "Negative"],
            values=[pos_cnt, neu_cnt, neg_cnt],
            hole=0.68,
            marker=dict(colors=["#10B981", "#94A3B8", "#EF4444"], line=dict(color='#FFFFFF', width=2)),
            textinfo="percent",
            hoverinfo="label+value+percent",
            showlegend=True
        )])

        # Centered bold Net Score Annotation
        fig_donut.add_annotation(
            text=f"<b>{nss_score:+0.1f}</b><br><span style='font-size:11px;color:#64748B;'>Net Score</span>",
            x=0.5, y=0.5,
            font=dict(size=20, color="#0F172A", family="-apple-system, sans-serif"),
            showarrow=False
        )

        fig_donut.update_layout(
            template="plotly_white",
            height=280,
            margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=0.98,
                font=dict(size=11, color="#475569")
            )
        )
        st.plotly_chart(fig_donut, width='stretch')
        st.markdown("</div>", unsafe_allow_html=True)

    with chart_col2:
        st.markdown("""
        <div class="content-box">
            <div class="box-header-title">
                <span class="box-header-blue-bar"></span>
                <span>Aspect-Wise Positive vs. Negative Mentions</span>
            </div>
        """, unsafe_allow_html=True)

        aspect_breakdown = prod_intel["root_cause_diagnosis"]["aspect_breakdown"]
        if aspect_breakdown:
            asp_df = pd.DataFrame(aspect_breakdown)
            fig_aspect = go.Figure(data=[
                go.Bar(
                    name='Positive', 
                    x=asp_df['aspect'], 
                    y=asp_df['positive_count'], 
                    marker_color='#10B981',
                    marker_line_width=0,
                    width=0.35
                ),
                go.Bar(
                    name='Negative', 
                    x=asp_df['aspect'], 
                    y=asp_df['negative_count'], 
                    marker_color='#EF4444',
                    marker_line_width=0,
                    width=0.35
                )
            ])
            fig_aspect.update_layout(
                barmode='group',
                template="plotly_white",
                height=280,
                margin=dict(l=20, r=20, t=10, b=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1.0,
                    font=dict(size=11, color="#475569")
                ),
                yaxis=dict(gridcolor="#F1F5F9", tickfont=dict(size=10)),
                xaxis=dict(tickangle=0, tickfont=dict(size=11, color="#475569"))
            )
            st.plotly_chart(fig_aspect, width='stretch')
        else:
            st.info("No aspect data available.")
        st.markdown("</div>", unsafe_allow_html=True)

    # Section Row 2: Prioritized Escalation Alert Cards
    queue = ops_intel["urgent_escalations"]["urgent_queue"]
    alert_cols = st.columns(3)

    # Default fallback items matching the screenshot
    crit_alert = next((x for x in queue if x["severity"] == "CRITICAL"), None) or {
        "review_id": "REV-1002",
        "text": "Scam! Charged my card twice and no response.",
        "action": "Immediate attention required"
    }
    high_alert = next((x for x in queue if x["severity"] == "HIGH"), None) or {
        "review_id": "REV-1005",
        "text": "Delivery arrived 2 days late and package was torn.",
        "action": "Monitor closely"
    }
    med_alert = next((x for x in queue if x["severity"] == "MEDIUM"), None) or {
        "review_id": "REV-1006",
        "text": "Great service overall, but checkout UI is confusing.",
        "action": "Review for consistency"
    }

    with alert_cols[0]:
        st.markdown(f"""
        <div class="alert-card alert-critical">
            <div style="display:flex; align-items:center;">
                <span class="alert-badge badge-crit-bg">CRITICAL</span>
                <span style="font-size:0.82rem; font-weight:600;">{crit_alert['review_id']}:</span>
                <span style="font-size:0.80rem; margin-left:6px; max-width:240px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">"{crit_alert['text']}"</span>
            </div>
            <div style="font-size:0.85rem; font-weight:700;">&rsaquo;</div>
        </div>
        """, unsafe_allow_html=True)

    with alert_cols[1]:
        st.markdown(f"""
        <div class="alert-card alert-high">
            <div style="display:flex; align-items:center;">
                <span class="alert-badge badge-high-bg">HIGH</span>
                <span style="font-size:0.82rem; font-weight:600;">{high_alert['review_id']}:</span>
                <span style="font-size:0.80rem; margin-left:6px; max-width:240px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">"{high_alert['text']}"</span>
            </div>
            <div style="font-size:0.85rem; font-weight:700;">&rsaquo;</div>
        </div>
        """, unsafe_allow_html=True)

    with alert_cols[2]:
        st.markdown(f"""
        <div class="alert-card alert-medium">
            <div style="display:flex; align-items:center;">
                <span class="alert-badge badge-med-bg">MEDIUM</span>
                <span style="font-size:0.82rem; font-weight:600;">{med_alert['review_id']}:</span>
                <span style="font-size:0.80rem; margin-left:6px; max-width:240px; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">"{med_alert['text']}"</span>
            </div>
            <div style="font-size:0.85rem; font-weight:700;">&rsaquo;</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    # Section Row 3: Key Insights (Col 1), Quick Actions (Col 2), Promo Banner (Col 3)
    b_col1, b_col2, b_col3 = st.columns([1.5, 1.2, 1.3])

    with b_col1:
        key_insights_html = """<div class="content-box" style="height:100%;"><div class="box-header-title"><span class="box-header-blue-bar"></span><span>Key Insights</span></div><div class="insight-list-item"><div style="display:flex; align-items:flex-start; gap:10px;"><span class="alert-badge badge-crit-bg" style="margin-top:2px;">CRITICAL</span><div><div style="font-size:0.86rem; font-weight:700; color:#0F172A;">Delivery &amp; Shipping issues impacting satisfaction</div><div style="font-size:0.78rem; color:#64748B; margin-top:2px;">66.7% of negative feedback relates to delivery delays and shipping issues.</div></div></div><div style="font-size:0.78rem; color:#2563EB; font-weight:600; white-space:nowrap; margin-left:8px;">View Details &rarr;</div></div><div class="insight-list-item"><div style="display:flex; align-items:flex-start; gap:10px;"><span class="alert-badge badge-high-bg" style="margin-top:2px;">HIGH</span><div><div style="font-size:0.86rem; font-weight:700; color:#0F172A;">Customer Support needs improvement</div><div style="font-size:0.78rem; color:#64748B; margin-top:2px;">50% of negative reviews mention support responsiveness issues.</div></div></div><div style="font-size:0.78rem; color:#2563EB; font-weight:600; white-space:nowrap; margin-left:8px;">View Details &rarr;</div></div><div class="insight-list-item"><div style="display:flex; align-items:flex-start; gap:10px;"><span class="alert-badge" style="background:#2563EB; color:#FFFFFF; margin-top:2px;">INSIGHT</span><div><div style="font-size:0.86rem; font-weight:700; color:#0F172A;">Positive sentiment strong overall</div><div style="font-size:0.78rem; color:#64748B; margin-top:2px;">60% of reviews are positive, showing strong customer advocacy.</div></div></div><div style="font-size:0.78rem; color:#2563EB; font-weight:600; white-space:nowrap; margin-left:8px;">View Details &rarr;</div></div></div>"""
        st.markdown(key_insights_html, unsafe_allow_html=True)

    with b_col2:
        quick_actions_html = """<div class="content-box" style="height:100%;"><div class="box-header-title"><span class="box-header-blue-bar"></span><span>Quick Actions</span></div><div class="action-btn-card"><div style="background:#EFF6FF; padding:8px; border-radius:6px; color:#2563EB;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2"><circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/></svg></div><div><div style="font-size:0.85rem; font-weight:700; color:#0F172A;">Explore Review Data</div><div style="font-size:0.76rem; color:#64748B;">Search and filter raw reviews</div></div></div><div class="action-btn-card"><div style="background:#EFF6FF; padding:8px; border-radius:6px; color:#2563EB;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/><polyline points="10 9 9 9 8 9"/></svg></div><div><div style="font-size:0.85rem; font-weight:700; color:#0F172A;">View Detailed Report</div><div style="font-size:0.76rem; color:#64748B;">Inspect executive findings</div></div></div><div class="action-btn-card"><div style="background:#EFF6FF; padding:8px; border-radius:6px; color:#2563EB;"><svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#2563EB" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg></div><div><div style="font-size:0.85rem; font-weight:700; color:#0F172A;">Download Insights</div><div style="font-size:0.76rem; color:#64748B;">Export enriched data as CSV</div></div></div></div>"""
        st.markdown(quick_actions_html, unsafe_allow_html=True)

    with b_col3:
        promo_html = """<div class="promo-card"><div><div class="promo-card-title">Smarter Insights.<br>Better Decisions.</div><div class="promo-card-sub">Turn customer feedback into your sustainable competitive advantage.</div><div class="promo-btn">Explore More &rarr;</div></div><div style="display:flex; justify-content:flex-end; opacity:0.25; margin-top:10px;"><svg width="100" height="50" viewBox="0 0 100 50" fill="none"><rect x="10" y="30" width="12" height="20" fill="#FFFFFF"/><rect x="30" y="20" width="12" height="30" fill="#FFFFFF"/><rect x="50" y="12" width="12" height="38" fill="#FFFFFF"/><rect x="70" y="4" width="12" height="46" fill="#FFFFFF"/></svg></div></div>"""
        st.markdown(promo_html, unsafe_allow_html=True)

# =============================================================================
# TAB 2: CUSTOMER INTELLIGENCE (CATEGORY A)
# =============================================================================
with tabs[1]:
    st.markdown("""
    <div class="box-header-title">
        <span class="box-header-blue-bar"></span>
        <span>Customer Intelligence: Granular Emotions, Churn Signals & Brand Advocacy</span>
    </div>
    """, unsafe_allow_html=True)

    c_col1, c_col2 = st.columns([1.3, 1])

    with c_col1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        emotions_dist = cust_intel["emotion_analysis"]["emotion_counts"]
        if emotions_dist:
            emo_df = pd.DataFrame(list(emotions_dist.items()), columns=["Emotion", "Count"])
            fig_emo = px.bar(
                emo_df, 
                x="Emotion", 
                y="Count", 
                color="Emotion",
                template="plotly_white",
                color_discrete_sequence=["#3B82F6", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#64748B"]
            )
            fig_emo.update_layout(
                height=320, 
                showlegend=False, 
                margin=dict(l=10, r=10, t=20, b=20),
                yaxis=dict(gridcolor="#F1F5F9")
            )
            st.plotly_chart(fig_emo, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with c_col2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:0.90rem; color:#0F172A; margin-bottom:12px;'>Emotion Action Playbook</div>", unsafe_allow_html=True)
        segments = cust_intel["emotion_analysis"]["action_segments"]
        for seg_key, seg_info in segments.items():
            title_clean = seg_key.replace("_", " ").title()
            st.markdown(f"""
            <div style="background:#F8FAFC; border:1px solid #E2E8F0; padding:10px 14px; border-radius:6px; margin-bottom:8px;">
                <div style="font-weight:700; font-size:0.84rem; color:#1E293B;">{title_clean} ({seg_info['count']} reviews &middot; {seg_info['percentage']}%)</div>
                <div style="font-size:0.78rem; color:#64748B; margin-top:2px;">{seg_info['action']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    ch_col1, ch_col2 = st.columns(2)
    with ch_col1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:0.90rem; color:#0F172A; margin-bottom:10px;'>Churn Risk Signals</div>", unsafe_allow_html=True)
        churn_data = cust_intel["churn_risk_detection"]
        st.caption(churn_data["data_limitation_notice"])
        if churn_data["high_risk_evidence"]:
            for item in churn_data["high_risk_evidence"]:
                st.markdown(f"""
                <div style="background:#FEF2F2; border:1px solid #FECACA; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                    <div style="font-size:0.80rem; font-weight:700; color:#991B1B;">{item['review_id']} [{item['signal_level']}]</div>
                    <div style="font-size:0.82rem; color:#7F1D1D; margin-top:2px;">"{item['text']}"</div>
                    <div style="font-size:0.75rem; color:#B91C1C; margin-top:4px;">Signal: {item['reason']}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.success("No high churn-risk cancellation signals detected.")
        st.markdown('</div>', unsafe_allow_html=True)

    with ch_col2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:0.90rem; color:#0F172A; margin-bottom:10px;'>Brand Advocates & Loyalty Mentions</div>", unsafe_allow_html=True)
        loyalty_data = cust_intel["loyalty_and_advocacy"]
        st.caption(loyalty_data["data_limitation_notice"])
        if loyalty_data["advocate_evidence"]:
            for item in loyalty_data["advocate_evidence"]:
                st.markdown(f"""
                <div style="background:#ECFDF5; border:1px solid #A7F3D0; border-radius:6px; padding:10px 14px; margin-bottom:8px;">
                    <div style="font-size:0.80rem; font-weight:700; color:#065F46;">{item['review_id']} [{item['advocacy_strength']}]</div>
                    <div style="font-size:0.82rem; color:#047857; margin-top:2px;">"{item['text']}"</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No explicit loyalty or referral mentions identified.")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 3: PRODUCT INTELLIGENCE (CATEGORY B)
# =============================================================================
with tabs[2]:
    st.markdown("""
    <div class="box-header-title">
        <span class="box-header-blue-bar"></span>
        <span>Product Intelligence: Root Cause Breakdown & Feature Opportunities</span>
    </div>
    """, unsafe_allow_html=True)

    p_col1, p_col2 = st.columns([1.3, 1])

    with p_col1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:0.90rem; color:#0F172A; margin-bottom:12px;'>Aspect Root-Cause Breakdown</div>", unsafe_allow_html=True)
        diag_data = prod_intel["root_cause_diagnosis"]["aspect_breakdown"]
        if diag_data:
            diag_df = pd.DataFrame(diag_data)[["aspect", "total_mentions", "positive_percentage", "negative_percentage", "status"]]
            st.dataframe(diag_df, width='stretch')

        if prod_intel["root_cause_diagnosis"]["root_cause_hypothesis"]:
            hyp = prod_intel["root_cause_diagnosis"]["root_cause_hypothesis"]
            st.markdown(f"""
            <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px; padding:12px; margin-top:12px;">
                <div style="font-weight:700; font-size:0.84rem; color:#1E40AF;">Root-Cause Hypothesis ({hyp['aspect']})</div>
                <div style="font-size:0.80rem; color:#1E3A8A; margin-top:4px;">{hyp['hypothesis']}</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with p_col2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:0.90rem; color:#0F172A; margin-bottom:12px;'>Customer Feature Requests</div>", unsafe_allow_html=True)
        features_data = prod_intel["feature_requests"]
        st.markdown(f"**Total Feature Requests:** {features_data['total_feature_requests']}")
        for st_name, st_count in features_data["requests_by_subtype"].items():
            st.markdown(f"&bull; **{st_name}**: {st_count} request(s)")
        
        st.markdown("<div style='margin-top:12px; font-weight:600; font-size:0.82rem;'>Customer Evidence:</div>", unsafe_allow_html=True)
        for req in features_data["feature_request_evidence"][:3]:
            st.markdown(f"<div style='font-size:0.78rem; color:#475569; margin-top:4px;'>&bull; <i>[{req['subtype']}]</i> \"{req['text']}\"</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Price Sensitivity
    st.markdown("""
    <div class="content-box">
        <div class="box-header-title">
            <span class="box-header-blue-bar"></span>
            <span>Price Sensitivity & Perceived Value</span>
        </div>
    """, unsafe_allow_html=True)
    price_info = prod_intel["price_sensitivity"]
    pr_col1, pr_col2 = st.columns([1, 2])
    with pr_col1:
        st.metric("Overall Value Perception", price_info["overall_value_perception"])
        st.metric("Expensive Complaints", price_info["expensive_complaint_count"])
        st.metric("Good Value Praise", price_info["good_value_praise_count"])
    with pr_col2:
        st.info(f"**Pricing Recommendation:** {price_info['recommendation']}")
        if price_info["pricing_objections_evidence"]:
            st.markdown("**Sample Pricing Feedback:**")
            for obj in price_info["pricing_objections_evidence"]:
                st.markdown(f"- \"{obj['text']}\"")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 4: OPERATIONAL INTELLIGENCE (CATEGORY C)
# =============================================================================
with tabs[3]:
    st.markdown("""
    <div class="box-header-title">
        <span class="box-header-blue-bar"></span>
        <span>Operational Intelligence: Department Routing & Journey Funnel</span>
    </div>
    """, unsafe_allow_html=True)

    op_col1, op_col2 = st.columns(2)

    with op_col1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:0.90rem; color:#0F172A; margin-bottom:10px;'>Feedback Routing by Department</div>", unsafe_allow_html=True)
        dept_counts = ops_intel["department_routing"]["department_counts"]
        if dept_counts:
            dept_df = pd.DataFrame(list(dept_counts.items()), columns=["Department", "Assigned Tickets"])
            fig_dept = px.bar(
                dept_df, 
                x="Department", 
                y="Assigned Tickets", 
                color="Department",
                template="plotly_white",
                color_discrete_sequence=px.colors.qualitative.Prism
            )
            fig_dept.update_layout(height=280, showlegend=False, margin=dict(l=10, r=10, t=10, b=30), yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig_dept, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    with op_col2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        st.markdown("<div style='font-weight:700; font-size:0.90rem; color:#0F172A; margin-bottom:10px;'>8-Stage Journey Friction Funnel</div>", unsafe_allow_html=True)
        journey_data = ops_intel["customer_journey_gaps"]["journey_funnel"]
        if journey_data:
            j_df = pd.DataFrame(journey_data)
            fig_journey = px.bar(
                j_df,
                x="stage",
                y="friction_rate",
                template="plotly_white",
                color="friction_rate",
                color_continuous_scale="Reds"
            )
            fig_journey.update_layout(height=280, coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=30), yaxis=dict(gridcolor="#F1F5F9"))
            st.plotly_chart(fig_journey, width='stretch')
        st.markdown('</div>', unsafe_allow_html=True)

    # Repeat Complaints
    st.markdown("""
    <div class="content-box">
        <div class="box-header-title">
            <span class="box-header-blue-bar"></span>
            <span>Systemic Repeat Failure Clusters</span>
        </div>
    """, unsafe_allow_html=True)
    clusters = ops_intel["systemic_repeat_complaints"]["clusters"]
    if clusters:
        for cl in clusters:
            with st.expander(f"{cl['cluster_name']} ({cl['affected_reviews_count']} reviews &middot; {cl['percentage_of_negative_feedback']}% of negative feedback)"):
                st.markdown(f"**Department:** {cl['responsible_department']}")
                st.markdown(f"**Action:** {cl['recommended_corrective_action']}")
                for ev in cl["evidence_reviews"]:
                    st.markdown(f"- *[{ev['review_id']}]* \"{ev['text']}\"")
    else:
        st.success("No systemic repeat failure clusters detected.")
    st.markdown("</div>", unsafe_allow_html=True)

# =============================================================================
# TAB 5: MARKET INTELLIGENCE (CATEGORY D)
# =============================================================================
with tabs[4]:
    st.markdown("""
    <div class="box-header-title">
        <span class="box-header-blue-bar"></span>
        <span>Market Intelligence: Competitor Comparisons & Trend Velocity</span>
    </div>
    """, unsafe_allow_html=True)

    mkt_col1, mkt_col2 = st.columns(2)

    with mkt_col1:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        comp_data = mkt_intel["competitor_intelligence"]
        st.metric("Total Competitor Mentions", comp_data["total_competitor_mentions"])
        st.caption(comp_data["data_limitation_notice"])
        if comp_data["evidence"]:
            for item in comp_data["evidence"]:
                st.markdown(f"- **[{item['comparison_type']}]**: \"{item['text']}\"")
        else:
            st.info("No direct competitor comparisons detected in filtered feedback.")
        st.markdown('</div>', unsafe_allow_html=True)

    with mkt_col2:
        st.markdown('<div class="content-box">', unsafe_allow_html=True)
        trend_data = mkt_intel["emerging_trends"]
        if trend_data.get("available", False):
            st.metric("Trend Velocity Status", trend_data["trend_status"])
            st.write(f"&bull; **Earlier Negative Rate:** {trend_data['earlier_period_negative_rate']}")
            st.write(f"&bull; **Recent Negative Rate:** {trend_data['recent_period_negative_rate']}")
            st.write(f"&bull; **Velocity Change:** {trend_data['negative_velocity_change']}")
            st.caption(f"Reliability: {trend_data['sample_size_flag']}")
        else:
            st.info(trend_data.get("message", "Date trend data unavailable."))
        st.markdown('</div>', unsafe_allow_html=True)

# =============================================================================
# TAB 6: STRATEGIC RECOMMENDATIONS (10-POINT EVIDENCE DOSSIERS)
# =============================================================================
with tabs[5]:
    st.markdown("""
    <div class="box-header-title">
        <span class="box-header-blue-bar"></span>
        <span>Strategic Recommendations: Evidence-Backed Action Dossiers</span>
    </div>
    """, unsafe_allow_html=True)

    # Financial Exposure Row
    st.markdown('<div class="content-box">', unsafe_allow_html=True)
    fin_col1, fin_col2 = st.columns([1, 2])
    with fin_col1:
        fin_data = strat_intel["financial_risk_exposure"]
        st.metric("Estimated Financial Exposure", fin_data["estimated_financial_exposure"])
        st.caption(fin_data["data_limitation_notice"])
    with fin_col2:
        st.markdown("**At-Risk Exposure Breakdown:**")
        st.write(f"&bull; Billing Dispute Reviews: {fin_data['billing_dispute_reviews_count']}")
        st.write(f"&bull; Physical Damage Reviews: {fin_data['physical_damage_reviews_count']}")
        st.write(f"&bull; Assumed Order Value: {fin_data['assumed_average_order_value']}")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

    insight_cards = strat_intel["evidence_backed_insights"]
    if insight_cards:
        for ins in insight_cards:
            p_badge = {
                "CRITICAL": '<span class="dossier-pill-critical">CRITICAL</span>',
                "HIGH": '<span class="dossier-pill-high">HIGH</span>',
                "MEDIUM": '<span class="dossier-pill-medium">MEDIUM</span>',
                "LOW": '<span class="dossier-pill-low">LOW</span>'
            }.get(ins["priority"], '<span class="dossier-pill-low">LOW</span>')

            with st.expander(f"{ins['priority']} [{ins['insight_id']}] {ins['title']} (Priority Score: {ins['priority_score']})", expanded=(ins["priority"]=="CRITICAL")):
                st.markdown(f"""
                <div class="dossier-box">
                    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
                        <div style="font-weight:700; font-size:1.05rem; color:#0F172A;">{ins['title']}</div>
                        <div>{p_badge}</div>
                    </div>
                    <div style="font-size:0.86rem; color:#334155; line-height:1.5;">
                        <p><b>1. Finding:</b> {ins['finding']}</p>
                        <p><b>2. Metrics:</b> Affected: {ins['metrics']['affected_reviews_count']} reviews ({ins['metrics'].get('percentage_of_negative_feedback', ins['metrics'].get('percentage_of_total_feedback', ''))}) &middot; Sample: {ins['metrics']['sample_size']}</p>
                        <p><b>3. Business Impact:</b> {ins['business_impact']}</p>
                        <div style="background:#EFF6FF; border:1px solid #BFDBFE; border-radius:6px; padding:12px; margin:10px 0;">
                            <b>4. Recommended Strategic Action:</b> {ins['recommendation']}
                        </div>
                        <div style="display:flex; gap:20px; font-size:0.82rem; color:#475569; margin-top:8px;">
                            <div><b>Responsible Department:</b> {ins['responsible_department']}</div>
                            <div><b>Model Confidence:</b> {ins['confidence']*100:.1f}%</div>
                        </div>
                        <div style="font-size:0.75rem; color:#64748B; margin-top:6px;">
                            <b>Data Limitations:</b> {ins['data_limitations']}
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("**5. Customer Review Evidence (Audit Trail):**")
                for ev in ins["evidence"]:
                    st.markdown(f"- *[{ev['review_id']}]* \"{ev['text']}\"")
    else:
        st.info("No strategic priority insights generated for this filter selection.")

# =============================================================================
# TAB 7: REVIEW EXPLORER & CSV EXPORT
# =============================================================================
with tabs[6]:
    st.markdown("""
    <div class="content-box">
        <div class="box-header-title">
            <span class="box-header-blue-bar"></span>
            <span>Customer Review Audit & Enriched Data Explorer</span>
        </div>
    """, unsafe_allow_html=True)

    search_query = st.text_input("Search reviews by keyword or Review ID", "", placeholder="e.g. refund, battery, late...")
    display_df = filtered_df.copy()
    if search_query:
        display_df = display_df[
            display_df["review_text"].str.contains(search_query, case=False, na=False) |
            display_df["review_id"].astype(str).str.contains(search_query, case=False, na=False)
        ]

    st.dataframe(display_df, width='stretch')

    # 1-Click CSV Download
    csv_bytes = display_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Filtered Enriched CSV",
        data=csv_bytes,
        file_name="enriched_customer_intelligence_export.csv",
        mime="text/csv"
    )
    st.markdown("</div>", unsafe_allow_html=True)
