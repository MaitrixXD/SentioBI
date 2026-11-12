# 📈 SentioBI — AI-Powered Customer & Business Intelligence Platform

> **Transforming raw, unstructured customer reviews into actionable business strategy, operational fixes, and revenue opportunities.**

[![Python 3.14](https://img.shields.io/badge/Python-3.14-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.42+-red.svg)](https://streamlit.io/)
[![HuggingFace](https://img.shields.io/badge/Transformers-Offline%20Cached-yellow.svg)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Most sentiment analysis tools stop at labeling text as *Positive*, *Negative*, or *Neutral*. **SentioBI** bridges the gap between natural language processing and executive decision-making. It doesn't just tell businesses **what** customers are saying — it explains **why** it matters, **what business impact** it causes, and **what specific actions** leadership should take.

---

## 🌟 Master Business Intelligence Framework

The platform organizes intelligence into **5 core business dimensions**:

### 1. 👥 Customer Intelligence (Category A)
- **Net Sentiment Score (NSS):** Standardized metric computed as `(% Positive) - (% Negative)` on a `-100` to `+100` scale.
- **7-Class Emotional Pulse:** Fine-grained emotion distribution (Joy, Anger, Sadness, Fear, Disgust, Surprise, Neutral) paired with an empathetic customer action playbook.
- **Customer Churn Risk Signals:** Pattern-driven detection of cancellation threats, competitor switching signals, and refund demands.
- **Brand Advocates & Loyalty Mentions:** Identification of organic customer praise and testimonial opportunities.

### 2. 📦 Product Intelligence (Category B)
- **Aspect Root-Cause Breakdown:** Granular diagnostics across key business areas (Delivery & Shipping, Customer Support, Pricing & Billing, Product Quality, User Interface).
- **Automated Root-Cause Hypotheses:** Generates testable explanations for why specific product aspects are failing.
- **Customer Feature Innovation Taxonomy:** Discovers and categorizes customer requests (New Features, Improvements, Integrations, Usability).
- **Price Sensitivity & Value-for-Money:** Tracks complaints regarding pricing vs. praise for value to inform pricing strategy.

### 3. ⚙️ Operational Intelligence (Category C)
- **Automatic Department Intent Routing:** Classifies reviews by customer intent and routes them to Engineering, Customer Support, Finance & Billing, Operations, Product Management, or Marketing.
- **Urgent Escalation Triage:** 4-tier escalation model (CRITICAL, HIGH, MEDIUM, LOW) based on severe business impact (duplicate charges, fraud, safety concerns).
- **Systemic Repeat Failure Clusters:** Groups reviews describing the same underlying problem using semantic patterns.
- **8-Stage Customer Journey Friction Funnel:** Identifies stage-by-stage drop-off and friction across Discovery, Research, Purchase, Payment, Delivery, Product Usage, After-Sales Support, and Refunds.

### 4. 🌐 Market Intelligence (Category D)
- **Competitor Mention Extraction:** Flags when customers compare your product against market alternatives.
- **Emerging Complaint Velocity Monitor:** Detects sudden spikes in new issues across time periods.

### 5. 🎯 Strategic Recommendations (Category E)
- **Transparent Decision Priority Score:** A weighted multi-variable ranking formula:
  $$\text{Priority Score} = (\text{Severity} \times 0.35) + (\text{Frequency} \times 0.30) + (\text{ChurnRisk} \times 0.20) + (\text{Urgency} \times 0.15)$$
- **Potential Financial Risk Exposure:** Quantifies the estimated dollar risk of unresolved complaints (e.g., billing disputes and physical transit damage $\times$ Assumed Average Order Value).
- **10-Point Evidence-Backed Action Dossiers:** Every recommendation includes: Finding, Metrics, Business Impact, Recommended Action, Traceable Customer Review Evidence, Responsible Department, Model Confidence, and Data Limitations.

---

## 🧠 Multi-Model AI Architecture

SentioBI runs **100% offline** with locally cached models—ensuring zero data leakage, complete data privacy, and fast inference:

| Model | Architecture | Purpose |
|---|---|---|
| `cardiffnlp/twitter-roberta-base-sentiment-latest` | RoBERTa (125M) | 3-Class Sentiment Analysis |
| `j-hartmann/emotion-english-distilroberta-base` | DistilRoBERTa (82M) | 7-Class Granular Emotion Detection |
| `martin-ha/toxic-comment-model` | DistilRoBERTa (82M) | Toxicity & Escalation Signal Detection |
| `valhalla/distilbart-mnli-12-3` | DistilBART-MNLI (306M) | Zero-Shot Aspect Tagging & Intent Classification |

---

## ⚡ Performance & Optimization

- **Vectorized Batched Pipelines (`batch_size=16`):** Ingests and processes batches of reviews in parallel, cutting Python loop overhead.
- **Multi-Core CPU Parallelism:** Utilizes all CPU cores via PyTorch intra-op threading.
- **Automated Enriched Dataset Caching:** When a raw CSV is uploaded, the enriched output is automatically saved (`enriched_<filename>.csv`) for instant (<0.1s) subsequent loads.

---

## 🖥️ Executive Analytics Dashboard

SentioBI features an executive UI built with Streamlit and Plotly:
* **Executive Top Banner & 5 KPI Metric Cards:** Analyzed Feedback, Net Sentiment Score, Urgent Escalations, High Churn Signals, and Top Problem Aspect with inline SVG trend waves.
* **7 Structured Navigation Tabs:** Seamlessly switch between high-level overview and deep-dive intelligence categories.
* **Review-Level Audit Trail:** Inspect every single customer feedback item alongside its AI tags, with full-text search and 1-click enriched CSV export.

---

## 🚀 Quickstart

### 1. Clone the repository
```bash
git clone https://github.com/MaitrixXD/SentioBI.git
cd SentioBI
```

### 2. Set up virtual environment & install dependencies
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Dashboard
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📄 License
MIT License.
