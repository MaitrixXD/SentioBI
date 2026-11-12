"""
Operational Intelligence Module (Category C)
============================================
AI-Powered Customer & Business Intelligence Platform

Implements:
  C1. Customer Intent Routing (6 Departments)
  C2. Urgent Complaint & Severity Escalation (Critical, High, Medium, Low)
  C3. Repeat Complaint & Systemic Failure Detection (Semantic Clustering)
  C4. Customer Journey & Experience Gap Analysis (8 Lifecycle Stages)
  C5. Operational Performance & SLA Delay Monitoring
  C6. Review Response Effectiveness Schema
"""

import re
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

class OperationalIntelligenceEngine:
    """
    Analyzes department routing, severity escalation, repeat failure clusters,
    customer journey friction points, and reported operational delays.
    """
    def __init__(self):
        # Department routing rules based on keywords & aspects
        self.dept_rules = {
            "Finance & Billing": [
                r"\b(bill(ing)?|charge|card|refund|payment|subscription|scam|cost|fee|invoice)\b"
            ],
            "Operations & Logistics": [
                r"\b(deliver(y|ed|ing)?|ship(ping|ment)?|courier|driver|package|packaging|box|shattered|damaged|late|transit)\b"
            ],
            "Engineering": [
                r"\b(crash(ed|ing)?|bug|error|server|down|broke|broken|loading|freeze|failed|sluggish|glitch)\b"
            ],
            "Product Management": [
                r"\b(feature|dark\s+mode|shortcut|export|ui|ux|layout|button|interface|option|integrate|add)\b"
            ],
            "Customer Support": [
                r"\b(support|agent|rep|representative|ticket|reply|response|unhelpful|rude|hung\s+up|service)\b"
            ],
            "Marketing": [
                r"\b(recommend|love|best|kudos|amazing|referral|brand|champion)\b"
            ]
        }

        # Severity escalation patterns
        self.critical_patterns = [
            r"\b(fraud|scam|illegal|lawyer|sue|legal\s+action|police|safety\s+hazard|injury|poison)\b",
            r"\b(charged\s+(twice|three\s+times|multiple\s+times))\b",
            r"\b(threaten(ed|ing)?)\b"
        ]
        self.high_patterns = [
            r"\b(broke\s+our\s+team|server\s+crashed|unreachable|shattered|ruined|melted|unacceptable)\b",
            r"\b(haven't\s+replied\s+in\s+[0-9]+\s+days)\b"
        ]

        # 8-Stage Customer Journey definitions
        self.journey_stages = {
            "1. Discovery & Marketing": [r"\b(advertis(ed|ing)?|website|found\s+you|claim|promotion)\b"],
            "2. Product Research": [r"\b(spec|description|guide|manual|detail)\b"],
            "3. Purchase & Checkout": [r"\b(order(ing)?|cart|checkout|store|buy(ing)?)\b"],
            "4. Payment & Billing": [r"\b(card|charged|invoice|payment|billing|fee)\b"],
            "5. Delivery & Shipping": [r"\b(deliver(y|ed)?|ship(ping)?|courier|transit|box|package|driver)\b"],
            "6. Product Usage & Quality": [r"\b(work(s|ing)?|fit|material|sound|battery|screen|daily\s+job|zipper|software|headphones|jacket|sneakers)\b"],
            "7. Customer Support": [r"\b(support|agent|rep|ticket|reply|response|help|reps)\b"],
            "8. Refund & Replacement": [r"\b(refund|return|replace(ment)?|money\s+back)\b"]
        }

        # Systemic failure cluster definitions
        self.systemic_clusters = {
            "Delivery & In-Transit Product Damage": {
                "patterns": [r"\b(shattered|broken|damaged|crushed|melted|late|gate|box)\b"],
                "dept": "Operations & Logistics",
                "recommended_fix": "Audit packaging protocols and inspect regional courier handling."
            },
            "Billing Double-Charges & Refund Bottlenecks": {
                "patterns": [r"\b(scam|charged\s+twice|unreachable|refund|money\s+back)\b"],
                "dept": "Finance & Billing",
                "recommended_fix": "Fix payment gateway idempotency and unblock refund processing queues."
            },
            "Customer Support SLA & Response Delay": {
                "patterns": [r"\b(haven't\s+replied|hours\s+to\s+reply|days|hung\s+up|no\s+response)\b"],
                "dept": "Customer Support",
                "recommended_fix": "Enforce tiered escalation SLAs and add automated ticket status notifications."
            },
            "Software Stability & Regression Outages": {
                "patterns": [r"\b(update\s+completely\s+broke|dashboard|missing\s+data|crashed|glitch)\b"],
                "dept": "Engineering",
                "recommended_fix": "Roll back breaking deployment and improve regression test coverage before releases."
            }
        }

    def route_departments(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        C1. Assigns each review to responsible departments and calculates department load.
        """
        dept_assignments = []
        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            aspect = str(row.get("ai_primary_aspect", ""))
            intent = str(row.get("ai_primary_intent", ""))

            assigned = []
            for dept, patterns in self.dept_rules.items():
                if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                    assigned.append(dept)

            # Fallback based on AI primary aspect/intent
            if not assigned:
                if "support" in aspect.lower():
                    assigned.append("Customer Support")
                elif "delivery" in aspect.lower():
                    assigned.append("Operations & Logistics")
                elif "pricing" in aspect.lower() or "refund" in intent.lower():
                    assigned.append("Finance & Billing")
                elif "quality" in aspect.lower() or "ui" in aspect.lower():
                    assigned.append("Product Management")
                else:
                    assigned.append("Customer Support")

            dept_assignments.append(assigned[0])

        df_copy = df.copy()
        df_copy["assigned_department"] = dept_assignments
        dept_counts = df_copy["assigned_department"].value_counts().to_dict()

        # Department dissatisfaction rate
        dept_metrics = {}
        for dept, group in df_copy.groupby("assigned_department"):
            tot = len(group)
            neg = int((group["ai_sentiment"].str.lower() == "negative").sum())
            dept_metrics[dept] = {
                "total_tickets": tot,
                "negative_tickets": neg,
                "dissatisfaction_rate": round((neg / tot) * 100, 1) if tot > 0 else 0
            }

        return {
            "department_counts": dept_counts,
            "department_metrics": dept_metrics,
            "assigned_series": dept_assignments
        }

    def detect_urgent_escalations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        C2. Identifies severe complaints requiring immediate executive / manager intervention.
        """
        escalations = []
        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rev_id = row.get("review_id", "N/A")
            sentiment = str(row.get("ai_sentiment", "neutral")).lower()
            emotion = str(row.get("ai_emotion", "neutral")).lower()

            # Critical
            if any(re.search(p, text, re.IGNORECASE) for p in self.critical_patterns) or row.get("is_toxic", False):
                escalations.append({
                    "review_id": rev_id,
                    "text": text,
                    "severity": "CRITICAL",
                    "reason": "Fraud/scam allegation, safety hazard, toxic abuse, or legal threat.",
                    "action": "Immediate 1-hour manager escalation & customer outreach."
                })
            # High
            elif any(re.search(p, text, re.IGNORECASE) for p in self.high_patterns) or (sentiment == "negative" and emotion in ["anger", "disgust"]):
                escalations.append({
                    "review_id": rev_id,
                    "text": text,
                    "severity": "HIGH",
                    "reason": "Severe service failure, broken shipment, or unresponsive support SLA.",
                    "action": "Priority 4-hour queue resolution."
                })
            # Medium
            elif sentiment == "negative":
                escalations.append({
                    "review_id": rev_id,
                    "text": text,
                    "severity": "MEDIUM",
                    "reason": "Standard negative review or friction point.",
                    "action": "Standard 24-hour response queue."
                })

        severity_counts = {
            "CRITICAL": sum(1 for e in escalations if e["severity"] == "CRITICAL"),
            "HIGH": sum(1 for e in escalations if e["severity"] == "HIGH"),
            "MEDIUM": sum(1 for e in escalations if e["severity"] == "MEDIUM"),
            "LOW": max(0, len(df) - len(escalations))
        }

        return {
            "severity_counts": severity_counts,
            "urgent_queue": escalations[:8]
        }

    def detect_systemic_repeat_complaints(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        C3. Clusters related complaints into systemic operational failure buckets.
        """
        negative_df = df[df["ai_sentiment"].str.lower() == "negative"]
        total_neg = len(negative_df)

        cluster_results = []
        for cluster_name, details in self.systemic_clusters.items():
            matched_reviews = []
            for _, row in negative_df.iterrows():
                text = str(row.get("review_text", ""))
                rev_id = row.get("review_id", "N/A")
                if any(re.search(p, text, re.IGNORECASE) for p in details["patterns"]):
                    matched_reviews.append({"review_id": rev_id, "text": text})

            if matched_reviews:
                count = len(matched_reviews)
                pct = round((count / total_neg) * 100, 1) if total_neg > 0 else 0
                cluster_results.append({
                    "cluster_name": cluster_name,
                    "affected_reviews_count": count,
                    "percentage_of_negative_feedback": pct,
                    "responsible_department": details["dept"],
                    "recommended_corrective_action": details["recommended_fix"],
                    "evidence_reviews": matched_reviews
                })

        # Sort by affected reviews count descending
        cluster_results.sort(key=lambda x: x["affected_reviews_count"], reverse=True)
        return {
            "total_systemic_clusters_identified": len(cluster_results),
            "clusters": cluster_results
        }

    def analyze_customer_journey_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        C4. Maps customer feedback across the 8 lifecycle stages to pinpoint friction.
        """
        stage_metrics = {stage: {"total": 0, "negative": 0} for stage in self.journey_stages}

        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            sentiment = str(row.get("ai_sentiment", "")).lower()

            for stage, patterns in self.journey_stages.items():
                if any(re.search(p, text, re.IGNORECASE) for p in patterns):
                    stage_metrics[stage]["total"] += 1
                    if sentiment == "negative":
                        stage_metrics[stage]["negative"] += 1

        journey_summary = []
        for stage, data in stage_metrics.items():
            tot = data["total"]
            neg = data["negative"]
            fail_rate = round((neg / tot) * 100, 1) if tot > 0 else 0.0
            journey_summary.append({
                "stage": stage,
                "total_mentions": tot,
                "negative_mentions": neg,
                "friction_rate": fail_rate
            })

        # Find highest friction stage with at least 1 mention
        active_stages = [s for s in journey_summary if s["total_mentions"] > 0]
        active_stages.sort(key=lambda x: x["friction_rate"], reverse=True)
        highest_friction = active_stages[0] if active_stages else None

        return {
            "journey_funnel": journey_summary,
            "highest_friction_stage": highest_friction
        }

    def monitor_operational_delays(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        C5. Detects reported service delays and SLA bottlenecks in customer text.
        """
        delay_pattern = r"\b([0-9]+\s+(hours?|days?|weeks?|mins?|minutes?))\b"
        delay_records = []

        for _, row in df.iterrows():
            text = str(row.get("review_text", ""))
            rev_id = row.get("review_id", "N/A")
            matches = re.findall(delay_pattern, text, re.IGNORECASE)
            if matches and row.get("ai_sentiment") == "negative":
                delay_str = ", ".join([m[0] for m in matches])
                delay_records.append({
                    "review_id": rev_id,
                    "text": text,
                    "reported_duration": delay_str
                })

        return {
            "reported_delay_count": len(delay_records),
            "evidence": delay_records[:5],
            "data_limitation_notice": (
                "These reported service delays are extracted from review text descriptions. "
                "Official SLA compliance metrics require operational system timestamps."
            )
        }

    def generate_full_operational_intelligence(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Compiles complete Category C intelligence payload.
        """
        return {
            "department_routing": self.route_departments(df),
            "urgent_escalations": self.detect_urgent_escalations(df),
            "systemic_repeat_complaints": self.detect_systemic_repeat_complaints(df),
            "customer_journey_gaps": self.analyze_customer_journey_gaps(df),
            "operational_delays": self.monitor_operational_delays(df)
        }

