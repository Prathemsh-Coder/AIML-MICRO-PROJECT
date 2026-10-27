"""
Natural Language Analytics Query Engine.
Interprets analytical questions from users and executes predefined statistical queries
without requiring paid third-party LLM APIs.
"""

from typing import Dict, Any, List
import pandas as pd

def process_analytics_query(
    query_text: str,
    df_clustered: pd.DataFrame,
    cluster_profiles: Dict[int, Any],
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    Parses natural language query text and returns structured response with metrics and charts.
    """
    q = query_text.lower().strip()
    
    # 1. Which segment spends the most? / Highest monetary / Highest spending
    if any(k in q for k in ["spends the most", "highest spend", "most spend", "maximum spend", "highest monetary", "top spend"]):
        spend_col = None
        for col in ["Monetary", "Spending Score (1-100)", "Annual Income (k$)"]:
            if col in feature_cols:
                spend_col = col
                break
        if not spend_col:
            spend_col = feature_cols[0]
            
        best_cluster = max(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]["feature_means"].get(spend_col, 0))
        prof = cluster_profiles[best_cluster]
        val = prof["feature_means"].get(spend_col, 0)
        
        return {
            "query": query_text,
            "answer": f"**{prof['icon']} {prof['persona_name']} (Cluster {best_cluster})** is the highest spending group, with an average {spend_col} of **{val:,.2f}**.",
            "data": [{"Cluster": f"Cluster {c}: {cluster_profiles[c]['persona_name']}", spend_col: cluster_profiles[c]["feature_means"].get(spend_col, 0)} for c in cluster_profiles],
            "highlight_metric": f"{spend_col}: {val:,.2f}"
        }

    # 2. How many customers in each segment? / Segment distribution / sizes
    if any(k in q for k in ["how many", "count", "distribution", "belong to each", "size of each", "customer count"]):
        breakdown = []
        for c, p in cluster_profiles.items():
            breakdown.append(f"- **{p['icon']} {p['persona_name']}**: {p['customer_count']:,} customers ({p['percentage']}%)")
            
        text = "Here is the customer distribution across discovered segments:\n\n" + "\n".join(breakdown)
        return {
            "query": query_text,
            "answer": text,
            "data": [{"Cluster": p["persona_name"], "Customers": p["customer_count"], "Percentage": p["percentage"]} for c, p in cluster_profiles.items()],
            "highlight_metric": f"Total Customers: {len(df_clustered):,}"
        }

    # 3. Which segment is largest or smallest?
    if any(k in q for k in ["largest segment", "biggest segment", "most customers"]):
        largest_c = max(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]["customer_count"])
        p = cluster_profiles[largest_c]
        return {
            "query": query_text,
            "answer": f"The largest segment is **{p['icon']} {p['persona_name']}** with **{p['customer_count']:,} customers** ({p['percentage']}% of total).",
            "highlight_metric": f"{p['customer_count']:,} customers"
        }
        
    if any(k in q for k in ["smallest segment", "fewest customers", "least customers"]):
        smallest_c = min(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]["customer_count"])
        p = cluster_profiles[smallest_c]
        return {
            "query": query_text,
            "answer": f"The smallest segment is **{p['icon']} {p['persona_name']}** with **{p['customer_count']:,} customers** ({p['percentage']}% of total).",
            "highlight_metric": f"{p['customer_count']:,} customers"
        }

    # 4. At risk / churn / inactive
    if any(k in q for k in ["at risk", "churn", "inactive", "leaving", "stop purchasing"]):
        at_risk_clusters = [c for c, p in cluster_profiles.items() if "risk" in p["persona_name"].lower() or "hibernating" in p["persona_name"].lower()]
        if not at_risk_clusters:
            # Check cluster with highest recency if recency in features
            if "Recency" in feature_cols:
                at_risk_clusters = [max(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]["feature_means"].get("Recency", 0))]
            else:
                at_risk_clusters = [min(cluster_profiles.keys(), key=lambda c: cluster_profiles[c]["customer_count"])]
                
        details = []
        for c in at_risk_clusters:
            p = cluster_profiles[c]
            details.append(f"**{p['icon']} {p['persona_name']} (Cluster {c})**: {p['customer_count']} customers. High priority for targeted win-back campaigns and re-engagement vouchers.")
            
        return {
            "query": query_text,
            "answer": "Identified at-risk / churn-prone groups:\n\n" + "\n\n".join(details),
            "highlight_metric": f"{len(at_risk_clusters)} At-Risk Segment(s)"
        }

    # 5. Loyalty / VIP / Rewards
    if any(k in q for k in ["loyalty", "vip", "reward", "special offer", "discounts"]):
        vip_clusters = [c for c, p in cluster_profiles.items() if "champion" in p["persona_name"].lower() or "vip" in p["persona_name"].lower() or "affluent" in p["persona_name"].lower() or "loyal" in p["persona_name"].lower()]
        if not vip_clusters:
            vip_clusters = [0]
            
        details = []
        for c in vip_clusters:
            p = cluster_profiles[c]
            details.append(f"- **{p['icon']} {p['persona_name']}**: Qualifies for high-touch rewards, priority support, and exclusive VIP tiers.")
            
        return {
            "query": query_text,
            "answer": "Segments recommended for loyalty & special privileges:\n\n" + "\n".join(details),
            "highlight_metric": "VIP Loyalty Candidates"
        }

    # 6. Specific cluster query (e.g., "tell me about cluster 0" or "cluster 1")
    for c in cluster_profiles.keys():
        if f"cluster {c}" in q or f"segment {c}" in q or f"cluster {c+1}" in q:
            p = cluster_profiles[c]
            metrics_str = ", ".join([f"{feat}: {val}" for feat, val in p["feature_means"].items()])
            return {
                "query": query_text,
                "answer": f"**{p['icon']} Cluster {c}: {p['persona_name']}**\n- **Size**: {p['customer_count']:,} ({p['percentage']}%)\n- **Overview**: {p['description']}\n- **Averages**: {metrics_str}",
                "highlight_metric": p["persona_name"]
            }

    # 7. Fallback / General Summary
    summary_lines = [f"- **{p['icon']} {p['persona_name']}**: {p['customer_count']} members ({p['percentage']}%)" for c, p in cluster_profiles.items()]
    return {
        "query": query_text,
        "answer": f"I couldn't match a specific single-metric question, so here is the summary of your **{len(cluster_profiles)} customer segments**:\n\n" + "\n".join(summary_lines) + "\n\n*Try asking: 'Which segment spends the most?', 'Who are the at-risk customers?', or 'How many customers belong to each segment?'*",
        "highlight_metric": f"{len(cluster_profiles)} Clusters Active"
    }
