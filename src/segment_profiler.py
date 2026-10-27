"""
Automated Segment Profiling and Persona Generation.
Translates raw mathematical cluster statistics into human-readable business personas.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

def profile_clusters(
    df: pd.DataFrame, 
    labels: np.ndarray, 
    feature_cols: List[str],
    dataset_name: str = "Generic"
) -> Dict[str, Any]:
    """
    Computes statistical profiles, percentile rankings, and dynamic business labels
    for each discovered customer cluster.
    """
    df_profile = df.copy()
    df_profile['Cluster'] = labels
    
    total_customers = len(df_profile)
    unique_clusters = sorted([c for c in set(labels) if c != -1])
    
    cluster_profiles = {}
    
    # Compute global medians for benchmark comparisons
    global_medians = {col: df[col].median() for col in feature_cols}
    global_means = {col: df[col].mean() for col in feature_cols}
    
    for c in unique_clusters:
        sub_df = df_profile[df_profile['Cluster'] == c]
        count = len(sub_df)
        pct = round((count / total_customers) * 100, 1)
        
        means = {col: round(float(sub_df[col].mean()), 2) for col in feature_cols}
        medians = {col: round(float(sub_df[col].median()), 2) for col in feature_cols}
        stds = {col: round(float(sub_df[col].std()), 2) for col in feature_cols}
        
        # Determine business label & persona description
        persona_info = _assign_business_persona(
            c, means, global_means, dataset_name, feature_cols
        )
        
        cluster_profiles[c] = {
            "cluster_id": c,
            "persona_name": persona_info["title"],
            "icon": persona_info["icon"],
            "badge_color": persona_info["color"],
            "description": persona_info["description"],
            "key_traits": persona_info["key_traits"],
            "customer_count": count,
            "percentage": pct,
            "feature_means": means,
            "feature_medians": medians,
            "feature_stds": stds
        }
        
    return {
        "cluster_profiles": cluster_profiles,
        "global_means": {col: round(float(v), 2) for col, v in global_means.items()},
        "global_medians": {col: round(float(v), 2) for col, v in global_medians.items()},
        "total_customers": total_customers
    }

def _assign_business_persona(
    cluster_id: int, 
    cluster_means: Dict[str, float], 
    global_means: Dict[str, float],
    dataset_name: str,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """Generates intelligent business title, traits, and description."""
    
    # 1. Online Retail / RFM Dataset logic
    if any(col.lower() in ['recency', 'frequency', 'monetary'] for col in feature_cols):
        r = cluster_means.get('Recency', global_means.get('Recency', 0))
        f = cluster_means.get('Frequency', global_means.get('Frequency', 0))
        m = cluster_means.get('Monetary', global_means.get('Monetary', 0))
        
        r_mean = global_means.get('Recency', 1)
        f_mean = global_means.get('Frequency', 1)
        m_mean = global_means.get('Monetary', 1)
        
        # Low Recency = bought recently (good). High F & M = loyal & high spender.
        if r < r_mean and f >= f_mean and m >= m_mean:
            return {
                "title": "Champions & High-Value VIPs",
                "icon": "👑",
                "color": "#10B981", # Emerald
                "description": "High spending and frequent transactions with recent activity. Top drivers of business revenue.",
                "key_traits": ["High Monetary Value", "High Purchase Frequency", "Recent Engagement"]
            }
        elif r < r_mean and f >= f_mean and m < m_mean:
            return {
                "title": "Loyal Frequent Shoppers",
                "icon": "🔁",
                "color": "#3B82F6", # Blue
                "description": "Consistent repeat buyers with high transaction volume, though basket sizes are moderate.",
                "key_traits": ["Frequent Repeat Visits", "Moderate Basket Size", "Active Engagement"]
            }
        elif r < r_mean and f < f_mean:
            return {
                "title": "Promising New / Active Shoppers",
                "icon": "🌱",
                "color": "#8B5CF6", # Purple
                "description": "Recent shoppers with growing potential. Great targets for nurture flows and repeat purchase incentives.",
                "key_traits": ["Recent First Orders", "Low/Moderate Frequency", "Growth Opportunity"]
            }
        elif r >= r_mean and m >= m_mean:
            return {
                "title": "At-Risk Inactive Spenders",
                "icon": "⚠️",
                "color": "#F59E0B", # Amber
                "description": "Historically high-value customers who have not made a purchase recently. Immediate win-back needed.",
                "key_traits": ["High Lifetime Value", "Elevated Days Since Purchase", "Risk of Churn"]
            }
        else:
            return {
                "title": "Hibernating / Price-Sensitive",
                "icon": "💤",
                "color": "#64748B", # Slate
                "description": "Infrequent purchases with low total spend and lengthy inactivity. Highly price-sensitive.",
                "key_traits": ["Long Inactivity", "Low Purchase Volume", "Discount Driven"]
            }

    # 2. Mall Customers logic (Income & Spending Score)
    has_income = any('income' in c.lower() for c in feature_cols)
    has_spending = any('spending' in c.lower() for c in feature_cols)
    
    if has_income and has_spending:
        inc_col = next(c for c in feature_cols if 'income' in c.lower())
        spd_col = next(c for c in feature_cols if 'spending' in c.lower())
        
        inc = cluster_means[inc_col]
        spd = cluster_means[spd_col]
        inc_avg = global_means[inc_col]
        spd_avg = global_means[spd_col]
        
        if inc >= inc_avg and spd >= spd_avg:
            return {
                "title": "Affluent High-Spenders (Prime Target)",
                "icon": "💎",
                "color": "#10B981",
                "description": "High annual earnings paired with high propensity to spend. Ideal audience for luxury and premium offerings.",
                "key_traits": ["High Purchasing Power", "High Brand Affinity", "Premium Taste"]
            }
        elif inc >= inc_avg and spd < spd_avg:
            return {
                "title": "High-Income Savers / Conservative Buyers",
                "icon": "💼",
                "color": "#3B82F6",
                "description": "High purchasing power but selective spending habits. Needs clear value propositions or utilitarian justification.",
                "key_traits": ["High Disposable Income", "Cautious Spender", "Value-Conscious"]
            }
        elif inc < inc_avg and spd >= spd_avg:
            return {
                "title": "Aspirational / Trend Enthusiasts",
                "icon": "🛍️",
                "color": "#EC4899",
                "description": "Moderate to low income yet very high retail spending. Highly responsive to lifestyle marketing and latest trends.",
                "key_traits": ["High Spending Propensity", "Trend Seeking", "Impulsive Tendencies"]
            }
        elif inc < inc_avg and spd < spd_avg:
            return {
                "title": "Budget-Conscious / Frugal Consumers",
                "icon": "🏷️",
                "color": "#F97316",
                "description": "Low income with restrained spending behavior. Responsive almost exclusively to steep clearances and essentials.",
                "key_traits": ["Low Spending", "Price Sensitive", "Bargain Hunter"]
            }
        else:
            return {
                "title": "Middle-Tier Balanced Shoppers",
                "icon": "⚖️",
                "color": "#6366F1",
                "description": "Balanced income and average spending score. Steady baseline demographic of retail traffic.",
                "key_traits": ["Moderate Income", "Standard Spending", "Consistent Footfall"]
            }

    # 3. Dynamic Fallback for Any Custom Dataset
    # Find the feature with the highest relative z-score / deviation
    best_feat = max(feature_cols, key=lambda c: (cluster_means[c] - global_means[c]) / (global_means[c] + 1e-5))
    lowest_feat = min(feature_cols, key=lambda c: (cluster_means[c] - global_means[c]) / (global_means[c] + 1e-5))
    
    return {
        "title": f"Segment {cluster_id + 1} (High {best_feat})",
        "icon": "📊",
        "color": "#6366F1",
        "description": f"Characterized by high {best_feat} and comparatively lower {lowest_feat}.",
        "key_traits": [f"High {best_feat}", f"Relative {lowest_feat}", "Distinct Behavior"]
    }
