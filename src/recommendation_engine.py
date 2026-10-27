"""
Personalized Marketing Recommendation Engine.
Produces targeted business marketing strategies, campaign playbooks, 
and individualized customer engagement blueprints.
"""

from typing import Dict, Any, List
import pandas as pd

def generate_segment_marketing_playbook(segment_title: str, traits: List[str]) -> Dict[str, Any]:
    """Generates an end-to-end tactical marketing campaign playbook for a segment."""
    title_lower = segment_title.lower()
    
    if "champion" in title_lower or "affluent" in title_lower or "vip" in title_lower:
        return {
            "objective": "VIP Retention & Lifetime Value Maximization",
            "recommended_channels": ["Dedicated Concierge", "Private In-App Lounge", "Exclusive Email Invitations"],
            "campaign_title": "Platinum Elite Preview & White-Glove Perks",
            "suggested_offer": "Early access to new product drops + zero-threshold free express shipping + invitation-only VIP events.",
            "discount_strategy": "Avoid heavy percentage discounts; offer luxury gifts-with-purchase or experiential rewards instead.",
            "communication_frequency": "Low frequency (High touch, 1-2 curated contacts per month)",
            "action_priority": "High (Protect Core Revenue)"
        }
    elif "loyal" in title_lower or "frequent" in title_lower:
        return {
            "objective": "Basket Size Expansion & Brand Advocacy",
            "recommended_channels": ["Automated Email Drip", "Push Notifications", "Tiered Loyalty Program"],
            "campaign_title": "Next-Tier VIP Accelerator & Tiered Multi-Buys",
            "suggested_offer": "Spend $50 more this month to unlock double reward points and next-tier tier benefits.",
            "discount_strategy": "Threshold discounts (e.g., 'Get $20 off on orders over $120') or product bundles.",
            "communication_frequency": "Medium frequency (Weekly updates, new category cross-sells)",
            "action_priority": "High (Cross-Sell Engine)"
        }
    elif "promising" in title_lower or "active" in title_lower or "aspirational" in title_lower or "trend" in title_lower:
        return {
            "objective": "Second Purchase Conversion & Habit Formation",
            "recommended_channels": ["Social Retargeting (Instagram/Meta)", "SMS Notifications", "Welcome Series Email"],
            "campaign_title": "Curated Essentials & Second-Order Kickstart",
            "suggested_offer": "15% off your second order within 14 days + complimentary best-seller samples.",
            "discount_strategy": "Time-limited percentage discounts on trending or complementary accessories.",
            "communication_frequency": "Medium-High frequency during the first 30 days post-order",
            "action_priority": "Medium (Nurture Pipeline)"
        }
    elif "at-risk" in title_lower or "savers" in title_lower or "conservative" in title_lower:
        return {
            "objective": "Churn Prevention & Re-Activation",
            "recommended_channels": ["Personalized 'We Miss You' Email", "Dynamic Retargeting Ads", "Direct SMS Voucher"],
            "campaign_title": "Comeback Exclusive & Customer Pulse Survey",
            "suggested_offer": "$25 gift credit applied automatically on your cart + short feedback survey for an extra bonus.",
            "discount_strategy": "Direct dollar-off vouchers with a defined expiry date (7 days).",
            "communication_frequency": "Strategic 3-step re-engagement cadence over 21 days",
            "action_priority": "Urgent (Mitigate Revenue Loss)"
        }
    else:  # Budget / Hibernating / Price-Sensitive
        return {
            "objective": "Low-Cost Re-engagement & Inventory Clearance",
            "recommended_channels": ["Bulk Promotional Newsletter", "Push Clearance Alerts", "SMS Flash Sales"],
            "campaign_title": "Seasonal Clearance & Outlet Flash Sale",
            "suggested_offer": "Up to 50% off seasonal clearance items + buy-one-get-one-half-price essentials.",
            "discount_strategy": "Aggressive clearance pricing, bulk bundles, and clearance markdowns.",
            "communication_frequency": "Low cost / Automated batch blasts during peak sale events (Black Friday, End-of-Season)",
            "action_priority": "Low/Automated (Cost-Efficient Outreach)"
        }

def analyze_individual_customer(
    customer_id: Any,
    df_with_clusters: pd.DataFrame,
    cluster_profiles: Dict[int, Any],
    feature_cols: List[str],
    id_col: str = "CustomerID"
) -> Dict[str, Any]:
    """Analyzes a specific customer's position within their cluster and generates custom guidance."""
    matches = df_with_clusters[df_with_clusters[id_col].astype(str) == str(customer_id)]
    if len(matches) == 0:
        return {"found": False, "message": f"Customer ID '{customer_id}' not found in dataset."}
        
    cust_row = matches.iloc[0]
    cluster_id = int(cust_row['Cluster'])
    profile = cluster_profiles.get(cluster_id, {})
    
    # Feature comparisons: Customer vs Cluster Average vs Global Average
    comparisons = []
    for feat in feature_cols:
        val = cust_row[feat]
        cluster_mean = profile.get("feature_means", {}).get(feat, val)
        diff_pct = round(((val - cluster_mean) / (cluster_mean + 1e-5)) * 100, 1)
        comparisons.append({
            "feature": feat,
            "customer_value": round(float(val), 2),
            "cluster_average": round(float(cluster_mean), 2),
            "deviation_pct": diff_pct,
            "status": "Above Average" if diff_pct > 5 else ("Below Average" if diff_pct < -5 else "On Par")
        })
        
    playbook = generate_segment_marketing_playbook(
        profile.get("persona_name", f"Segment {cluster_id}"),
        profile.get("key_traits", [])
    )
    
    return {
        "found": True,
        "customer_id": str(customer_id),
        "cluster_id": cluster_id,
        "persona_name": profile.get("persona_name", f"Cluster {cluster_id}"),
        "icon": profile.get("icon", "👤"),
        "badge_color": profile.get("badge_color", "#6366F1"),
        "comparisons": comparisons,
        "playbook": playbook,
        "raw_attributes": {col: cust_row[col] for col in df_with_clusters.columns if col != 'Cluster'}
    }
