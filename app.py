"""
SmartSegment — AI-Powered Customer Segmentation & Personalized Marketing Recommendation System
Streamlit Web Application
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from typing import List, Dict, Any

from src.data_loader import load_preloaded_dataset, inspect_dataset, clean_and_prepare_features
from src.ml_pipeline import scale_features, compute_elbow_and_silhouette, fit_clustering_model, apply_pca_reduction
from src.segment_profiler import profile_clusters
from src.recommendation_engine import generate_segment_marketing_playbook, analyze_individual_customer
from src.query_engine import process_analytics_query

# Page Configuration
st.set_page_config(
    page_title="SmartSegment AI | Customer Segmentation & Marketing",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# SIDEBAR CONTROLS & THEME TOGGLE
# -------------------------------------------------------------
with st.sidebar:
    st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=400&auto=format&fit=crop&q=60", use_container_width=True)
    st.title("🛍️ SmartSegment AI")
    st.caption("Machine Learning Customer Segmentation & Marketing Recommendation System")
    
    st.markdown("---")
    st.subheader("🎨 Appearance Mode")
    theme_choice = st.radio(
        "Display Mode",
        options=["🌙 Dark Mode", "☀️ Light Mode"],
        index=0 if st.session_state.get("theme_choice", "🌙 Dark Mode") == "🌙 Dark Mode" else 1,
        key="theme_choice",
        horizontal=True,
        label_visibility="collapsed"
    )
    is_light = (theme_choice == "☀️ Light Mode")
    plotly_template = "plotly_white" if is_light else "plotly_dark"
    pio.templates.default = plotly_template

    st.markdown("---")
    st.subheader("1. Dataset Selection")
    data_source = st.radio(
        "Choose Customer Dataset:",
        ["Online Retail (RFM)", "Mall Customers", "Upload Custom CSV"],
        index=0
    )
    
    uploaded_file = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV File", type=["csv", "xlsx"])
        
    st.markdown("---")
    st.subheader("2. Model & Hyperparameters")
    algorithm = st.selectbox(
        "Clustering Algorithm:",
        ["K-Means", "Hierarchical (Agglomerative)", "DBSCAN"],
        index=0
    )
    
    scaler_type = st.selectbox(
        "Feature Scaler:",
        ["StandardScaler", "RobustScaler", "MinMaxScaler"],
        index=0
    )
    
    if algorithm in ["K-Means", "Hierarchical (Agglomerative)"]:
        n_clusters = st.slider("Number of Clusters (k):", min_value=2, max_value=8, value=4)
        eps = 0.5
        min_samples = 5
    else:
        n_clusters = 4
        eps = st.slider("DBSCAN Epsilon (eps):", min_value=0.1, max_value=2.0, value=0.6, step=0.1)
        min_samples = st.slider("DBSCAN Min Samples:", min_value=2, max_value=20, value=5)

# -------------------------------------------------------------
# THEME INJECTION & DYNAMIC STYLING
# -------------------------------------------------------------
if is_light:
    theme_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* 1. Base Variables & Page Canvas */
        :root {
            --primary-color: #4F46E5 !important;
            --background-color: #F8FAFC !important;
            --secondary-background-color: #FFFFFF !important;
            --text-color: #0F172A !important;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #F8FAFC !important;
            color: #0F172A !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(248, 250, 252, 0.9) !important;
            backdrop-filter: blur(8px) !important;
        }
        
        /* 2. Sidebar */
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"] {
            background-color: #FFFFFF !important;
            border-right: 1px solid #E2E8F0 !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: #1E293B !important;
        }
        [data-testid="stSidebar"] .stCaption {
            color: #64748B !important;
        }
        
        /* 3. Typography & Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #0F172A !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        [data-testid="stAppViewContainer"] > .main p,
        [data-testid="stAppViewContainer"] > .main li {
            color: #334155;
        }
        
        /* 4. Main Executive Header Banner */
        .main-header {
            background: linear-gradient(135deg, #1E1B4B 0%, #3730A3 50%, #2563EB 100%);
            padding: 24px 32px;
            border-radius: 16px;
            color: #FFFFFF !important;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(55, 48, 163, 0.25);
            border: 1px solid rgba(255, 255, 255, 0.15);
        }
        .main-header h1 {
            color: #FFFFFF !important;
        }
        .main-header p, .main-header span, .main-header strong, .main-header div {
            color: #E0E7FF !important;
        }
        
        /* 5. Metric KPI Cards */
        [data-testid="stMetric"] {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 14px !important;
            padding: 16px 20px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(0, 0, 0, 0.08) !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #64748B !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stMetricValue"] div {
            color: #0F172A !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricDelta"] svg {
            fill: currentColor !important;
        }
        
        /* 6. Selectbox & Multiselect */
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stMultiSelect"] label p {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
            color: #0F172A !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        }
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="select"] > div:focus-within {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
        }
        div[data-baseweb="select"] * {
            color: #0F172A !important;
        }
        div[data-baseweb="select"] svg {
            fill: #475569 !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        ul[role="listbox"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 10px !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1) !important;
        }
        li[role="option"] {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background-color: #EEF2FF !important;
            color: #4338CA !important;
        }
        span[data-baseweb="tag"] {
            background-color: #EEF2FF !important;
            border: 1px solid #C7D2FE !important;
            border-radius: 6px !important;
        }
        span[data-baseweb="tag"] span {
            color: #4338CA !important;
            font-weight: 600 !important;
        }
        span[data-baseweb="tag"] svg {
            fill: #4338CA !important;
        }
        
        /* 7. Sliders */
        div[data-testid="stSlider"] label p {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
            background-color: transparent !important;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background: #E2E8F0 !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {
            background: linear-gradient(90deg, #6366F1, #4F46E5) !important;
        }
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: #4F46E5 !important;
            border: 2.5px solid #FFFFFF !important;
            box-shadow: 0 2px 6px rgba(79, 70, 229, 0.4) !important;
            width: 18px !important;
            height: 18px !important;
        }
        div[data-testid="stSlider"] [data-testid="stTickBar"] div,
        div[data-testid="stSlider"] div[data-testid="stSliderThumbValue"] {
            color: #475569 !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
        }
        
        /* 8. Expanders */
        div[data-testid="stExpander"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04) !important;
            margin-bottom: 14px !important;
            overflow: hidden !important;
        }
        div[data-testid="stExpander"] details {
            background-color: #FFFFFF !important;
        }
        div[data-testid="stExpander"] summary {
            background-color: #F8FAFC !important;
            border-bottom: 1px solid #E2E8F0 !important;
            color: #0F172A !important;
            font-weight: 700 !important;
            padding: 12px 18px !important;
        }
        div[data-testid="stExpander"] summary:hover {
            background-color: #EEF2FF !important;
            color: #4338CA !important;
        }
        div[data-testid="stExpander"] summary svg {
            fill: #475569 !important;
        }
        div[data-testid="stExpanderDetails"] {
            background-color: #FFFFFF !important;
            padding: 16px 20px !important;
        }
        
        /* 9. Radio Buttons & Modern Toggle Pills */
        div[data-testid="stRadio"] label p {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 8px !important;
            padding: 6px 14px !important;
            margin-right: 6px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03) !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            border-color: #6366F1 !important;
            background-color: #F8FAFC !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #1E293B !important;
            font-size: 0.88rem !important;
        }
        
        /* 10. Text Inputs */
        div[data-testid="stTextInput"] label p {
            color: #1E293B !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="input"] {
            background-color: #FFFFFF !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #6366F1 !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.12) !important;
        }
        div[data-baseweb="input"] input {
            background-color: transparent !important;
            color: #0F172A !important;
            font-size: 0.95rem !important;
        }
        div[data-baseweb="input"] input::placeholder {
            color: #94A3B8 !important;
        }
        
        /* 11. Buttons (Standard & Download) */
        div[data-testid="stButton"] button {
            background-color: #FFFFFF !important;
            color: #1E293B !important;
            border: 1px solid #CBD5E1 !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
            padding: 8px 16px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #EEF2FF !important;
            border-color: #6366F1 !important;
            color: #4F46E5 !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 10px rgba(99, 102, 241, 0.15) !important;
        }
        div[data-testid="stDownloadButton"] button {
            background: linear-gradient(135deg, #4F46E5 0%, #3B82F6 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.25) !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stDownloadButton"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px rgba(79, 70, 229, 0.35) !important;
        }
        div[data-testid="stDownloadButton"] button * {
            color: #FFFFFF !important;
        }
        
        /* 12. Alert Boxes (Info, Success, Warning) */
        div[data-testid="stAlert"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            margin-bottom: 14px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.03) !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconInfo"]),
        div[data-testid="stAlert"]:has(svg[data-testid="stAlertIconInfo"]) {
            background-color: #EFF6FF !important;
            border-left: 5px solid #3B82F6 !important;
            border-color: #BFDBFE !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconInfo"]) p,
        div[data-testid="stAlert"]:has([data-testid="stAlertIconInfo"]) strong {
            color: #1E40AF !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconSuccess"]),
        div[data-testid="stAlert"]:has(svg[data-testid="stAlertIconSuccess"]) {
            background-color: #ECFDF5 !important;
            border-left: 5px solid #10B981 !important;
            border-color: #A7F3D0 !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconSuccess"]) p,
        div[data-testid="stAlert"]:has([data-testid="stAlertIconSuccess"]) strong {
            color: #065F46 !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconWarning"]),
        div[data-testid="stAlert"]:has(svg[data-testid="stAlertIconWarning"]) {
            background-color: #FFFBEB !important;
            border-left: 5px solid #F59E0B !important;
            border-color: #FDE68A !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconWarning"]) p,
        div[data-testid="stAlert"]:has([data-testid="stAlertIconWarning"]) strong {
            color: #92400E !important;
        }
        div[data-testid="stAlert"] p {
            color: #1E293B !important;
            font-size: 0.92rem !important;
        }
        
        /* 13. Inline Code & Badges */
        code {
            background-color: #EEF2FF !important;
            color: #4338CA !important;
            border: 1px solid #C7D2FE !important;
            padding: 2px 6px !important;
            border-radius: 6px !important;
            font-size: 0.85em !important;
            font-weight: 600 !important;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        /* 14. Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 2px solid #E2E8F0 !important;
            gap: 6px !important;
            padding-bottom: 2px !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #64748B !important;
            font-weight: 600 !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 18px !important;
            border: 1px solid transparent !important;
            background-color: #FFFFFF !important;
            transition: all 0.2s ease !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #4F46E5 !important;
            background-color: #F8FAFC !important;
        }
        .stTabs [aria-selected="true"] {
            color: #4F46E5 !important;
            border: 1px solid #E2E8F0 !important;
            border-bottom: 3px solid #4F46E5 !important;
            background-color: #FFFFFF !important;
            box-shadow: 0 -2px 6px rgba(0, 0, 0, 0.02) !important;
        }
        
        /* 15. File Uploader */
        div[data-testid="stFileUploader"] section {
            background-color: #FFFFFF !important;
            border: 2px dashed #CBD5E1 !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: #6366F1 !important;
            background-color: #F8FAFC !important;
        }
        div[data-testid="stFileUploader"] section small,
        div[data-testid="stFileUploader"] section span {
            color: #64748B !important;
        }
        div[data-testid="stFileUploader"] section button {
            background-color: #6366F1 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        
        /* 16. DataFrames */
        div[data-testid="stDataFrame"] {
            background-color: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 12px !important;
            padding: 6px !important;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04) !important;
        }
        
        /* 17. Persona Cards */
        .persona-card {
            background: #FFFFFF !important;
            color: #1E293B !important;
            border-radius: 16px !important;
            padding: 22px !important;
            margin-bottom: 18px !important;
            border: 1px solid #E2E8F0 !important;
            border-left: 6px solid #6366F1 !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05) !important;
            transition: transform 0.2s ease !important;
        }
        .persona-card:hover {
            transform: translateY(-2px) !important;
        }
        .persona-title {
            margin: 10px 0 6px 0;
            font-size: 1.15rem;
            font-weight: 700;
            color: #0F172A !important;
        }
        .persona-desc {
            color: #64748B !important;
            font-size: 0.88rem;
            min-height: 55px;
            margin-bottom: 12px;
        }
        .persona-stat-box {
            margin-bottom: 12px;
            background: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
            padding: 10px;
            border-radius: 8px;
        }
        .persona-stat-label {
            font-size: 0.8rem;
            color: #64748B !important;
        }
        .persona-stat-val {
            font-size: 1.1rem;
            color: #0F172A !important;
        }
        .persona-stat-pct {
            font-size: 0.8rem;
            color: #64748B !important;
        }
        .persona-traits-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #475569 !important;
            margin-bottom: 4px;
        }
        .persona-traits-list {
            margin: 0;
            padding-left: 18px;
            font-size: 0.82rem;
            color: #334155 !important;
        }
        
        /* 18. Customer 360 Card */
        .customer-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-radius: 14px !important;
            padding: 20px !important;
            box-shadow: 0 4px 10px rgba(0,0,0,0.05) !important;
            color: #1E293B !important;
        }
        .customer-title {
            margin: 4px 0;
            color: #0F172A !important;
        }
        .customer-divider {
            margin: 14px 0;
            border: none;
            border-top: 1px solid #E2E8F0 !important;
        }
        .customer-next-label {
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 6px;
            color: #475569 !important;
        }
        .customer-campaign-box {
            background: #F8FAFC !important;
            border: 1px solid #E2E8F0 !important;
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #1E293B !important;
        }
        
        /* 19. Natural Language Query Card */
        .query-card {
            background: #FFFFFF !important;
            border: 1px solid #E2E8F0 !important;
            border-left: 5px solid #3B82F6 !important;
            border-radius: 12px !important;
            padding: 18px !important;
            margin-top: 14px !important;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05) !important;
            color: #0F172A !important;
        }
        .query-header {
            font-size: 0.8rem;
            font-weight: 700;
            color: #2563EB !important;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .query-body {
            font-size: 1.05rem;
            line-height: 1.6;
            color: #1E293B !important;
        }
        
        hr {
            border-color: #E2E8F0 !important;
            margin: 20px 0 !important;
        }
    </style>
    """
else:
    theme_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        
        /* 1. Base Variables & Page Canvas */
        :root {
            --primary-color: #6366F1 !important;
            --background-color: #0B0F19 !important;
            --secondary-background-color: #141B2D !important;
            --text-color: #F8FAFC !important;
        }
        .stApp, [data-testid="stAppViewContainer"] {
            background-color: #0B0F19 !important;
            color: #F8FAFC !important;
        }
        [data-testid="stHeader"] {
            background-color: rgba(11, 15, 25, 0.9) !important;
            backdrop-filter: blur(8px) !important;
        }
        
        /* 2. Sidebar */
        [data-testid="stSidebar"],
        [data-testid="stSidebarContent"],
        [data-testid="stSidebarUserContent"] {
            background-color: #111827 !important;
            border-right: 1px solid #1F2937 !important;
        }
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] h4, [data-testid="stSidebar"] label, [data-testid="stSidebar"] p {
            color: #F1F5F9 !important;
        }
        [data-testid="stSidebar"] .stCaption {
            color: #94A3B8 !important;
        }
        
        /* 3. Typography & Headings */
        h1, h2, h3, h4, h5, h6 {
            color: #F8FAFC !important;
            font-family: 'Plus Jakarta Sans', sans-serif;
        }
        [data-testid="stAppViewContainer"] > .main p,
        [data-testid="stAppViewContainer"] > .main li {
            color: #CBD5E1;
        }
        
        /* 4. Main Executive Header Banner */
        .main-header {
            background: linear-gradient(135deg, #1E1E2F 0%, #2D1B4E 50%, #111827 100%);
            padding: 24px 32px;
            border-radius: 16px;
            color: white;
            margin-bottom: 24px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .main-header h1 {
            color: #FFFFFF !important;
        }
        .main-header p, .main-header span, .main-header strong, .main-header div {
            color: #CBD5E1 !important;
        }
        
        /* 5. Metric KPI Cards */
        [data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.04) !important;
            border: 1px solid rgba(255, 255, 255, 0.08) !important;
            border-radius: 14px !important;
            padding: 16px 20px !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2) !important;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        [data-testid="stMetric"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3) !important;
        }
        [data-testid="stMetricLabel"] p {
            color: #94A3B8 !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
        }
        [data-testid="stMetricValue"] div {
            color: #F8FAFC !important;
            font-weight: 800 !important;
        }
        [data-testid="stMetricDelta"] svg {
            fill: currentColor !important;
        }
        
        /* 6. Selectbox & Multiselect */
        div[data-testid="stSelectbox"] label p,
        div[data-testid="stMultiSelect"] label p {
            color: #F1F5F9 !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="select"] > div {
            background-color: #182032 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 10px !important;
            color: #F8FAFC !important;
        }
        div[data-baseweb="select"] > div:hover,
        div[data-baseweb="select"] > div:focus-within {
            border-color: #818CF8 !important;
            box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
        }
        div[data-baseweb="select"] * {
            color: #F8FAFC !important;
        }
        div[data-baseweb="select"] svg {
            fill: #94A3B8 !important;
        }
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] > div,
        ul[role="listbox"] {
            background-color: #182032 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 10px !important;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4) !important;
        }
        li[role="option"] {
            background-color: #182032 !important;
            color: #F1F5F9 !important;
        }
        li[role="option"]:hover,
        li[role="option"][aria-selected="true"] {
            background-color: #25304B !important;
            color: #A5B4FC !important;
        }
        span[data-baseweb="tag"] {
            background-color: #25304B !important;
            border: 1px solid #3B4261 !important;
            border-radius: 6px !important;
        }
        span[data-baseweb="tag"] span {
            color: #A5B4FC !important;
            font-weight: 600 !important;
        }
        span[data-baseweb="tag"] svg {
            fill: #A5B4FC !important;
        }
        
        /* 7. Sliders */
        div[data-testid="stSlider"] label p {
            color: #F1F5F9 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div {
            background-color: transparent !important;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background: #1E293B !important;
            height: 6px !important;
            border-radius: 3px !important;
        }
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div > div {
            background: linear-gradient(90deg, #6366F1, #818CF8) !important;
        }
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: #818CF8 !important;
            border: 2.5px solid #0B0F19 !important;
            box-shadow: 0 2px 6px rgba(129, 140, 248, 0.4) !important;
            width: 18px !important;
            height: 18px !important;
        }
        div[data-testid="stSlider"] [data-testid="stTickBar"] div,
        div[data-testid="stSlider"] div[data-testid="stSliderThumbValue"] {
            color: #94A3B8 !important;
            font-size: 0.8rem !important;
            font-weight: 600 !important;
        }
        
        /* 8. Expanders */
        div[data-testid="stExpander"] {
            background-color: #141B2D !important;
            border: 1px solid #1E293B !important;
            border-radius: 12px !important;
            margin-bottom: 14px !important;
            overflow: hidden !important;
        }
        div[data-testid="stExpander"] details {
            background-color: #141B2D !important;
        }
        div[data-testid="stExpander"] summary {
            background-color: #182032 !important;
            border-bottom: 1px solid #1E293B !important;
            color: #F8FAFC !important;
            font-weight: 700 !important;
            padding: 12px 18px !important;
        }
        div[data-testid="stExpander"] summary:hover {
            background-color: #25304B !important;
            color: #A5B4FC !important;
        }
        div[data-testid="stExpander"] summary svg {
            fill: #94A3B8 !important;
        }
        div[data-testid="stExpanderDetails"] {
            background-color: #141B2D !important;
            padding: 16px 20px !important;
        }
        
        /* 9. Radio Buttons & Modern Toggle Pills */
        div[data-testid="stRadio"] label p {
            color: #F1F5F9 !important;
            font-weight: 600 !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background-color: #182032 !important;
            border: 1px solid #1E293B !important;
            border-radius: 8px !important;
            padding: 6px 14px !important;
            margin-right: 6px !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            border-color: #818CF8 !important;
            background-color: #25304B !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #F1F5F9 !important;
            font-size: 0.88rem !important;
        }
        
        /* 10. Text Inputs */
        div[data-testid="stTextInput"] label p {
            color: #F1F5F9 !important;
            font-weight: 600 !important;
        }
        div[data-baseweb="input"] {
            background-color: #182032 !important;
            border: 1px solid #2D3748 !important;
            border-radius: 10px !important;
        }
        div[data-baseweb="input"]:focus-within {
            border-color: #818CF8 !important;
            box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
        }
        div[data-baseweb="input"] input {
            background-color: transparent !important;
            color: #F8FAFC !important;
            font-size: 0.95rem !important;
        }
        div[data-baseweb="input"] input::placeholder {
            color: #64748B !important;
        }
        
        /* 11. Buttons (Standard & Download) */
        div[data-testid="stButton"] button {
            background-color: #182032 !important;
            color: #F8FAFC !important;
            border: 1px solid #1E293B !important;
            border-radius: 10px !important;
            font-weight: 600 !important;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
            padding: 8px 16px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stButton"] button:hover {
            background-color: #25304B !important;
            border-color: #818CF8 !important;
            color: #A5B4FC !important;
            transform: translateY(-1px) !important;
            box-shadow: 0 4px 10px rgba(129, 140, 248, 0.2) !important;
        }
        div[data-testid="stDownloadButton"] button {
            background: linear-gradient(135deg, #6366F1 0%, #3B82F6 100%) !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 10px !important;
            font-weight: 700 !important;
            box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3) !important;
            padding: 10px 20px !important;
            transition: all 0.2s ease !important;
        }
        div[data-testid="stDownloadButton"] button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 16px rgba(99, 102, 241, 0.4) !important;
        }
        div[data-testid="stDownloadButton"] button * {
            color: #FFFFFF !important;
        }
        
        /* 12. Alert Boxes */
        div[data-testid="stAlert"] {
            background-color: #141B2D !important;
            border: 1px solid #1E293B !important;
            border-radius: 12px !important;
            padding: 14px 18px !important;
            margin-bottom: 14px !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconInfo"]),
        div[data-testid="stAlert"]:has(svg[data-testid="stAlertIconInfo"]) {
            background-color: #132338 !important;
            border-left: 5px solid #3B82F6 !important;
            border-color: #1E3A5F !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconInfo"]) p,
        div[data-testid="stAlert"]:has([data-testid="stAlertIconInfo"]) strong {
            color: #93C5FD !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconSuccess"]),
        div[data-testid="stAlert"]:has(svg[data-testid="stAlertIconSuccess"]) {
            background-color: #0F2D24 !important;
            border-left: 5px solid #10B981 !important;
            border-color: #134E3A !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconSuccess"]) p,
        div[data-testid="stAlert"]:has([data-testid="stAlertIconSuccess"]) strong {
            color: #6EE7B7 !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconWarning"]),
        div[data-testid="stAlert"]:has(svg[data-testid="stAlertIconWarning"]) {
            background-color: #2D2012 !important;
            border-left: 5px solid #F59E0B !important;
            border-color: #4D3316 !important;
        }
        div[data-testid="stAlert"]:has([data-testid="stAlertIconWarning"]) p,
        div[data-testid="stAlert"]:has([data-testid="stAlertIconWarning"]) strong {
            color: #FDE68A !important;
        }
        div[data-testid="stAlert"] p {
            color: #F1F5F9 !important;
            font-size: 0.92rem !important;
        }
        
        /* 13. Inline Code & Badges */
        code {
            background-color: #1E293B !important;
            color: #A5B4FC !important;
            border: 1px solid #334155 !important;
            padding: 2px 6px !important;
            border-radius: 6px !important;
            font-size: 0.85em !important;
            font-weight: 600 !important;
        }
        .badge {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }
        
        /* 14. Tabs */
        .stTabs [data-baseweb="tab-list"] {
            background-color: transparent !important;
            border-bottom: 2px solid #1E293B !important;
            gap: 6px !important;
            padding-bottom: 2px !important;
        }
        .stTabs [data-baseweb="tab"] {
            color: #94A3B8 !important;
            font-weight: 600 !important;
            border-radius: 8px 8px 0 0 !important;
            padding: 10px 18px !important;
            border: 1px solid transparent !important;
            background-color: #141B2D !important;
            transition: all 0.2s ease !important;
        }
        .stTabs [data-baseweb="tab"]:hover {
            color: #818CF8 !important;
            background-color: #182032 !important;
        }
        .stTabs [aria-selected="true"] {
            color: #818CF8 !important;
            border: 1px solid #1E293B !important;
            border-bottom: 3px solid #818CF8 !important;
            background-color: #182032 !important;
            box-shadow: 0 -2px 6px rgba(0, 0, 0, 0.2) !important;
        }
        
        /* 15. File Uploader */
        div[data-testid="stFileUploader"] section {
            background-color: #182032 !important;
            border: 2px dashed #2D3748 !important;
            border-radius: 12px !important;
            padding: 20px !important;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: #818CF8 !important;
            background-color: #25304B !important;
        }
        div[data-testid="stFileUploader"] section small,
        div[data-testid="stFileUploader"] section span {
            color: #94A3B8 !important;
        }
        div[data-testid="stFileUploader"] section button {
            background-color: #6366F1 !important;
            color: #FFFFFF !important;
            border: none !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }
        
        /* 16. DataFrames */
        div[data-testid="stDataFrame"] {
            background-color: #182032 !important;
            border: 1px solid #1E293B !important;
            border-radius: 12px !important;
            padding: 6px !important;
        }
        
        /* 17. Persona Cards */
        .persona-card {
            background: #182032;
            color: #F8FAFC;
            border-radius: 16px;
            padding: 22px;
            margin-bottom: 18px;
            border: 1px solid #1E293B;
            border-left: 6px solid #6366F1;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
            transition: transform 0.2s ease;
        }
        .persona-card:hover {
            transform: translateY(-2px);
        }
        .persona-title {
            margin: 10px 0 6px 0;
            font-size: 1.15rem;
            font-weight: 700;
            color: #F8FAFC;
        }
        .persona-desc {
            color: #94A3B8;
            font-size: 0.88rem;
            min-height: 55px;
            margin-bottom: 12px;
        }
        .persona-stat-box {
            margin-bottom: 12px;
            background: #0F172A;
            border: 1px solid #1E293B;
            padding: 10px;
            border-radius: 8px;
        }
        .persona-stat-label {
            font-size: 0.8rem;
            color: #94A3B8;
        }
        .persona-stat-val {
            font-size: 1.1rem;
            color: #F1F5F9;
        }
        .persona-stat-pct {
            font-size: 0.8rem;
            color: #94A3B8;
        }
        .persona-traits-title {
            font-size: 0.8rem;
            font-weight: 600;
            color: #94A3B8;
            margin-bottom: 4px;
        }
        .persona-traits-list {
            margin: 0;
            padding-left: 18px;
            font-size: 0.82rem;
            color: #CBD5E1;
        }
        
        /* 18. Customer 360 Card */
        .customer-card {
            background: #182032;
            border: 1px solid #1E293B;
            border-radius: 14px;
            padding: 20px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            color: #F8FAFC;
        }
        .customer-title {
            margin: 4px 0;
            color: #FFFFFF;
        }
        .customer-divider {
            margin: 14px 0;
            border: none;
            border-top: 1px solid #1E293B;
        }
        .customer-next-label {
            font-weight: 600;
            font-size: 0.9rem;
            margin-bottom: 6px;
            color: #CBD5E1;
        }
        .customer-campaign-box {
            background: #0F172A;
            border: 1px solid #1E293B;
            padding: 10px 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            color: #CBD5E1;
        }
        
        /* 19. Natural Language Query Card */
        .query-card {
            background: #182032;
            border: 1px solid #1E293B;
            border-left: 5px solid #3B82F6;
            border-radius: 12px;
            padding: 18px;
            margin-top: 14px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.3);
            color: #F8FAFC;
        }
        .query-header {
            font-size: 0.8rem;
            font-weight: 700;
            color: #60A5FA;
            text-transform: uppercase;
            margin-bottom: 6px;
        }
        .query-body {
            font-size: 1.05rem;
            line-height: 1.6;
            color: #F1F5F9;
        }
        
        hr {
            border-color: #1E293B !important;
            margin: 20px 0 !important;
        }
    </style>
    """

st.markdown(theme_css, unsafe_allow_html=True)

# -------------------------------------------------------------
# DATA LOADING & PREPARATION
# -------------------------------------------------------------
@st.cache_data
def get_dataset(source_name: str, uploaded_bytes=None) -> pd.DataFrame:
    if source_name in ["Online Retail (RFM)", "Mall Customers"]:
        return load_preloaded_dataset(source_name)
    elif uploaded_bytes is not None:
        try:
            return pd.read_csv(uploaded_bytes)
        except Exception:
            uploaded_bytes.seek(0)
            return pd.read_excel(uploaded_bytes)
    return pd.DataFrame()

if data_source == "Upload Custom CSV" and uploaded_file is None:
    st.info("👆 Please upload a CSV dataset or switch to one of the preloaded datasets (Online Retail RFM / Mall Customers) from the left sidebar.")
    st.stop()

df_raw = get_dataset(data_source, uploaded_file)
if df_raw.empty:
    st.error("Dataset could not be loaded. Please verify file format.")
    st.stop()

summary_info = inspect_dataset(df_raw)

# Identify default features and ID column
if data_source == "Online Retail (RFM)":
    default_features = ["Recency", "Frequency", "Monetary"]
    default_id = "CustomerID"
elif data_source == "Mall Customers":
    default_features = ["Annual Income (k$)", "Spending Score (1-100)"]
    default_id = "CustomerID"
else:
    default_features = [c for c in summary_info["numeric_cols"] if c not in summary_info["id_candidates"]][:3]
    default_id = summary_info["id_candidates"][0] if summary_info["id_candidates"] else df_raw.columns[0]

# Feature selector in expander if user wants custom feature subset
with st.sidebar.expander("🛠️ Feature Engineering & Selection", expanded=False):
    selected_features = st.multiselect(
        "Clustering Features:",
        options=summary_info["numeric_cols"],
        default=[f for f in default_features if f in summary_info["numeric_cols"]]
    )
    selected_id_col = st.selectbox(
        "Customer ID Column:",
        options=df_raw.columns.tolist(),
        index=df_raw.columns.tolist().index(default_id) if default_id in df_raw.columns else 0
    )

if len(selected_features) < 2:
    st.warning("⚠️ Please select at least 2 numerical features for clustering.")
    st.stop()

# Clean and scale features
df_clean, X_df = clean_and_prepare_features(df_raw, selected_features, id_col=selected_id_col)
X_scaled, scaler_obj = scale_features(X_df, scaler_type=scaler_type)

# Run clustering model
cluster_labels, cluster_model, eval_metrics = fit_clustering_model(
    algorithm=algorithm,
    X_scaled=X_scaled,
    n_clusters=n_clusters,
    eps=eps,
    min_samples=min_samples
)

df_clean["Cluster"] = cluster_labels

# Automated Segment Profiling
segment_profile_results = profile_clusters(
    df=df_clean,
    labels=cluster_labels,
    feature_cols=selected_features,
    dataset_name=data_source
)
cluster_profiles = segment_profile_results["cluster_profiles"]

# PCA coordinates for visualization
pca_results = apply_pca_reduction(X_scaled)

# -------------------------------------------------------------
# MAIN APP HEADER & EXECUTIVE BANNER
# -------------------------------------------------------------
st.markdown(f"""
<div class="main-header">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
        <div>
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 6px;">
                <span style="background: rgba(99, 102, 241, 0.35); color: #E0E7FF; padding: 4px 12px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">
                    AIML Mini-Project • Unsupervised Learning
                </span>
                <span style="background: {'rgba(251, 191, 36, 0.25)' if is_light else 'rgba(96, 165, 250, 0.2)'}; color: {'#FDE68A' if is_light else '#BFDBFE'}; padding: 4px 12px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">
                    {'☀️ Light Mode' if is_light else '🌙 Dark Mode'}
                </span>
            </div>
            <h1 style="margin: 4px 0 4px 0; font-size: 2rem; font-weight: 800; color: #FFFFFF;">
                AI-Based Customer Segmentation System
            </h1>
            <p style="margin: 0; color: #E0E7FF; font-size: 0.95rem;">
                Active Dataset: <strong>{data_source}</strong> &nbsp;|&nbsp; Algorithm: <strong>{algorithm}</strong> &nbsp;|&nbsp; Features: <strong>{", ".join(selected_features)}</strong>
            </p>
        </div>
        <div style="text-align: right;">
            <span style="font-size: 2rem; font-weight: 800; color: #38BDF8;">{len(cluster_profiles)}</span>
            <div style="font-size: 0.8rem; color: #CBD5E1; text-transform: uppercase; font-weight: 600;">Discovered Segments</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPI Metric Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
with kpi1:
    st.metric(label="👥 Total Customers", value=f"{len(df_clean):,}")
with kpi2:
    sil = eval_metrics.get("silhouette")
    st.metric(label="🎯 Silhouette Score", value=f"{sil:.3f}" if sil is not None else "N/A", delta="Quality Metric" if sil and sil > 0.4 else None)
with kpi3:
    db = eval_metrics.get("davies_bouldin")
    st.metric(label="📐 Davies-Bouldin Index", value=f"{db:.3f}" if db is not None else "N/A", delta="(Lower is better)" if db else None)
with kpi4:
    if "Monetary" in df_clean.columns:
        tot_rev = df_clean["Monetary"].sum()
        st.metric(label="💰 Total Dataset Revenue", value=f"${tot_rev:,.0f}")
    elif "Spending Score (1-100)" in df_clean.columns:
        avg_spend = df_clean["Spending Score (1-100)"].mean()
        st.metric(label="💳 Average Spending Score", value=f"{avg_spend:.1f} / 100")
    else:
        st.metric(label="⚙️ Features Analyzed", value=len(selected_features))

st.write("")

# -------------------------------------------------------------
# APPLICATION TABS
# -------------------------------------------------------------
tabs = st.tabs([
    "📊 Dataset & EDA",
    "🧠 Model Evaluation (Elbow/Silhouette)",
    "📈 Interactive 2D/3D Visualizer",
    "🏷️ Automated Segment Personas",
    "🎯 Targeted Marketing Playbooks",
    "🔍 Customer 360° Lookup",
    "💬 Smart Natural Language Query",
    "📥 Export Center"
])

# -------------------------------------------------------------
# TAB 1: DATASET & EDA
# -------------------------------------------------------------
with tabs[0]:
    st.subheader("📋 Dataset Inspection & Preprocessing")
    
    col_d1, col_d2 = st.columns([3, 2])
    with col_d1:
        st.markdown(f"**Data Preview (First 5 Rows of {len(df_clean):,} records):**")
        st.dataframe(df_clean.head(10), use_container_width=True)
    with col_d2:
        st.markdown("**Data Integrity Summary:**")
        integrity_df = pd.DataFrame({
            "Metric": ["Total Rows", "Total Columns", "Missing Values Imputed", "Duplicate Rows Removed", "Memory Size"],
            "Value": [
                f"{summary_info['rows']:,}",
                f"{summary_info['columns']}",
                f"{summary_info['missing_count']}",
                f"{summary_info['duplicate_count']}",
                f"{summary_info['memory_mb']} MB"
            ]
        })
        st.dataframe(integrity_df, hide_index=True, use_container_width=True)
        
    st.markdown("---")
    st.subheader("📊 Exploratory Feature Distributions")
    f_cols = st.columns(len(selected_features))
    for idx, feat in enumerate(selected_features):
        with f_cols[idx]:
            fig_hist = px.histogram(
                df_clean, 
                x=feat, 
                marginal="box", 
                title=f"{feat} Distribution",
                color_discrete_sequence=["#6366F1"],
                nbins=25,
                template=plotly_template
            )
            fig_hist.update_layout(height=280, margin=dict(l=10, r=10, t=35, b=10), template=plotly_template)
            st.plotly_chart(fig_hist, use_container_width=True)

# -------------------------------------------------------------
# TAB 2: MODEL EVALUATION & ELBOW METHOD
# -------------------------------------------------------------
with tabs[1]:
    st.subheader("🧠 Mathematical Cluster Evaluation & Optimal K")
    st.markdown("""
    In unsupervised machine learning, choosing the optimal number of clusters ($k$) requires evaluating mathematical compactness and separation.
    - **Elbow Method (Inertia/WCSS)**: Identifies point of diminishing returns where adding another cluster yields marginal variance reduction.
    - **Silhouette Score**: Measures how similar a customer is to their own cluster compared to neighboring clusters (ranges from -1 to +1, higher is better).
    """)
    
    eval_curves = compute_elbow_and_silhouette(X_scaled, k_min=2, k_max=8)
    
    col_ev1, col_ev2 = st.columns(2)
    with col_ev1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=eval_curves["k_values"],
            y=eval_curves["wcss"],
            mode="lines+markers",
            name="WCSS (Inertia)",
            line=dict(color="#3B82F6", width=3),
            marker=dict(size=8, color="#1D4ED8")
        ))
        fig_elbow.add_vline(
            x=n_clusters, 
            line_dash="dash", 
            line_color="#EF4444", 
            annotation_text=f"Current k={n_clusters}",
            annotation_font_color="#0F172A" if is_light else "#F8FAFC"
        )
        fig_elbow.update_layout(
            title="Elbow Method Curve (WCSS vs k)",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Within-Cluster Sum of Squares (WCSS)",
            height=380,
            template=plotly_template
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
        
    with col_ev2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=eval_curves["k_values"],
            y=eval_curves["silhouette_scores"],
            name="Silhouette Score",
            marker_color=["#10B981" if k == eval_curves["best_k_silhouette"] else ("#94A3B8" if is_light else "#475569") for k in eval_curves["k_values"]]
        ))
        fig_sil.update_layout(
            title=f"Silhouette Analysis by k (Optimal peak at k = {eval_curves['best_k_silhouette']})",
            xaxis_title="Number of Clusters (k)",
            yaxis_title="Average Silhouette Score",
            height=380,
            template=plotly_template
        )
        st.plotly_chart(fig_sil, use_container_width=True)
        
    st.success(f"💡 **Algorithmic Recommendation:** Based on silhouette maximization, the mathematical optimal cluster count is **k = {eval_curves['best_k_silhouette']}**.")

# -------------------------------------------------------------
# TAB 3: INTERACTIVE 2D & 3D VISUALIZATION
# -------------------------------------------------------------
with tabs[2]:
    st.subheader("📈 Multi-Dimensional Customer Visualizer")
    
    vis_mode = st.radio("Visualization Perspective:", ["2D Scatter & Centroids", "3D Interactive Space", "PCA Projections", "Segment Radar Benchmark"], horizontal=True)
    
    # Map cluster names to dataframe for charts
    cluster_name_map = {c: f"{p['icon']} {p['persona_name']}" for c, p in cluster_profiles.items()}
    df_clean["Segment Name"] = df_clean["Cluster"].map(cluster_name_map).fillna("Noise / Outlier")
    
    if vis_mode == "2D Scatter & Centroids":
        col_v1, col_v2 = st.columns(2)
        with col_v1:
            x_axis = st.selectbox("X-Axis Feature:", options=selected_features, index=0)
        with col_v2:
            y_axis = st.selectbox("Y-Axis Feature:", options=selected_features, index=min(1, len(selected_features)-1))
            
        fig_2d = px.scatter(
            df_clean,
            x=x_axis,
            y=y_axis,
            color="Segment Name",
            hover_data=[selected_id_col] + selected_features,
            title=f"Customer Segments: {x_axis} vs {y_axis}",
            template=plotly_template,
            height=550
        )
        fig_2d.update_traces(marker=dict(size=8, opacity=0.85, line=dict(width=0.5, color="DarkSlateGrey" if is_light else "#1E293B")))
        st.plotly_chart(fig_2d, use_container_width=True)
        
    elif vis_mode == "3D Interactive Space":
        if len(selected_features) >= 3:
            fig_3d = px.scatter_3d(
                df_clean,
                x=selected_features[0],
                y=selected_features[1],
                z=selected_features[2],
                color="Segment Name",
                hover_data=[selected_id_col] + selected_features,
                title=f"3D Cluster Space: {selected_features[0]} × {selected_features[1]} × {selected_features[2]}",
                template=plotly_template,
                height=650
            )
            fig_3d.update_traces(marker=dict(size=4.5, opacity=0.8))
            st.plotly_chart(fig_3d, use_container_width=True)
        else:
            st.info("3D scatter plot requires at least 3 numerical features. Please select an additional feature or use PCA Projections.")
            
    elif vis_mode == "PCA Projections":
        st.write(f"Principal Component Analysis (PCA) combines multi-feature variance: PC1 explains **{pca_results['var_2d'][0]}%** and PC2 explains **{pca_results['var_2d'][1]}%**.")
        pca_df = pd.DataFrame({
            "PC1": pca_results["coords_2d"][:, 0],
            "PC2": pca_results["coords_2d"][:, 1],
            "Segment Name": df_clean["Segment Name"],
            "CustomerID": df_clean[selected_id_col]
        })
        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color="Segment Name",
            hover_data=["CustomerID"],
            title="PCA 2D Cluster Separation Projection",
            template=plotly_template,
            height=550
        )
        st.plotly_chart(fig_pca, use_container_width=True)
        
    elif vis_mode == "Segment Radar Benchmark":
        st.write("Normalized spider chart comparing relative strength across all features (0 = minimum, 1 = maximum):")
        # Normalize cluster means between 0 and 1 for radar chart
        radar_categories = selected_features
        fig_radar = go.Figure()
        
        for c, prof in cluster_profiles.items():
            means = [prof["feature_means"][feat] for feat in selected_features]
            # scale min-max using global min-max
            norm_means = []
            for idx, feat in enumerate(selected_features):
                f_min = df_clean[feat].min()
                f_max = df_clean[feat].max()
                scaled_val = (means[idx] - f_min) / (f_max - f_min + 1e-5)
                norm_means.append(round(scaled_val, 2))
                
            fig_radar.add_trace(go.Scatterpolar(
                r=norm_means + [norm_means[0]],
                theta=radar_categories + [radar_categories[0]],
                fill='toself',
                name=f"{prof['icon']} {prof['persona_name']}"
            ))
            
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500,
            title="Relative Segment Behavioral Radar",
            template=plotly_template
        )
        st.plotly_chart(fig_radar, use_container_width=True)

# -------------------------------------------------------------
# TAB 4: AUTOMATED SEGMENT PERSONAS
# -------------------------------------------------------------
with tabs[3]:
    st.subheader("🏷️ Automated Segment Discovery & Personas")
    st.markdown("Machine learning identified the following distinct customer groups, automatically named from statistical distributions:")
    
    cols = st.columns(min(len(cluster_profiles), 4))
    for idx, (c, prof) in enumerate(cluster_profiles.items()):
        col_idx = idx % len(cols)
        with cols[col_idx]:
            st.markdown(f"""
            <div class="persona-card" style="border-left-color: {prof['badge_color']};">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 1.8rem;">{prof['icon']}</span>
                    <span class="badge" style="background: {prof['badge_color']}20; color: {prof['badge_color']};">
                        Cluster {c}
                    </span>
                </div>
                <h3 class="persona-title">
                    {prof['persona_name']}
                </h3>
                <p class="persona-desc">
                    {prof['description']}
                </p>
                <div class="persona-stat-box">
                    <div class="persona-stat-label">Group Size:</div>
                    <strong class="persona-stat-val">{prof['customer_count']:,}</strong> 
                    <span class="persona-stat-pct">({prof['percentage']}%)</span>
                </div>
                <div class="persona-traits-title">Key Traits:</div>
                <ul class="persona-traits-list">
                    {"".join(f"<li>{trait}</li>" for trait in prof['key_traits'])}
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    st.subheader("📊 Statistical Cluster Comparison Table")
    summary_rows = []
    for c, p in cluster_profiles.items():
        row = {
            "Cluster": f"Cluster {c}",
            "Persona": f"{p['icon']} {p['persona_name']}",
            "Customers": p["customer_count"],
            "Share (%)": f"{p['percentage']}%"
        }
        for feat, val in p["feature_means"].items():
            row[f"Avg {feat}"] = f"{val:,.2f}"
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, use_container_width=True)

# -------------------------------------------------------------
# TAB 5: PERSONALIZED MARKETING PLAYBOOKS
# -------------------------------------------------------------
with tabs[4]:
    st.subheader("🎯 Targeted Marketing & Retention Playbooks")
    st.markdown("Tailored marketing strategies and activation blueprints customized for each cluster's spending and activity profile:")
    
    for c, prof in cluster_profiles.items():
        playbook = generate_segment_marketing_playbook(prof["persona_name"], prof["key_traits"])
        with st.expander(f"{prof['icon']} Segment {c}: {prof['persona_name']} ({prof['customer_count']:,} customers)", expanded=True):
            p1, p2, p3 = st.columns([1, 1, 1])
            with p1:
                st.markdown(f"**🎯 Campaign Objective:**\n{playbook['objective']}")
                st.markdown(f"**⚡ Action Priority:**\n`{playbook['action_priority']}`")
            with p2:
                st.markdown(f"**📢 Recommended Channels:**\n" + ", ".join([f"`{ch}`" for ch in playbook['recommended_channels']]))
                st.markdown(f"**⏱️ Communication Cadence:**\n{playbook['communication_frequency']}")
            with p3:
                st.markdown(f"**🏷️ Suggested Campaign Name:**\n*{playbook['campaign_title']}*")
                st.markdown(f"**🎁 Offer & Incentive:**\n{playbook['suggested_offer']}")
                
            st.info(f"💡 **Pricing & Discount Guideline:** {playbook['discount_strategy']}")

# -------------------------------------------------------------
# TAB 6: CUSTOMER 360° LOOKUP & SIMULATOR
# -------------------------------------------------------------
with tabs[5]:
    st.subheader("🔍 Individual Customer 360° Lookup & Strategy")
    st.markdown("Look up any individual customer by ID to view their assigned persona, metric deviations from the segment benchmark, and immediate next steps.")
    
    col_c1, col_c2 = st.columns([1, 2])
    with col_c1:
        sample_ids = df_clean[selected_id_col].astype(str).tolist()
        cust_search = st.selectbox("Select Customer ID:", options=sample_ids, index=0)
        
    cust_analysis = analyze_individual_customer(
        customer_id=cust_search,
        df_with_clusters=df_clean,
        cluster_profiles=cluster_profiles,
        feature_cols=selected_features,
        id_col=selected_id_col
    )
    
    if cust_analysis["found"]:
        col_res1, col_res2 = st.columns([1, 1.3])
        with col_res1:
            st.markdown(f"""
            <div class="customer-card">
                <div style="font-size: 2rem;">{cust_analysis['icon']}</div>
                <h3 class="customer-title">Customer #{cust_analysis['customer_id']}</h3>
                <span class="badge" style="background: {cust_analysis['badge_color']}25; color: {cust_analysis['badge_color']};">
                    {cust_analysis['persona_name']} (Cluster {cust_analysis['cluster_id']})
                </span>
                <hr class="customer-divider">
                <div class="customer-next-label">Next Recommended Campaign:</div>
                <div class="customer-campaign-box">
                    <strong>{cust_analysis['playbook']['campaign_title']}</strong><br>
                    {cust_analysis['playbook']['suggested_offer']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_res2:
            st.markdown(f"**Customer #{cust_analysis['customer_id']} vs Cluster Average:**")
            comp_df = pd.DataFrame(cust_analysis["comparisons"])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)
            
    st.markdown("---")
    st.subheader("🔮 Interactive 'What-If' Segment Migration Simulator")
    st.caption("Adjust hypothetical customer metrics to predict what segment they would shift into:")
    
    sim_cols = st.columns(len(selected_features))
    sim_inputs = {}
    for idx, feat in enumerate(selected_features):
        with sim_cols[idx]:
            min_v = float(df_clean[feat].min())
            max_v = float(df_clean[feat].max())
            mean_v = float(df_clean[feat].mean())
            sim_inputs[feat] = st.slider(f"Simulated {feat}:", min_value=min_v, max_value=max_v, value=mean_v)
            
    # Scale hypothetical customer vector
    sim_vector = pd.DataFrame([sim_inputs])
    sim_scaled = scaler_obj.transform(sim_vector)
    
    if algorithm == "K-Means":
        sim_cluster = int(cluster_model.predict(sim_scaled)[0])
        sim_persona = cluster_profiles.get(sim_cluster, {})
        st.success(f"🎯 **Predicted Cluster:** Customer migrates into **{sim_persona.get('icon', '🏷️')} {sim_persona.get('persona_name', f'Cluster {sim_cluster}')}**!")
    else:
        st.info("What-If instant prediction is natively optimized for K-Means centroid assignment.")

# -------------------------------------------------------------
# TAB 7: NATURAL LANGUAGE QUERY
# -------------------------------------------------------------
with tabs[6]:
    st.subheader("💬 Smart Natural-Language Analytics Queries")
    st.markdown("Ask direct business questions about your customer segments in plain English:")
    
    sample_queries = [
        "Which customer segment spends the most?",
        "How many customers belong to each segment?",
        "Who are the at-risk customers?",
        "Which segment should receive loyalty offers?",
        "Which is the largest segment?"
    ]
    
    st.markdown("**Quick Prompt Ideas:**")
    prompt_cols = st.columns(len(sample_queries))
    selected_prompt = None
    for idx, prompt_text in enumerate(sample_queries):
        with prompt_cols[idx]:
            if st.button(prompt_text, key=f"btn_prompt_{idx}"):
                selected_prompt = prompt_text
                
    user_query = st.text_input(
        "Type your question:",
        value=selected_prompt if selected_prompt else "Which customer segment spends the most?",
        placeholder="e.g. Which segment spends the most? or How many customers in each segment?"
    )
    
    if user_query:
        query_result = process_analytics_query(
            query_text=user_query,
            df_clustered=df_clean,
            cluster_profiles=cluster_profiles,
            feature_cols=selected_features
        )
        
        st.markdown(f"""
        <div class="query-card">
            <div class="query-header">Query Response</div>
            <div class="query-body">
                {query_result['answer']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# -------------------------------------------------------------
# TAB 8: EXPORT & DOWNLOAD CENTER
# -------------------------------------------------------------
with tabs[7]:
    st.subheader("📥 Export Labeled Data & Executive Reports")
    st.markdown("Download your segmented customer records and generated business insights for reporting and presentations.")
    
    # Export CSV with clusters and persona names
    export_df = df_clean.copy()
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    
    col_ex1, col_ex2 = st.columns(2)
    with col_ex1:
        st.download_button(
            label="⬇️ Download Segmented Customer Dataset (CSV)",
            data=csv_bytes,
            file_name="segmented_customers_with_personas.csv",
            mime="text/csv"
        )
        
    with col_ex2:
        # Generate summary markdown report
        report_text = f"""# Customer Segmentation Executive Report
Dataset: {data_source}
Algorithm: {algorithm}
Total Customers: {len(df_clean):,}
Features Analyzed: {', '.join(selected_features)}

## Cluster Summary
"""
        for c, p in cluster_profiles.items():
            report_text += f"\n### {p['icon']} Cluster {c}: {p['persona_name']}\n"
            report_text += f"- Customer Count: {p['customer_count']:,} ({p['percentage']}%)\n"
            report_text += f"- Description: {p['description']}\n"
            report_text += f"- Average Metrics: {p['feature_means']}\n"
            
        st.download_button(
            label="⬇️ Download Executive Summary Report (.txt)",
            data=report_text,
            file_name="Customer_Segmentation_Executive_Report.txt",
            mime="text/plain"
        )
