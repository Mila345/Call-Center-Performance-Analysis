#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# streamlit-calls.py
import io
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Optional: enable plotly trendline if statsmodels is present
try:
    import statsmodels.api as sm  # noqa: F401
    HAS_SM = True
except Exception:
    HAS_SM = False

# ---------- Configure ----------
st.set_page_config(page_title="Contact Center Analysis", layout="wide")
st.title("Multi-Channel Contact Center Performance Dashboard")

# ---------- Resolve datatest.xlsx (no upload required) ----------
def resolve_datatest_path() -> Path:
    """
    Look for datatest.xlsx in the repo:
      1) ./datatest.xlsx
      2) ./data/datatest.xlsx
    """
    here = Path(__file__).parent
    candidates = [
        here / "datatest.xlsx",
        here / "data" / "datatest.xlsx",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find 'datatest.xlsx'. Place it in repo root or in 'data/datatest.xlsx'."
    )

# ---------- Data loader ----------
@st.cache_data
def load_and_preprocess_data(xls_like):
    """
    Load and preprocess both sheets with proper data type handling.
    `xls_like` can be a path, file-like, or bytes.
    """
    xls = pd.ExcelFile(xls_like)
    phone_data = pd.read_excel(xls, sheet_name='Phone Data')
    case_data  = pd.read_excel(xls, sheet_name='Case Data')

    # ---- Phone preprocessing ----
    phone_data['Call Time'] = pd.to_datetime(phone_data['Call Time'], errors='coerce')
    phone_data['Call Date'] = phone_data['Call Time'].dt.date
    phone_data['Call_Hour'] = phone_data['Call Time'].dt.hour

    # If 'week' column is Period-like, stringify it
    if 'week' in phone_data.columns:
        try:
            phone_data['Week_Number'] = phone_data['week'].astype(str)
        except Exception:
            pass

    # Normalize duration-like columns (numeric or HH:MM:SS) to seconds
    duration_cols = [
        'Ringing', 'Talking', 'Ring time', 'Wait Time in Queue',
        'Total Waiting Time (queue+ring)', 'Talk Time',
        'Total Call Duration (excl IVR)'
    ]
    for col in duration_cols:
        if col in phone_data.columns:
            s = phone_data[col]
            if not pd.api.types.is_numeric_dtype(s):
                td = pd.to_timedelta(s, errors='coerce')
                if td.notna().any():
                    phone_data[col] = td.dt.total_seconds()
                else:
                    phone_data[col] = pd.to_numeric(s, errors='coerce')
            else:
                phone_data[col] = pd.to_numeric(s, errors='coerce')

    if 'Cost' in phone_data.columns:
        phone_data['Cost'] = pd.to_numeric(phone_data['Cost'], errors='coerce')

    # ---- Case preprocessing ----
    case_data['Created Date'] = pd.to_datetime(case_data['Created Date'], errors='coerce')
    case_data['Closed Date']  = pd.to_datetime(case_data['Closed Date'],  errors='coerce')
    case_data['Case Date']    = case_data['Created Date'].dt.date
    case_data['Case_Hour']    = case_data['Created Date'].dt.hour

    case_data['Case Duration (Hours)'] = (
        case_data['Closed Date'] - case_data['Created Date']
    ).dt.total_seconds() / 3600

    for col in ['Number of Emails Received', 'Number of Emails Sent']:
        if col in case_data.columns:
            case_data[col] = pd.to_numeric(case_data[col], errors='coerce')

    return phone_data, case_data

# ---------- UI & Analytics ----------
def create_overview_metrics(phone_data, case_data):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Phone Calls", f"{len(phone_data):,}")
        if 'Status' in phone_data.columns:
            answer_rate = (phone_data['Status'] == 'Answered').mean() * 100
            st.metric("Answer Rate", f"{answer_rate:.1f}%")
    with col2:
        st.metric("Total Cases", f"{len(case_data):,}")
        resolution_rate = case_data['Closed Date'].notna().mean() * 100
        st.metric("Resolution Rate", f"{resolution_rate:.1f}%")
    with col3:
        total_interactions = len(phone_data) + len(case_data)
        st.metric("Total Interactions", f"{total_interactions:,}")
        phone_pct = (len(phone_data) / total_interactions * 100) if total_interactions else 0.0
        st.metric("Phone Channel %", f"{phone_pct:.1f}%")
    with col4:
        if 'Cost' in phone_data.columns:
            total_cost = phone_data['Cost'].sum(skipna=True)
            st.metric("Total Phone Cost", f"${total_cost:,.2f}")
            avg_cost = phone_data['Cost'].mean(skipna=True)
            st.metric("Avg Cost/Call", f"${avg_cost:,.3f}")

def create_hourly_analysis(phone_data, case_data):
    st.subheader("📊 Hourly Pattern Analysis")
    phone_hourly = phone_data.groupby('Call_Hour').size().reset_index()
    phone_hourly.columns = ['Hour', 'Phone_Calls']
    case_hourly = case_data.groupby('Case_Hour').size().reset_index()
    case_hourly.columns = ['Hour', 'Cases']
    hourly_combined = pd.merge(phone_hourly, case_hourly, on='Hour', how='outer').fillna(0)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=hourly_combined['Hour'], y=hourly_combined['Phone_Calls'],
                         name='Phone Calls', marker_color='lightblue'))
    fig.add_trace(go.Bar(x=hourly_combined['Hour'], y=hourly_combined['Cases'],
                         name='Cases Created', marker_color='lightcoral'))
    fig.update_layout(title='Hourly Distribution: Phone Calls vs Cases',
                      xaxis_title='Hour of Day', yaxis_title='Volume', barmode='group')
    st.plotly_chart(fig, use_container_width=True)
    return hourly_combined

def create_daily_trends(phone_data, case_data):
    st.subheader("📈 Daily Volume Trends")
    phone_daily = phone_data.groupby('Call Date').size().reset_index()
    phone_daily.columns = ['Date', 'Phone_Volume']
    phone_daily['Date'] = pd.to_datetime(phone_daily['Date'])
    case_daily = case_data.groupby('Case Date').size().reset_index()
    case_daily.columns = ['Date', 'Case_Volume']
    case_daily['Date'] = pd.to_datetime(case_daily['Date'])
    daily_combined = pd.merge(phone_daily, case_daily, on='Date', how='outer').fillna(0).sort_values('Date')

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily_combined['Date'], y=daily_combined['Phone_Volume'],
                             mode='lines+markers', name='Phone Calls',
                             line=dict(color='blue', width=2)))
    fig.add_trace(go.Scatter(x=daily_combined['Date'], y=daily_combined['Case_Volume'],
                             mode='lines+markers', name='Cases',
                             line=dict(color='red', width=2)))
    fig.update_layout(title='Daily Volume Comparison', xaxis_title='Date', yaxis_title='Volume')
    st.plotly_chart(fig, use_container_width=True)

    corr = daily_combined['Phone_Volume'].corr(daily_combined['Case_Volume'])
    st.metric("Daily Volume Correlation", f"{corr:.3f}")
    return daily_combined

def analyze_phone_performance(phone_data):
    st.subheader("📞 Phone Channel Analysis")
    col1, col2 = st.columns(2)
    with col1:
        if 'Status' in phone_data.columns:
            status_counts = phone_data['Status'].value_counts()
            fig = px.pie(values=status_counts.values, names=status_counts.index,
                         title='Call Status Distribution')
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'SLA breach' in phone_data.columns:
            sla_breach_count = pd.to_numeric(phone_data['SLA breach'], errors='coerce').fillna(0).sum()
            sla_compliance_count = len(phone_data) - sla_breach_count
            fig = px.pie(values=[sla_compliance_count, sla_breach_count],
                         names=['SLA Compliant', 'SLA Breach'],
                         title='SLA Performance',
                         color_discrete_map={'SLA Compliant': 'green', 'SLA Breach': 'red'})
            st.plotly_chart(fig, use_container_width=True)
    if 'Talk Time' in phone_data.columns:
        talk_times = phone_data['Talk Time'].dropna()
        if len(talk_times) > 0:
            fig = px.histogram(x=talk_times, nbins=30, title='Talk Time Distribution',
                               labels={'x': 'Talk Time (seconds)', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)

def analyze_case_performance(case_data):
    st.subheader("📧 Case/Email Channel Analysis")
    col1, col2 = st.columns(2)
    with col1:
        if 'Case Origin' in case_data.columns:
            origin_counts = case_data['Case Origin'].value_counts()
            fig = px.bar(x=origin_counts.values, y=origin_counts.index, orientation='h',
                         title='Case Origins', labels={'x': 'Number of Cases', 'y': 'Origin'})
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if 'Topic' in case_data.columns:
            topic_counts = case_data['Topic'].value_counts().head(10)
            fig = px.bar(x=topic_counts.values, y=topic_counts.index, orientation='h',
                         title='Top 10 Case Topics', labels={'x': 'Number of Cases', 'y': 'Topic'})
            st.plotly_chart(fig, use_container_width=True)
    if 'Case Duration (Hours)' in case_data.columns:
        duration_data = case_data['Case Duration (Hours)'].dropna()
        duration_data = duration_data[duration_data >= 0]
        if len(duration_data) > 0:
            fig = px.histogram(x=duration_data, nbins=30, title='Case Resolution Time Distribution',
                               labels={'x': 'Duration (Hours)', 'y': 'Frequency'})
            st.plotly_chart(fig, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            with c1: st.metric("Avg Duration", f"{duration_data.mean():.1f} hours")
            with c2: st.metric("Median Duration", f"{duration_data.median():.1f} hours")
            with c3: st.metric("Max Duration", f"{duration_data.max():.1f} hours")

def create_correlation_analysis(phone_data, case_data, daily_combined):
    st.subheader("🔗 Cross-Channel Correlation Analysis")
    c1, c2 = st.columns(2)
    with c1:
        fig = px.scatter(daily_combined, x='Phone_Volume', y='Case_Volume',
                         title='Phone vs Case Daily Volume Correlation',
                         trendline=('ols' if HAS_SM else None))
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        phone_numeric = phone_data.select_dtypes(include=[np.number])
        if len(phone_numeric.columns) > 1:
            corr_matrix = phone_numeric.corr()
            fig = px.imshow(corr_matrix, title='Phone Data Correlation Matrix',
                            color_continuous_scale='RdBu_r', aspect='auto')
            st.plotly_chart(fig, use_container_width=True)

def display_data_quality_report(phone_data, case_data):
    st.subheader("🔍 Data Quality Report")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Phone Data Missing Values:**")
        phone_missing = phone_data.isnull().sum()
        phone_quality_df = pd.DataFrame({
            'Column': phone_missing.index,
            'Missing Count': phone_missing.values,
            'Missing %': (phone_missing.values / max(len(phone_data), 1)) * 100
        })
        st.dataframe(phone_quality_df[phone_quality_df['Missing Count'] > 0]
                     .sort_values('Missing Count', ascending=False), use_container_width=True)
    with c2:
        st.write("**Case Data Missing Values:**")
        case_missing = case_data.isnull().sum()
        case_quality_df = pd.DataFrame({
            'Column': case_missing.index,
            'Missing Count': case_missing.values,
            'Missing %': (case_missing.values / max(len(case_data), 1)) * 100
        })
        st.dataframe(case_quality_df[case_quality_df['Missing Count'] > 0]
                     .sort_values('Missing Count', ascending=False), use_container_width=True)

def main():
    """Main Streamlit app function"""
    # Resolve and load datatest.xlsx from the repo
    try:
        data_path = resolve_datatest_path()
    except Exception as e:
        st.error(str(e))
        st.stop()

    with st.spinner('Loading and preprocessing data...'):
        phone_data, case_data = load_and_preprocess_data(data_path)

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview", "📞 Phone Analysis", "📧 Case Analysis",
        "🔗 Cross-Channel", "🔍 Data Quality"
    ])

    with tab1:
        create_overview_metrics(phone_data, case_data)
        st.markdown("---")
        _ = create_hourly_analysis(phone_data, case_data)
        st.markdown("---")
        _ = create_daily_trends(phone_data, case_data)

    with tab2:
        analyze_phone_performance(phone_data)

    with tab3:
        analyze_case_performance(case_data)

    with tab4:
        phone_daily = phone_data.groupby('Call Date').size().reset_index()
        phone_daily.columns = ['Date', 'Phone_Volume']
        phone_daily['Date'] = pd.to_datetime(phone_daily['Date'])
        case_daily = case_data.groupby('Case Date').size().reset_index()
        case_daily.columns = ['Date', 'Case_Volume']
        case_daily['Date'] = pd.to_datetime(case_daily['Date'])
        daily_combined = pd.merge(phone_daily, case_daily, on='Date', how='outer').fillna(0)
        create_correlation_analysis(phone_data, case_data, daily_combined)

    with tab5:
        display_data_quality_report(phone_data, case_data)

    # Sidebar quick stats
    st.sidebar.header("📊 Quick Stats")
    st.sidebar.metric("Phone Data Shape", f"{phone_data.shape[0]} × {phone_data.shape[1]}")
    st.sidebar.metric("Case Data Shape", f"{case_data.shape[0]} × {case_data.shape[1]}")
    st.sidebar.subheader("Date Ranges")
    if 'Call Time' in phone_data.columns and phone_data['Call Time'].notna().any():
        st.sidebar.write(f"**Phone Data:** {phone_data['Call Time'].min().date()} → {phone_data['Call Time'].max().date()}")
    if 'Created Date' in case_data.columns and case_data['Created Date'].notna().any():
        st.sidebar.write(f"**Case Data:** {case_data['Created Date'].min().date()} → {case_data['Created Date'].max().date()}")

if __name__ == "__main__":
    main()

