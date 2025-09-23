#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Add these functions and modifications to your existing Streamlit code

import pandas as pd
import numpy as np
import streamlit as st

# 1. STANDARDIZE ROUNDING FUNCTION
def standardize_metrics(value, decimals=2):
    """Ensure consistent rounding across all metrics"""
    if pd.isna(value):
        return 0.0
    return round(float(value), decimals)

# 2. MODIFIED create_overview_metrics function with number consistency
def create_overview_metrics(phone_data, case_data):
    """Create overview metrics cards with consistent formatting"""
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Phone Calls", f"{len(phone_data):,}")
        if 'Status' in phone_data.columns:
            answer_rate = (phone_data['Status'] == 'Answered').mean() * 100
            # Consistent rounding to 1 decimal place
            st.metric("Answer Rate", f"{standardize_metrics(answer_rate, 1)}%")

    with col2:
        st.metric("Total Cases", f"{len(case_data):,}")
        resolution_rate = case_data['Closed Date'].notna().mean() * 100
        # Consistent rounding to 1 decimal place  
        st.metric("Resolution Rate", f"{standardize_metrics(resolution_rate, 1)}%")

    with col3:
        total_interactions = len(phone_data) + len(case_data)
        st.metric("Total Interactions", f"{total_interactions:,}")
        phone_percentage = (len(phone_data) / total_interactions) * 100 if total_interactions else 0.0
        # Consistent rounding to 1 decimal place
        st.metric("Phone Channel %", f"{standardize_metrics(phone_percentage, 1)}%")

    with col4:
        if 'Cost' in phone_data.columns:
            total_cost = phone_data['Cost'].sum(skipna=True)
            st.metric("Total Phone Cost", f"${standardize_metrics(total_cost, 2):,.2f}")
            avg_cost = phone_data['Cost'].mean(skipna=True)
            # Use 3 decimal places for cost per call as it's very small
            st.metric("Avg Cost/Call", f"${standardize_metrics(avg_cost, 3):,.3f}")

# 3. MODIFIED create_daily_trends function with correlation added
def create_daily_trends(phone_data, case_data):
    """Create daily trend analysis with correlation coefficient"""
    st.subheader("📈 Daily Volume Trends")

    phone_daily = phone_data.groupby('Call Date').size().reset_index()
    phone_daily.columns = ['Date', 'Phone_Volume']
    phone_daily['Date'] = pd.to_datetime(phone_daily['Date'])

    case_daily = case_data.groupby('Case Date').size().reset_index()
    case_daily.columns = ['Date', 'Case_Volume']
    case_daily['Date'] = pd.to_datetime(case_daily['Date'])

    daily_combined = pd.merge(phone_daily, case_daily, on='Date', how='outer').fillna(0)
    daily_combined = daily_combined.sort_values('Date')

    # Create the plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily_combined['Date'],
        y=daily_combined['Phone_Volume'],
        mode='lines+markers',
        name='Phone Calls',
        line=dict(color='blue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=daily_combined['Date'],
        y=daily_combined['Case_Volume'],
        mode='lines+markers',
        name='Cases',
        line=dict(color='red', width=2)
    ))
    fig.update_layout(
        title='Daily Volume Comparison',
        xaxis_title='Date',
        yaxis_title='Volume'
    )
    st.plotly_chart(fig, use_container_width=True)

    # ADD MISSING CORRELATION METRIC
    correlation = daily_combined['Phone_Volume'].corr(daily_combined['Case_Volume'])
    st.metric("Daily Volume Correlation", f"{standardize_metrics(correlation, 3)}")
    
    return daily_combined

# 4. ADD PEAK HOUR ANALYSIS to overview metrics  
def add_peak_hour_analysis(phone_data):
    """Add peak hour identification"""
    if 'Call_Hour' in phone_data.columns:
        peak_hour = phone_data['Call_Hour'].value_counts().idxmax()
        peak_volume = phone_data['Call_Hour'].value_counts().max()
        
        # Add this as an info box in the overview section
        st.info(f"📊 Peak Hour Analysis: {peak_hour}:00 with {peak_volume:,} calls ({standardize_metrics((peak_volume/len(phone_data))*100, 1)}% of daily volume)")

# 5. ENHANCED analyze_case_performance with consistent formatting
def analyze_case_performance(case_data):
    """Analyze case/email channel performance with consistent metrics"""
    st.subheader("📧 Case/Email Channel Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if 'Case Origin' in case_data.columns:
            origin_counts = case_data['Case Origin'].value_counts()
            fig = px.bar(
                x=origin_counts.values,
                y=origin_counts.index,
                orientation='h',
                title='Case Origins',
                labels={'x': 'Number of Cases', 'y': 'Origin'}
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'Topic' in case_data.columns:
            topic_counts = case_data['Topic'].value_counts().head(10)
            fig = px.bar(
                x=topic_counts.values,
                y=topic_counts.index,
                orientation='h',
                title='Top 10 Case Topics',
                labels={'x': 'Number of Cases', 'y': 'Topic'}
            )
            st.plotly_chart(fig, use_container_width=True)

    # Case Duration Analysis with consistent formatting
    if 'Case Duration (Hours)' in case_data.columns:
        duration_data = case_data['Case Duration (Hours)'].dropna()
        duration_data = duration_data[duration_data >= 0]  
        if len(duration_data) > 0:
            fig = px.histogram(
                x=duration_data,
                nbins=30,
                title='Case Resolution Time Distribution',
                labels={'x': 'Duration (Hours)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

            # CONSISTENT METRIC FORMATTING
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Duration", f"{standardize_metrics(duration_data.mean(), 1)} hours")
            with col2:
                st.metric("Median Duration", f"{standardize_metrics(duration_data.median(), 1)} hours")
            with col3:
                st.metric("Max Duration", f"{standardize_metrics(duration_data.max(), 1)} hours")

# 6. ADD SLA PERFORMANCE METRICS with consistent formatting
def add_sla_metrics(phone_data):
    """Add consistent SLA performance metrics"""
    if 'SLA breach' in phone_data.columns:
        sla_breaches = pd.to_numeric(phone_data['SLA breach'], errors='coerce').fillna(0).sum()
        total_calls = len(phone_data)
        sla_breach_rate = (sla_breaches / total_calls) * 100
        sla_compliance_rate = 100 - sla_breach_rate
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("SLA Compliance Rate", f"{standardize_metrics(sla_compliance_rate, 1)}%")
        with col2:
            st.metric("SLA Breach Rate", f"{standardize_metrics(sla_breach_rate, 1)}%")

# 7. UPDATED MAIN FUNCTION with all fixes
def main():
    """Main Streamlit app function with all fixes applied"""
    
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
    file_path = uploaded_file if uploaded_file is not None else 'datatest.xlsx'

    with st.spinner('Loading and preprocessing data...'):
        phone_data, case_data = load_and_preprocess_data(file_path)

    if phone_data is None or case_data is None:
        st.error("Failed to load data. Please check your file.")
        return

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Overview",
        "📞 Phone Analysis", 
        "📧 Case Analysis",
        "🔗 Cross-Channel",
        "🔍 Data Quality"
    ])

    with tab1:
        create_overview_metrics(phone_data, case_data)
        st.markdown("---")
        
        # ADD PEAK HOUR ANALYSIS
        add_peak_hour_analysis(phone_data)
        
        # ADD SLA METRICS  
        add_sla_metrics(phone_data)
        st.markdown("---")
        
        _ = create_hourly_analysis(phone_data, case_data)
        st.markdown("---")
        _ = create_daily_trends(phone_data, case_data)  # Now includes correlation

    with tab2:
        analyze_phone_performance(phone_data)

    with tab3:
        analyze_case_performance(case_data)  # Updated with consistent formatting

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

    # ENHANCED SIDEBAR with additional metrics
    st.sidebar.header("📊 Quick Stats")
    st.sidebar.metric("Phone Data Shape", f"{phone_data.shape[0]} × {phone_data.shape[1]}")
    st.sidebar.metric("Case Data Shape", f"{case_data.shape[0]} × {case_data.shape[1]}")
    
    # Add key performance indicators to sidebar
    if 'Status' in phone_data.columns:
        answer_rate = (phone_data['Status'] == 'Answered').mean() * 100
        st.sidebar.metric("Answer Rate", f"{standardize_metrics(answer_rate, 1)}%")
    
    resolution_rate = case_data['Closed Date'].notna().mean() * 100
    st.sidebar.metric("Case Resolution Rate", f"{standardize_metrics(resolution_rate, 1)}%")

    st.sidebar.subheader("Date Ranges")
    if 'Call Time' in phone_data.columns and phone_data['Call Time'].notna().any():
        st.sidebar.write(f"**Phone Data:** {phone_data['Call Time'].min().date()} to {phone_data['Call Time'].max().date()}")
    if 'Created Date' in case_data.columns and case_data['Created Date'].notna().any():
        st.sidebar.write(f"**Case Data:** {case_data['Created Date'].min().date()} to {case_data['Created Date'].max().date()}")

if __name__ == "__main__":
    main()

