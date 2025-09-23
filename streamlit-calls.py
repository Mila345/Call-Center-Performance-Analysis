#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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

# Streamlit page configuration
st.set_page_config(page_title="Contact Center Analysis", layout="wide")
st.title("🏢 Multi-Channel Contact Center Performance Dashboard")

@st.cache_data
def load_and_preprocess_data(file_path='datatest.xlsx'):
    """Load and preprocess both sheets with proper data type handling"""
    try:
        # Load Phone Data
        phone_data = pd.read_excel(file_path, sheet_name='Phone Data')
        
        # Load Case Data  
        case_data = pd.read_excel(file_path, sheet_name='Case Data')
        
        # Preprocess Phone Data
        phone_data['Call Time'] = pd.to_datetime(phone_data['Call Time'])
        phone_data['Call Date'] = phone_data['Call Time'].dt.date
        phone_data['Call_Hour'] = phone_data['Call Time'].dt.hour  # Use underscore to avoid conflicts
        
        # Fix the Week column issue - convert Period to string
        if 'week' in phone_data.columns:
            phone_data['Week_Number'] = phone_data['week'].astype(str)
        
        # Convert duration columns to numeric
        duration_cols = ['Ringing', 'Talking', 'Ring time', 'Wait Time in Queue', 
                        'Total Waiting Time (queue+ring)', 'Talk Time', 
                        'Total Call Duration (excl IVR)']
        for col in duration_cols:
            if col in phone_data.columns:
                phone_data[col] = pd.to_numeric(phone_data[col], errors='coerce')
        
        # Convert other numeric columns
        if 'Cost' in phone_data.columns:
            phone_data['Cost'] = pd.to_numeric(phone_data['Cost'], errors='coerce')
        
        # Preprocess Case Data
        case_data['Created Date'] = pd.to_datetime(case_data['Created Date'])
        case_data['Closed Date'] = pd.to_datetime(case_data['Closed Date'])
        case_data['Case Date'] = case_data['Created Date'].dt.date
        case_data['Case_Hour'] = case_data['Created Date'].dt.hour
        
        # Calculate case duration in hours
        case_data['Case Duration (Hours)'] = (
            case_data['Closed Date'] - case_data['Created Date']
        ).dt.total_seconds() / 3600
        
        # Convert email columns to numeric
        numeric_cols = ['Number of Emails Received', 'Number of Emails Sent']
        for col in numeric_cols:
            if col in case_data.columns:
                case_data[col] = pd.to_numeric(case_data[col], errors='coerce')
        
        return phone_data, case_data
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

def create_overview_metrics(phone_data, case_data):
    """Create overview metrics cards"""
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
        phone_percentage = (len(phone_data) / total_interactions) * 100
        st.metric("Phone Channel %", f"{phone_percentage:.1f}%")
    
    with col4:
        if 'Cost' in phone_data.columns:
            total_cost = phone_data['Cost'].sum()
            st.metric("Total Phone Cost", f"${total_cost:.2f}")
            avg_cost = phone_data['Cost'].mean()
            st.metric("Avg Cost/Call", f"${avg_cost:.3f}")

def create_hourly_analysis(phone_data, case_data):
    """Create hourly pattern analysis"""
    st.subheader("📊 Hourly Pattern Analysis")
    
    # Prepare hourly data
    phone_hourly = phone_data.groupby('Call_Hour').size().reset_index()
    phone_hourly.columns = ['Hour', 'Phone_Calls']
    
    case_hourly = case_data.groupby('Case_Hour').size().reset_index()
    case_hourly.columns = ['Hour', 'Cases']
    
    # Merge hourly data
    hourly_combined = pd.merge(phone_hourly, case_hourly, on='Hour', how='outer').fillna(0)
    
    # Create plotly chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly_combined['Hour'],
        y=hourly_combined['Phone_Calls'],
        name='Phone Calls',
        marker_color='lightblue'
    ))
    fig.add_trace(go.Bar(
        x=hourly_combined['Hour'],
        y=hourly_combined['Cases'],
        name='Cases Created',
        marker_color='lightcoral'
    ))
    
    fig.update_layout(
        title='Hourly Distribution: Phone Calls vs Cases',
        xaxis_title='Hour of Day',
        yaxis_title='Volume',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    return hourly_combined

def create_daily_trends(phone_data, case_data):
    """Create daily trend analysis"""
    st.subheader("📈 Daily Volume Trends")
    
    # Daily phone calls
    phone_daily = phone_data.groupby('Call Date').size().reset_index()
    phone_daily.columns = ['Date', 'Phone_Volume']
    phone_daily['Date'] = pd.to_datetime(phone_daily['Date'])
    
    # Daily cases
    case_daily = case_data.groupby('Case Date').size().reset_index()
    case_daily.columns = ['Date', 'Case_Volume']
    case_daily['Date'] = pd.to_datetime(case_daily['Date'])
    
    # Merge daily data
    daily_combined = pd.merge(phone_daily, case_daily, on='Date', how='outer').fillna(0)
    daily_combined = daily_combined.sort_values('Date')
    
    # Create line chart
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
    
    # Calculate and display correlation
    correlation = daily_combined['Phone_Volume'].corr(daily_combined['Case_Volume'])
    st.metric("Daily Volume Correlation", f"{correlation:.3f}")
    
    return daily_combined

def analyze_phone_performance(phone_data):
    """Analyze phone channel performance"""
    st.subheader("📞 Phone Channel Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Call Status Distribution
        if 'Status' in phone_data.columns:
            status_counts = phone_data['Status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Call Status Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # SLA Breach Analysis
        if 'SLA breach' in phone_data.columns:
            sla_breach_count = phone_data['SLA breach'].sum()
            sla_compliance_count = len(phone_data) - sla_breach_count
            
            fig = px.pie(
                values=[sla_compliance_count, sla_breach_count],
                names=['SLA Compliant', 'SLA Breach'],
                title='SLA Performance',
                color_discrete_map={'SLA Compliant': 'green', 'SLA Breach': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Talk Time Distribution
    if 'Talk Time' in phone_data.columns:
        talk_times = phone_data['Talk Time'].dropna()
        if len(talk_times) > 0:
            fig = px.histogram(
                x=talk_times,
                nbins=30,
                title='Talk Time Distribution',
                labels={'x': 'Talk Time (seconds)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)

def analyze_case_performance(case_data):
    """Analyze case/email channel performance"""
    st.subheader("📧 Case/Email Channel Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Case Origin Distribution
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
        # Top Topics
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
    
    # Case Duration Analysis
    if 'Case Duration (Hours)' in case_data.columns:
        duration_data = case_data['Case Duration (Hours)'].dropna()
        duration_data = duration_data[duration_data >= 0]  # Remove negative durations
        
        if len(duration_data) > 0:
            fig = px.histogram(
                x=duration_data,
                nbins=30,
                title='Case Resolution Time Distribution',
                labels={'x': 'Duration (Hours)', 'y': 'Frequency'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Duration statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Duration", f"{duration_data.mean():.1f} hours")
            with col2:
                st.metric("Median Duration", f"{duration_data.median():.1f} hours")
            with col3:
                st.metric("Max Duration", f"{duration_data.max():.1f} hours")

def create_correlation_analysis(phone_data, case_data, daily_combined):
    """Create correlation analysis"""
    st.subheader("🔗 Cross-Channel Correlation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Volume correlation scatter plot
        fig = px.scatter(
            daily_combined,
            x='Phone_Volume',
            y='Case_Volume',
            title='Phone vs Case Daily Volume Correlation',
            trendline='ols'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Phone data correlations
        phone_numeric = phone_data.select_dtypes(include=[np.number])
        if len(phone_numeric.columns) > 1:
            corr_matrix = phone_numeric.corr()
            fig = px.imshow(
                corr_matrix,
                title='Phone Data Correlation Matrix',
                color_continuous_scale='RdBu_r',
                aspect='auto'
            )
            st.plotly_chart(fig, use_container_width=True)

def display_data_quality_report(phone_data, case_data):
    """Display data quality analysis"""
    st.subheader("🔍 Data Quality Report")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Phone Data Missing Values:**")
        phone_missing = phone_data.isnull().sum()
        phone_missing_pct = (phone_missing / len(phone_data)) * 100
        phone_quality_df = pd.DataFrame({
            'Column': phone_missing.index,
            'Missing Count': phone_missing.values,
            'Missing %': phone_missing_pct.values
        })
        phone_quality_df = phone_quality_df[phone_quality_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        st.dataframe(phone_quality_df, use_container_width=True)
    
    with col2:
        st.write("**Case Data Missing Values:**")
        case_missing = case_data.isnull().sum()
        case_missing_pct = (case_missing / len(case_data)) * 100
        case_quality_df = pd.DataFrame({
            'Column': case_missing.index,
            'Missing Count': case_missing.values,
            'Missing %': case_missing_pct.values
        })
        case_quality_df = case_quality_df[case_quality_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        st.dataframe(case_quality_df, use_container_width=True)

def main():
    """Main Streamlit app function"""
    
    # File upload option
    uploaded_file = st.file_uploader("Upload Excel file", type=['xlsx'])
    
    if uploaded_file is not None:
        file_path = uploaded_file
    else:
        file_path = 'datatest.xlsx'
    
    # Load data
    with st.spinner('Loading and preprocessing data...'):
        phone_data, case_data = load_and_preprocess_data(file_path)
    
    if phone_data is None or case_data is None:
        st.error("Failed to load data. Please check your file.")
        return
    
    # Create tabs for different analyses
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
        hourly_data = create_hourly_analysis(phone_data, case_data)
        st.markdown("---")
        daily_data = create_daily_trends(phone_data, case_data)
    
    with tab2:
        analyze_phone_performance(phone_data)
    
    with tab3:
        analyze_case_performance(case_data)
    
    with tab4:
        # Recreate daily_data if needed
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
    
    # Sidebar with summary statistics
    st.sidebar.header("📊 Quick Stats")
    st.sidebar.metric("Phone Data Shape", f"{phone_data.shape[0]} × {phone_data.shape[1]}")
    st.sidebar.metric("Case Data Shape", f"{case_data.shape[0]} × {case_data.shape[1]}")
    
    # Date ranges
    st.sidebar.subheader("Date Ranges")
    st.sidebar.write(f"**Phone Data:** {phone_data['Call Time'].min().date()} to {phone_data['Call Time'].max().date()}")
    st.sidebar.write(f"**Case Data:** {case_data['Created Date'].min().date()} to {case_data['Created Date'].max().date()}")

if __name__ == "__main__":
    main()

