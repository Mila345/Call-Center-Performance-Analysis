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
st.title("Contact Center Performance Dashboard")

# Helper: robust duration parser -> seconds (never raises)
def _normalize_duration_column(df: pd.DataFrame, col: str) -> None:
    """
    Safely convert a duration-like column to seconds.
    Handles:
      - timedelta dtype
      - numeric seconds
      - HH:MM:SS strings (and similar)
      - messy strings with units/symbols
    Leaves NaN where parsing fails.
    """
    if col not in df.columns:
        return

    s = df[col]

    # timedelta -> seconds
    if pd.api.types.is_timedelta64_dtype(s):
        df[col] = s.dt.total_seconds()
        return

    # numeric -> coerce numeric
    if pd.api.types.is_numeric_dtype(s):
        df[col] = pd.to_numeric(s, errors="coerce")
        return

    # object/mixed: parse safely
    s_str = s.astype(str)
    seconds = pd.Series(np.nan, index=s.index, dtype="float64")

    # entries with ":" likely time strings -> try timedelta
    mask_colon = s_str.str.contains(":", regex=False, na=False)
    if mask_colon.any():
        try:
            td = pd.to_timedelta(s_str[mask_colon], errors="coerce")
            seconds.loc[mask_colon] = td.dt.total_seconds()
        except Exception:
            # per-item fallback if needed
            seconds.loc[mask_colon] = s_str[mask_colon].apply(
                lambda x: pd.to_timedelta(x, errors="coerce")
            ).dt.total_seconds()

    # everything else: strip non-numeric and coerce
    stripped = s_str[~mask_colon].str.replace(r"[^0-9\.\-eE]", "", regex=True)
    seconds.loc[~mask_colon] = pd.to_numeric(stripped, errors="coerce")

    df[col] = seconds

@st.cache_data
def load_and_preprocess_data(file_path='datatest.xlsx'):
    """Load and preprocess both sheets with proper data type handling"""
    try:
        # Load
        phone_data = pd.read_excel(file_path, sheet_name='Phone Data')
        case_data  = pd.read_excel(file_path, sheet_name='Case Data')

        # --- Phone preprocessing ---
        phone_data['Call Time'] = pd.to_datetime(phone_data['Call Time'], errors='coerce')
        phone_data['Call Date'] = phone_data['Call Time'].dt.date
        phone_data['Call_Hour'] = phone_data['Call Time'].dt.hour  # Use underscore to avoid conflicts

        # Fix the Week column issue - convert Period to string
        if 'week' in phone_data.columns:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                phone_data['Week_Number'] = phone_data['week'].astype(str)

        # Normalize duration columns to seconds (robust)
        duration_cols = [
            'Ringing', 'Talking', 'Ring time', 'Wait Time in Queue',
            'Total Waiting Time (queue+ring)', 'Talk Time',
            'Total Call Duration (excl IVR)'
        ]
        for col in duration_cols:
            _normalize_duration_column(phone_data, col)

        # Other numeric columns
        if 'Cost' in phone_data.columns:
            phone_data['Cost'] = pd.to_numeric(phone_data['Cost'], errors='coerce')

        # --- Case preprocessing ---
        case_data['Created Date'] = pd.to_datetime(case_data['Created Date'], errors='coerce')
        case_data['Closed Date']  = pd.to_datetime(case_data['Closed Date'],  errors='coerce')
        case_data['Case Date']    = case_data['Created Date'].dt.date
        case_data['Case_Hour']    = case_data['Created Date'].dt.hour

        # Calculate case duration in hours
        case_data['Case Duration (Hours)'] = (
            case_data['Closed Date'] - case_data['Created Date']
        ).dt.total_seconds() / 3600

        # Convert email columns to numeric
        for col in ['Number of Emails Received', 'Number of Emails Sent']:
            if col in case_data.columns:
                case_data[col] = pd.to_numeric(case_data[col], errors='coerce')

        return phone_data, case_data

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None

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

def create_hourly_analysis(phone_data, case_data):
    """Create hourly pattern analysis"""
    st.subheader("📊 Hourly Pattern Analysis")

    phone_hourly = phone_data.groupby('Call_Hour').size().reset_index()
    phone_hourly.columns = ['Hour', 'Phone_Calls']

    case_hourly = case_data.groupby('Case_Hour').size().reset_index()
    case_hourly.columns = ['Hour', 'Cases']

    hourly_combined = pd.merge(phone_hourly, case_hourly, on='Hour', how='outer').fillna(0)

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

def analyze_phone_performance(phone_data):
    """Analyze phone channel performance"""
    st.subheader("📞 Phone Channel Analysis")

    col1, col2 = st.columns(2)

    with col1:
        if 'Status' in phone_data.columns:
            status_counts = phone_data['Status'].value_counts()
            fig = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title='Call Status Distribution'
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if 'SLA breach' in phone_data.columns:
            sla_breach_count = pd.to_numeric(phone_data['SLA breach'], errors='coerce').fillna(0).sum()
            sla_compliance_count = len(phone_data) - sla_breach_count
            fig = px.pie(
                values=[sla_compliance_count, sla_breach_count],
                names=['SLA Compliant', 'SLA Breach'],
                title='SLA Performance',
                color_discrete_map={'SLA Compliant': 'green', 'SLA Breach': 'red'}
            )
            st.plotly_chart(fig, use_container_width=True)

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

def create_correlation_analysis(phone_data, case_data, daily_combined):
    """Create correlation analysis"""
    st.subheader("🔗 Cross-Channel Correlation Analysis")

    col1, col2 = st.columns(2)

    with col1:
        # Use trendline if statsmodels is installed; else skip
        trend = 'ols'
        try:
            import statsmodels.api as sm  # noqa: F401
        except Exception:
            trend = None

        fig = px.scatter(
            daily_combined,
            x='Phone_Volume',
            y='Case_Volume',
            title='Phone vs Case Daily Volume Correlation',
            trendline=trend
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
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
        phone_missing_pct = (phone_missing / max(len(phone_data), 1)) * 100
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
        case_missing_pct = (case_missing / max(len(case_data), 1)) * 100
        case_quality_df = pd.DataFrame({
            'Column': case_missing.index,
            'Missing Count': case_missing.values,
            'Missing %': case_missing_pct.values
        })
        case_quality_df = case_quality_df[case_quality_df['Missing Count'] > 0].sort_values('Missing Count', ascending=False)
        st.dataframe(case_quality_df, use_container_width=True)

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