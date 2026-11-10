import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import urllib.request
import json

st.set_page_config(page_title="NYC 311 Data Analysis Dashboard", layout="wide")

st.title("🏙️ NYC 311 Urban Pulse - 15 Core Investigations Dashboard")
st.write("GUVI Data Science Project Analysis based on 100,000 records.")

@st.cache_data
def load_data():
    try:
        df = pd.read_csv("urban_pulse_100k_data.csv", low_memory=False)
    except FileNotFoundError:
        url = "https://data.cityofnewyork.us/resource/erm2-nwe9.json?$limit=100000"
        response = urllib.request.urlopen(url)
        data = json.loads(response.read().decode())
        df = pd.read_json(json.dumps(data))
        
    if 'created_date' in df.columns:
        df['created_date'] = pd.to_datetime(df['created_date'], errors='coerce')
    if 'closed_date' in df.columns:
        df['closed_date'] = pd.to_datetime(df['closed_date'], errors='coerce')
    return df

df = load_data()

st.subheader("1. Total Distinct Request Volume")
total_requests = len(df)
distinct_unique_keys = df['unique_key'].nunique() if 'unique_key' in df.columns else total_requests
st.metric(label="Total Records", value=total_requests)
st.metric(label="Distinct Request Volume (Unique Keys)", value=distinct_unique_keys)

st.subheader("2. Top 10 Complaint Categories by Count")
if 'complaint_type' in df.columns:
    top_10 = df['complaint_type'].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=top_10.values, y=top_10.index, ax=ax, palette="viridis")
    ax.set_xlabel("Count")
    ax.set_ylabel("Complaint Type")
    st.pyplot(fig)

st.subheader("3. Agency Workload Distribution")
if 'agency' in df.columns:
    agency_counts = df['agency'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(x=agency_counts.index, y=agency_counts.values, ax=ax, palette="magma")
    ax.set_xlabel("Agency")
    ax.set_ylabel("Request Count")
    st.pyplot(fig)

st.subheader("4. Request Volume by Borough")
if 'borough' in df.columns:
    borough_counts = df['borough'].value_counts()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(x=borough_counts.index, y=borough_counts.values, ax=ax, palette="Set2")
    ax.set_xlabel("Borough")
    ax.set_ylabel("Request Count")
    plt.xticks(rotation=45)
    st.pyplot(fig)

st.subheader("5. Monthly Request Trend")
if 'created_date' in df.columns:
    df['month'] = df['created_date'].dt.to_period('M').astype(str)
    monthly_trend = df['month'].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 4))
    monthly_trend.plot(kind='line', marker='o', ax=ax, color='b')
    ax.set_xlabel("Month")
    ax.set_ylabel("Request Count")
    st.pyplot(fig)

st.subheader("6. Demand by Weekday and Hour")
if 'created_date' in df.columns:
    df['weekday'] = df['created_date'].dt.day_name()
    df['hour'] = df['created_date'].dt.hour
    pivot_table = df.pivot_table(index='weekday', columns='hour', values='unique_key', aggfunc='count', fill_value=0)
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(pivot_table, cmap="YlGnBu", ax=ax)
    st.pyplot(fig)

st.subheader("7. Distribution of Recorded Request Statuses")
if 'status' in df.columns:
    status_counts = df['status'].value_counts()
    st.bar_chart(status_counts)

if 'created_date' in df.columns and 'closed_date' in df.columns:
    df['resolution_duration_hours'] = (df['closed_date'] - df['created_date']).dt.total_seconds() / 3600
    valid_res = df['resolution_duration_hours'].dropna()
    valid_res = valid_res[valid_res >= 0]

    st.subheader("8. Descriptive Statistics for Valid Resolution Durations (Hours)")
    st.write(valid_res.describe())

    st.subheader("9. Median Resolution Duration by Complaint Category")
    if 'complaint_type' in df.columns:
        median_comp = df.groupby('complaint_type')['resolution_duration_hours'].median().sort_values(ascending=False).head(10)
        st.bar_chart(median_comp)

    st.subheader("10. Median Resolution Duration by Agency")
    if 'agency' in df.columns:
        median_agency = df.groupby('agency')['resolution_duration_hours'].median().sort_values(ascending=False)
        st.bar_chart(median_agency)

    st.subheader("11. IQR-Based Outlier Investigation on Resolution Time")
    Q1 = valid_res.quantile(0.25)
    Q3 = valid_res.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = valid_res[(valid_res < lower_bound) | (valid_res > upper_bound)]
    st.write(f"IQR Value: {IQR:.2f}")
    st.write(f"Total Outliers detected: {len(outliers)}")

    st.subheader("12. Resolution-Time Skewness and Distribution")
    skewness = valid_res.skew()
    st.write(f"Skewness of Resolution Time: {skewness:.2f}")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(valid_res.clip(upper=valid_res.quantile(0.95)), bins=50, kde=True, ax=ax, color='purple')
    ax.set_xlabel("Resolution Duration (Hours - Clipped at 95th percentile)")
    st.pyplot(fig)

st.subheader("13. Geographic Complaint Concentration (Borough Summaries)")
if 'borough' in df.columns:
    borough_geo = df['borough'].value_counts()
    st.bar_chart(borough_geo)

st.subheader("14. Recorded Closure-Rate Comparison by Agency")
if 'agency' in df.columns and 'status' in df.columns:
    df['is_closed'] = df['status'].str.lower() == 'closed'
    closure_rate = df.groupby('agency')['is_closed'].mean() * 100
    st.bar_chart(closure_rate.sort_values(ascending=False))

st.subheader("15. Categories with High Request Volume & Long Median Resolution Duration")
if 'complaint_type' in df.columns and 'resolution_duration_hours' in df.columns:
    summary_df = df.groupby('complaint_type').agg(
        request_count=('unique_key', 'count'),
        median_duration=('resolution_duration_hours', 'median')
    ).reset_index()
    top_high_long = summary_df.sort_values(by=['request_count', 'median_duration'], ascending=False).head(10)
    st.dataframe(top_high_long)

