import streamlit as st
import plotly.express as px
import pandas as pd
import json
import base64
import numpy as np
import inspect
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from utils import get_filtered_df, build_donut_chart, build_histogram, build_polar_chart, build_box_plot, radar_chart_plotly, get_insights_chart

# Set page configuration early
st.set_page_config(page_title="Leadership Competency Viewer", layout="wide")

# Load data
df = pd.read_csv('LDP_summary_anonymized.csv')
df['Dashboard Number'] = df['# Dashboard'].str.split(':', n=1).str[0].str.strip()
with open("skills_mapping_renamed.json", "r") as f:
    skills_mapping = json.load(f)

# Custom CSS for UI
st.markdown("""
    <style>
    .block-container { padding-top: 0.2rem; }
    .header-container { display: flex; justify-content: space-between; align-items: center; }
    .header-title { flex: 7; }
    .header-logo { flex: 1; text-align: right; }
    .header-logo img { max-width: 100%; height: auto; }
    .chat-container { background-color: #F7F7F8; padding: 15px; border-radius: 10px; max-height: 300px; overflow-y: auto; font-family: 'Helvetica', sans-serif; }
    .chat-message { margin-bottom: 10px; padding: 8px; border-radius: 5px; }
    .user-message { background-color: #DCF8C6; text-align: right; }
    .assistant-message { background-color: #FFFFFF; text-align: left; }
    .center-button { display: flex; justify-content: center; margin-top: 10px; }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="header-container">
  <div class="header-title">
     <h1>Leadership Results Overview</h1>
  </div>
  <div class="header-logo"></div>
  <div class="header-logo"></div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
This app displays interactive visualizations based on overall leadership competency levels.
""")

# Sidebar filters and chat
with st.sidebar:
    st.header("Filters")
    selected_dashboards = st.multiselect("Select Dashboard(s)", sorted(df["# Dashboard"].unique()))
    selected_positions = st.multiselect("Select Position(s)", sorted(df["Position"].unique())) if "Position" in df.columns else []

    st.markdown("---")
    st.header("Chat Assistant (Beta)")
    chat_input = st.text_input("Your message:")
    if st.button("Send", key="chat_send"):
        st.markdown('<div class="chat-message assistant-message">Placeholder response: LLM backend not implemented.</div>', unsafe_allow_html=True)
    if chat_input:
        st.markdown(f'<div class="chat-message user-message">{chat_input}</div>', unsafe_allow_html=True)

# Filter data
filtered_df = get_filtered_df(df, selected_dashboards, selected_positions)
lis_data = filtered_df['LIS']

# Build charts using utils
mean_lis = np.mean(lis_data) if len(lis_data) > 0 else 0
std_lis = np.std(lis_data) if len(lis_data) > 0 else 0
std_low = mean_lis - 1.5 * std_lis
std_high = mean_lis + 1.5 * std_lis

donut_fig = build_donut_chart(lis_data)
hist_fig = build_histogram(lis_data)
#fig_polar = build_polar_chart(lis_data)
#fig_box = build_box_plot(lis_data)

# Additional plots
avg_overall = filtered_df.groupby("Dashboard Number")["LIS"].mean().reset_index()
fig_polar = px.bar_polar(
    avg_overall,
    r="LIS",
    theta="Dashboard Number",
    color="Dashboard Number",
    template="plotly_white",
    title="Average Overall LIS by Dashboard (Polar Bar Chart)",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_polar.update_layout(
    margin=dict(l=50, r=50, t=100, b=50),
    polar=dict(radialaxis=dict(visible=True, range=[0, avg_overall["LIS"].max() * 1.1]))
)
fig_box = px.box(
    filtered_df,
    x="Dashboard Number",
    y="Overall Results",
    title="Box Plot of EQ by Dashboard",
    template="plotly_white",
    color="Dashboard Number",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# Initialize LLM
os.environ['GOOGLE_API_KEY'] = "AIzaSyDjv5kiOA45O25NPxjp9B60CcOLjBSS5vY"
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# Main page layout
st.header("Overall Results")
col1_row, col2_row = st.columns(2)
with col1_row:
    st.plotly_chart(donut_fig, use_container_width=True)
    with st.container():
        st.markdown("<div class='center-button'>", unsafe_allow_html=True)
        if st.button("🧠 Explain this graph", key="explain_donut"):
            source_code = inspect.getsource(build_donut_chart)
            with st.spinner("Generating insights..."):
                response = get_insights_chart(lis_data=lis_data, source_code=source_code, llm=llm)   
            st.info(response)
        st.markdown("</div>", unsafe_allow_html=True)

with col2_row:
    st.plotly_chart(hist_fig, use_container_width=True)
    with st.container():
        st.markdown("<div class='center-button'>", unsafe_allow_html=True)
        if st.button("🧠 Explain this graph", key="explain_hist"):
            source_code = inspect.getsource(build_histogram)
            with st.spinner("Generating insights..."):
                response = get_insights_chart(lis_data=lis_data, source_code=source_code, llm=llm)   
            st.info(response)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Additional Analysis")
col3_row, col4_row = st.columns(2)
with col3_row:
    st.plotly_chart(fig_polar, use_container_width=True)
    with st.container():
        st.markdown("<div class='center-button'>", unsafe_allow_html=True)
        if st.button("🧠 Explain this graph", key="explain_polar"):
            source_code = inspect.getsource(build_polar_chart)
            with st.spinner("Generating insights..."):
                response = get_insights_chart(lis_data=lis_data, source_code=source_code, llm=llm)   
            st.info(response)
        st.markdown("</div>", unsafe_allow_html=True)

with col4_row:
    st.plotly_chart(fig_box, use_container_width=True)
    with st.container():
        st.markdown("<div class='center-button'>", unsafe_allow_html=True)
        if st.button("🧠 Explain this graph", key="explain_box"):
            source_code = inspect.getsource(build_box_plot)
            with st.spinner("Generating insights..."):
                response = get_insights_chart(lis_data=lis_data, source_code=source_code, llm=llm)   
            st.info(response)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Radar Chart")
unique_dashboards = sorted(filtered_df["# Dashboard"].unique())
if unique_dashboards:
    selected_radar = st.selectbox("Select Dashboard for Radar Chart", unique_dashboards)
    radar_fig = radar_chart_plotly(selected_radar, filtered_df, skills_mapping)
    st.plotly_chart(radar_fig, use_container_width=True)
    with st.container():
        st.markdown("<div class='center-button'>", unsafe_allow_html=True)
        if st.button("🧠 Explain this graph", key="explain_radar"):
            source_code = inspect.getsource(radar_chart_plotly)
            with st.spinner("Generating insights..."):
                response = get_insights_chart(lis_data=lis_data, source_code=source_code, llm=llm)   
            st.info(response)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.write("No data available for the selected filters.")
