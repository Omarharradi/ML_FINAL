
## streamlit: 

import streamlit as st
import plotly.express as px
import pandas as pd
import json
import base64
import numpy as np
import inspect
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from utils import get_filtered_df, build_donut_chart, build_histogram, build_polar_chart, build_box_plot, radar_chart_plotly, get_insights_chart, initialize_pipeline

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/omar/development/my-service-account-key.json"

# Set page configuration early
st.set_page_config(page_title="Leadership Competency Viewer", layout="wide")

# Load data
df = pd.read_csv('LDP_summary_anonymized.csv')
df['Dashboard Number'] = df['# Dashboard'].str.split(':', n=1).str[0].str.strip()
with open("skills_mapping_renamed.json", "r") as f:
    skills_mapping = json.load(f)

@st.cache_resource
def get_pipeline_cached(json_path="result.json"):
    return initialize_pipeline(json_path)



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

# Custom CSS for chat styling (include only if not already present)
st.markdown("""
    <style>
    .chat-message { 
        margin-bottom: 10px; 
        padding: 8px; 
        border-radius: 5px; 
    }
    .user-message { 
        background-color: #ADD8E6; /* Light Blue */
        text-align: right; 
    }
    .assistant-message { 
        background-color: #ADD8E6; /* Light Gray */
        text-align: left; 
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Chat Assistant
with st.sidebar:
    st.header("Chat Assistant (Beta)")
    chat_input = st.text_input("Your message:")
    if chat_input:
        # Display the user's message in the chat container
        st.markdown(f'<div class="chat-message user-message">{chat_input}</div>', unsafe_allow_html=True)
        
        # Process the chat input using your cached pipeline
        ask = get_pipeline_cached("result.json")
        answer = ask(chat_input)
        
        # Display the assistant's response
        st.markdown(f'<div class="chat-message assistant-message">{answer}</div>', unsafe_allow_html=True)



# Filter data
filtered_df = get_filtered_df(df, selected_dashboards, selected_positions)
lis_data = filtered_df['LIS']

# Build charts using utils package
donut_fig, donut_summary = build_donut_chart(lis_data)
hist_fig, hist_summary = build_histogram(lis_data)
fig_box, summary_box  = build_box_plot(filtered_df)
fig_polar, plar_stats = build_polar_chart(filtered_df)


# Initialize LLM
os.environ['GOOGLE_API_KEY'] = "AIzaSyAV5qNzuQnQ3lnndlWXmcPbQwBnLSTG5Vg"
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

llm = get_llm()

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
                response = get_insights_chart(lis_data=donut_summary, source_code=source_code, llm=llm)   
            with st.expander("Insights for Donut Chart", expanded=True):
                st.info(response)
        st.markdown("</div>", unsafe_allow_html=True)

with col2_row:
    st.plotly_chart(hist_fig, use_container_width=True)
    with st.container():
        st.markdown("<div class='center-button'>", unsafe_allow_html=True)
        if st.button("🧠 Explain this graph", key="explain_hist"):
            source_code = inspect.getsource(build_histogram)
            with st.spinner("Generating insights..."):
                response = get_insights_chart(lis_data=hist_summary, source_code=source_code, llm=llm)   
            with st.expander("Insights for Histogram", expanded=True):
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
                response = get_insights_chart(lis_data=plar_stats, source_code=source_code, llm=llm)   
            with st.expander("Insights for Polar Chart", expanded=True):
                st.info(response)
        st.markdown("</div>", unsafe_allow_html=True)

with col4_row:
    st.plotly_chart(fig_box, use_container_width=True)
    with st.container():
        st.markdown("<div class='center-button'>", unsafe_allow_html=True)
        if st.button("🧠 Explain this graph", key="explain_box"):
            source_code = inspect.getsource(build_box_plot)
            with st.spinner("Generating insights..."):
                response = get_insights_chart(lis_data=summary_box, source_code=source_code, llm=llm)   
            with st.expander("Insights for Box Plot", expanded=True):
                st.info(response)
        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.subheader("Radar Chart")
unique_dashboards = sorted(filtered_df["# Dashboard"].unique())
if unique_dashboards:
    selected_radar = st.selectbox("Select Dashboard for Radar Chart", unique_dashboards)
    fig_radar, radar_stats = radar_chart_plotly(selected_radar, filtered_df, skills_mapping)

    st.plotly_chart(fig_radar, use_container_width=True)
    with st.container():
        st.markdown("<div class='center-button'>", unsafe_allow_html=True)
        if st.button("🧠 Explain this graph", key="explain_radar"):
            source_code = inspect.getsource(radar_chart_plotly)
            with st.spinner("Generating insights..."):
                response = get_insights_chart(lis_data=lis_data, source_code=source_code, llm=llm)   
            with st.expander("Insights for Radar Chart", expanded=True):
                st.info(response)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    st.write("No data available for the selected filters.")
