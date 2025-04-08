
## streamlit: 

import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass


import streamlit as st
import plotly.express as px
import pandas as pd
import json
import base64
import numpy as np
import inspect
from langchain_google_genai import ChatGoogleGenerativeAI
import os

from utils import get_filtered_df, build_donut_chart, build_histogram, build_polar_chart, build_box_plot, radar_chart_plotly, get_insights_chart, initialize_pipeline, display_insight, dynamic_sidebar_filters
from dotenv import load_dotenv
load_dotenv()


# Use secrets from Streamlit Cloud
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]



# Set page configuration early
st.set_page_config(page_title="Leadership Competency Viewer", layout="wide")

# Load data
df = pd.read_csv('LDP_summary.csv')
df['Dashboard Number'] = df['# Dashboard'].str.split(':', n=1).str[0].str.strip()
df['Leader'] = df['Last name'].str.strip() + ' ' + df['First name'].str.strip()

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
#with st.sidebar:
  #  st.header("Filters")
   # selected_dashboards = st.multiselect("Select Dashboard(s)", sorted(df["# Dashboard"].unique()))
   # selected_positions = st.multiselect("Select Position(s)", sorted(df["Position"].unique())) if "Position" in df.columns else []
   # selected_individuals = st.multiselect("Select Leader(s)", sorted(df["Leader"].unique())) if "Leader" in df.columns else []

df_filtered, selected_dashboards, selected_positions, selected_individuals = dynamic_sidebar_filters(df)


# Custom CSS for chat styling (include only if not already present)
st.markdown("""
    <style>
    .chat-message { 
        margin-bottom: 10px; 
        padding: 8px; 
        border-radius: 5px; 
    }
    .user-message { 
        background-color: #007AFF; /* Apple's system blue */
        color: white;
        text-align: right; 
    }
    .assistant-message { 
        background-color: #34C759; /* Apple's system green */
        color: white;
        text-align: left; 
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Chat Assistant
with st.sidebar:
    st.header("Chat Assistant:")
    chat_input = st.text_input("Your message:")

    if chat_input:
        # Display the user's message in the chat container
        st.markdown(f'<div class="chat-message user-message">{chat_input}</div>', unsafe_allow_html=True)
        
        # Show spinner while generating the assistant's response
        with st.spinner("Thinking..."):
            ask = get_pipeline_cached("result.json")
            answer = ask(chat_input)
        
        # Display the assistant's response
        st.markdown(f'<div class="chat-message assistant-message">{answer}</div>', unsafe_allow_html=True)




# Filter data
filtered_df = get_filtered_df(df, selected_dashboards, selected_positions, selected_individuals)
lis_data = filtered_df['LIS']

# Build charts using utils package
donut_fig, donut_summary = build_donut_chart(lis_data)
hist_fig, hist_summary = build_histogram(lis_data)
fig_box, summary_box  = build_box_plot(filtered_df)
fig_polar, plar_stats = build_polar_chart(filtered_df)


# Initialize LLM
#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

llm = get_llm()

# Main page layout
st.header("All Leaders")
st.dataframe(filtered_df[["# Dashboard", "Leader", "Email", 'LIS']].dropna().reset_index(drop=True))


# Main page layout
st.header("Overall Results")
col1_row, col2_row = st.columns(2)
with col1_row:
    st.plotly_chart(donut_fig, use_container_width=True)
    display_insight("donut", build_donut_chart, donut_summary, llm, "Insights for Donut Chart")


with col2_row:
    st.plotly_chart(hist_fig, use_container_width=True)
    display_insight("hist", build_histogram, hist_summary, llm, "Insights for Histogram")


st.markdown("---")
st.subheader("Additional Analysis")
col3_row, col4_row = st.columns(2)
with col3_row:
    st.plotly_chart(fig_polar, use_container_width=True)
    display_insight("polar", build_polar_chart, plar_stats, llm, "Insights for Ploar Chart")


with col4_row:
    st.plotly_chart(fig_box, use_container_width=True)
    display_insight("boxplot", build_box_plot, summary_box, llm, "Insights for Box Plot")

        

st.markdown("---")
st.subheader("Radar Chart")
unique_dashboards = sorted(filtered_df["# Dashboard"].unique())
if unique_dashboards:
    selected_radar = st.selectbox("Select Dashboard for Radar Chart", unique_dashboards)
    fig_radar, radar_stats = radar_chart_plotly(selected_radar, filtered_df, skills_mapping)

    st.plotly_chart(fig_radar, use_container_width=True)
    display_insight("radar", radar_chart_plotly, radar_stats, llm, "Insights for Radar Chart")
    print(radar_stats)

else:
    st.write("No data available for the selected filters.")

