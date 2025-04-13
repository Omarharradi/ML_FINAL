import sys
import os
import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Custom utils
from utils import (
    get_filtered_df,
    build_donut_chart,
    build_histogram,
    build_polar_chart,
    build_box_plot,
    radar_chart_plotly,
    display_insight,
    initialize_pipeline,
    dynamic_sidebar_filters,
    plot_resource_type_distribution,
    plot_top_skills,
    plot_resource_distribution_by_category,
    plot_top_resources_by_category,
    plot_leaders_below_threshold,
    plot_top_performers_by_skill
)

# SQLite patch for compatibility
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass

# Load environment variables
load_dotenv()
st.set_page_config(page_title="Leadership Competency Viewer", layout="wide")

# Set API keys from Streamlit secrets
#os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
#os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = st.secrets["GOOGLE_APPLICATION_CREDENTIALS"]
GOOGLE_API_KEY='AIzaSyAV5qNzuQnQ3lnndlWXmcPbQwBnLSTG5Vg'
GOOGLE_APPLICATION_CREDENTIALS='development/my-service-account-key.json'

# Load data and skills mapping
df = pd.read_csv('LDP_summary.csv')
df['Dashboard Number'] = df['# Dashboard'].str.split(':', n=1).str[0].str.strip()
df['Leader'] = df['Last name'].str.strip() + ' ' + df['First name'].str.strip()
resource=pd.read_csv('resources_summary.csv')


with open("skills_mapping_renamed.json", "r") as f:
    skills_mapping = json.load(f)

@st.cache_resource
def get_pipeline_cached(json_path="result.json"):
    return initialize_pipeline(json_path)

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")

llm = get_llm()

# Common filters
df_filtered, selected_dashboards, selected_positions, selected_individuals = dynamic_sidebar_filters(df)
filtered_resources=get_filtered_df(resource, selected_dashboards, selected_positions, selected_individuals)
grouped = filtered_resources.groupby(['Leader', 'Skill']).agg(
    Score=('Score', 'last'),
    Below_Threshold_Count=('Below Threshold', 'last'),
    Skill_category=('Skill Category', 'last'),
    Dashboard=('# Dashboard', 'last'),
    Position=('Position', 'last'),
).reset_index().rename(columns={'Dashboard': '# Dashboard'})

# Sidebar custom navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Overview", "Leader Details"])

# Sidebar Chat Assistant
st.sidebar.markdown("---")
st.sidebar.header("Chat Assistant:")
chat_input = st.sidebar.text_input("Your message:")

if chat_input:
    st.sidebar.markdown(f'<div class="chat-message user-message">{chat_input}</div>', unsafe_allow_html=True)
    with st.spinner("Thinking..."):
        ask = get_pipeline_cached("result.json")
        answer = ask(chat_input)
    st.sidebar.markdown(f'<div class="chat-message assistant-message">{answer}</div>', unsafe_allow_html=True)



# Overview Page
if page == "Overview":
    st.markdown("<h1>Leadership Results Overview</h1>", unsafe_allow_html=True)
    st.markdown("This app displays interactive visualizations based on overall leadership competency levels.")

    st.header("All Leaders")
    st.dataframe(df_filtered[["# Dashboard", "Leader", "Email", 'LIS']].dropna().reset_index(drop=True))

    st.header("Overall Results")
    donut_fig, donut_summary = build_donut_chart(df_filtered['LIS'])
    hist_fig, hist_summary = build_histogram(df_filtered['LIS'])
    fig_box, summary_box  = build_box_plot(df_filtered)
    fig_polar, plar_stats = build_polar_chart(df_filtered)

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
        display_insight("polar", build_polar_chart, plar_stats, llm, "Insights for Polar Chart")

    with col4_row:
        st.plotly_chart(fig_box, use_container_width=True)
        display_insight("boxplot", build_box_plot, summary_box, llm, "Insights for Box Plot")

    st.markdown("---")
    st.subheader("Radar Chart")
    unique_dashboards = sorted(df_filtered["# Dashboard"].unique())

    if unique_dashboards:
        selected_dashboard = st.selectbox("Select Dashboard", unique_dashboards)
        dashboard_leaders = sorted(df_filtered[df_filtered["# Dashboard"] == selected_dashboard]["Leader"].unique())
        selected_leader = st.selectbox("Select Leader", dashboard_leaders)

        fig_radar, radar_stats = radar_chart_plotly(selected_dashboard, selected_leader, df_filtered, skills_mapping)
        st.plotly_chart(fig_radar, use_container_width=True)
        display_insight("radar", radar_chart_plotly, radar_stats, llm, "Insights for Radar Chart")
    else:
        st.write("No data available for the selected filters.")

# Leader Detail Page
elif page == "Leader Details":
    st.markdown("---")
    st.subheader("📊 Learning Resource Insights")

    resource_type_fig, resource_type_summary = plot_resource_type_distribution(filtered_resources)
    top_skills_fig, top_skills_summary = plot_top_skills(filtered_resources)
    resource_distribution_fig, resource_distribution_summary = plot_resource_distribution_by_category(filtered_resources)


    # Row 1
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(resource_type_fig, use_container_width=True)
        display_insight("type_dist", plot_resource_type_distribution, resource_type_summary, llm, "Insights for Distribution Chart")
    with col2:
        st.plotly_chart(top_skills_fig, use_container_width=True)
        display_insight("top_skills", plot_top_skills, top_skills_summary, llm, "Insights for top skills Chart")
        

    # Row 2
    col3, col4 = st.columns(2)
    with col3:
        st.plotly_chart(resource_distribution_fig, use_container_width=True)
        display_insight("resource_dist", plot_resource_distribution_by_category, resource_distribution_summary, llm, "Insights for resources Chart")
    with col4:
        unique_skills = sorted([skill for skill in filtered_resources["Skill"].unique()])

        if unique_skills:
        # Let the user select a skill
            selected_skill = st.selectbox("Select Skill", unique_skills, key="skill_selector")
            top_fig, top_performers_df = plot_top_performers_by_skill(filtered_resources, skill=selected_skill)

            st.plotly_chart(top_fig, use_container_width=True)

            # Provide insights for the chart (optional)
            display_insight("top_performers_by_skill",plot_top_performers_by_skill,top_performers_df,llm,"Insights for top performers by skill")
            

        
    
    
    unique_resources = sorted([res for res in filtered_resources["Resource Type"].unique() if res != "Action Step"])

    if unique_resources:
        selected_resource = st.selectbox("Select Resource", unique_resources)

        top_res_fix, top_res = plot_top_resources_by_category(filtered_resources, selected_resource)

        st.plotly_chart(top_res_fix, use_container_width=True)
        display_insight("top_res", plot_top_resources_by_category, top_res, llm, "Insights for top  Chart")

    topn=st.number_input("Top N", min_value=1, max_value=10, value=5)

    leaders_fig, leader_below = plot_leaders_below_threshold(grouped,topn)

    st.plotly_chart(leaders_fig, use_container_width=True)
    display_insight("top_res_by_skill", plot_leaders_below_threshold, leader_below, llm, "Insights for top by skill Chart")
