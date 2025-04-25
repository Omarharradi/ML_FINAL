### TOOL
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
import streamlit.components.v1 as components


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
    plot_top_performers_by_skill,
    build_histogram_with_leaders,
    build_histogram_with_leaders_eq,
    plot_typology_distribution,
    plot_strongest_and_weakest_skills,
    plot_training_buckets_per_skill,
    plot_recommended_resources_by_skill,
    plot_eq_leader_skills,
    build_typology_bar_chart
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
os.environ["GOOGLE_API_KEY"] = 'AIzaSyAV5qNzuQnQ3lnndlWXmcPbQwBnLSTG5Vg'

#GOOGLE_API_KEY='AIzaSyAV5qNzuQnQ3lnndlWXmcPbQwBnLSTG5Vg'
GOOGLE_APPLICATION_CREDENTIALS='development/my-service-account-key.json'

# Load data and skills mapping
df = pd.read_csv('LDP_summary.csv')
df['Dashboard Number'] = df['# Dashboard'].str.split(':', n=1).str[0].str.strip()
df['Leader'] = df['Last name'].str.strip() + ' ' + df['First name'].str.strip()
df['EQ']=df['Overall Results']
df['Link'] = "https://ivy-dashboard-4833f144eaf4.herokuapp.com/page-2?user_id=" + df['ID'].astype(str)
df['Dashboard Link'] = df['Link'].apply(lambda x: f"[Open Dashboard]({x})")
below70=pd.read_csv('below_70.csv')
above85=pd.read_csv('above_85.csv')
between_70_84=pd.read_csv('between_70_84.csv')

resource=pd.read_csv('resources_summary.csv')


with open("skills_mapping_renamed.json", "r") as f:
    skills_mapping = json.load(f)

@st.cache_resource
def get_pipeline_cached():
    return initialize_pipeline()

@st.cache_resource
def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest")

llm = get_llm()

# Common filters
df_filtered, selected_dashboards, selected_positions, selected_individuals, selected_type = dynamic_sidebar_filters(df)
filtered_resources=get_filtered_df(resource, selected_dashboards, selected_positions, selected_individuals, selected_type)
training=pd.read_csv('Training Clusters.csv')
filtered_training=get_filtered_df(training, selected_dashboards, selected_positions, selected_individuals, selected_type)
grouped = filtered_resources.groupby(['Leader', 'Skill']).agg(
    Score=('Score', 'last'),
    Below_Threshold_Count=('Below Threshold', 'last'),
    Skill_category=('Skill Category', 'last'),
    Dashboard=('# Dashboard', 'last'),
    Position=('Position', 'last'),
).reset_index().rename(columns={'Dashboard': '# Dashboard'})


melted_df = df.melt(
    id_vars=["Leader", "# Dashboard", "Typology 1", "Position"], 
    value_vars=[
    "Emotional Facilitation of Thought",
    "Emotional Understanding",
    "Emotional Management",
    "Emotional Self-awareness",
    "Awareness of Strengths and Limitations",
    "Comfort with Emotions",
    "Recognition of Other's Emotions",
    "Rumination",
    "Problem-Solving",
    "Positive Mindset",
    "Emotional Reflection",
    "Emotional Integration",
    "Conflict-Resolution Knowledge",
    "Empathy",
    "Social Insight",
    "Self-Control",
    "Resilience/Hardiness",
    "Coping Skills",
    "Self-Motivation",
    "Striving",
    "Emotional Selectivity",
    "Adaptable Social Skills",
    "Conflict-Resolution Behavior",
    "Problem-Solving_manssa",
    "Planning, Organizing, & Controlling",
    "Goal Setting",
    "Information Gathering & Analysis",
    "Organization Skills",
    "Proactive Approach",
    "Monitoring Employee Performance",
    "Measuring & Evaluating Results",
    "Project Management",
    "Staffing & HR Functions",
    "Recruitment & Hiring",
    "Building Effective Teams",
    "Training & Onboarding",
    "Succession Planning",
    "Leading",
    "Coaching & Mentoring",
    "Rewarding Performance",
    "Fairness",
    "Supportiveness",
    "Managing Diversity",
    "Managerial Courage",
    "Charisma",
    "Business Skills",
    "Innovative Mindset",
    "Calculated Risk-Taking",
    "Negotiating Ability",
    "Customer Orientation",
    "Integrity & Ethics",
    "Soft Skills & Teamwork",
    "Communication Skills",
    "Listening Skills",
    "Social Insight_manssa",
    "Conflict Management",
    "Patience",
    "Approachability",
    "Intrapersonal Skills",
    "Accountability",
    "Change Management",
    "Drive",
    "Positive Mindset_manssa",
    "Steady Effort",
    "Key Skills",
    "Necessary Skills",
    "Beneficial Skills"
    ],
    var_name="Skill",
    value_name="Score"
)
filtered_melted=get_filtered_df(melted_df, selected_dashboards, selected_positions, selected_individuals, selected_type)


import streamlit as st

# --- Setup Chat Memory ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Handle Message Send ---
def handle_message():
    user_msg = st.session_state.temp_input
    if user_msg:
        st.session_state.chat_history.append({"role": "user", "content": user_msg})

        # Get assistant response
        with st.spinner("Thinking..."):
            ask = get_pipeline_cached()
            response = ask(user_msg)

        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.temp_input = ""  # clear input field

# --- Chat Interface ---
st.sidebar.markdown("---")
st.sidebar.markdown("### Chat Assistant")
# Clear chat history
if st.sidebar.button("🗑️ Clear History"):
    st.session_state.chat_history = []

# Use a temp key for the text input
st.sidebar.text_input("Your message:", key="temp_input", on_change=handle_message)

# --- Chat Display ---
st.sidebar.markdown("""
    <style>
    .chat-container {
        background-color: #f7f7f9;
        padding: 10px;
        border-radius: 10px;
        max-height: 400px;
        overflow-y: auto;
        font-family: sans-serif;
    }

    .chat-message {
        padding: 8px 12px;
        margin: 8px 0;
        border-radius: 12px;
        max-width: 80%;
        word-wrap: break-word;
    }

    .user-message {
        background-color: #d1e7dd;
        margin-left: auto;
        text-align: right;
    }

    .assistant-message {
        background-color: #e9ecef;
        margin-right: auto;
        text-align: left;
    }
    </style>
""", unsafe_allow_html=True)

#st.sidebar.markdown('<div class="chat-container">', unsafe_allow_html=True)

# Show last 5 messages only
for msg in st.session_state.chat_history[-5:]:
    role_class = "user-message" if msg["role"] == "user" else "assistant-message"
    st.sidebar.markdown(
        f'<div class="chat-message {role_class}">{msg["content"]}</div>',
        unsafe_allow_html=True,
    )

st.sidebar.markdown("</div>", unsafe_allow_html=True)



st.markdown("---")
st.header("Do all Leaders fit")
st.caption("ℹ️ This section shows key metrics to assess if Leaders fit their roles.")
st.markdown("---")

st.subheader("Fit Composition", help='This chart shows which leaders fit their roles, which are overqualified, and which are underqualified.')
donut_fig, donut_summary = build_donut_chart(df_filtered['LIS'])
type_pie, type_summary = plot_typology_distribution(df_filtered)

col1_row, col2_row = st.columns(2)
with st.container():
    display_insight("donut", build_donut_chart, donut_summary, llm, "Insights for Donut Chart")
    st.plotly_chart(donut_fig, use_container_width=True)

st.markdown("---")
st.subheader("Key Metrics")
st.subheader("Leadership Index Score (LIS)", help="This chart shows the distribution of Leaders by their LIS scores.")

with st.container():
    selected_leaders = st.multiselect("Highlight specific leaders", df['Leader'].unique(), help="Select one or more leaders to highlight them on the chart.")
    hist_fig, hist_summary = build_histogram_with_leaders(df_filtered, highlight_leaders=selected_leaders)
    display_insight("hist", build_histogram_with_leaders, hist_summary, llm, "Insights for Histogram")
    st.plotly_chart(hist_fig, use_container_width=True)

st.markdown("---")
st.subheader("Emotional Intelligence (EQ)", help="This chart shows the distribution of Leaders by their EQ scores.")

with st.container():
    selected_leaders_eq = st.multiselect("Highlight specific leaders", df['Leader'].unique(), key="eq_leader_selector", help="Select one or more leaders to highlight them on the chart.")
    hist_fig_eq, hist_summary_eq = build_histogram_with_leaders_eq(df_filtered, highlight_leaders=selected_leaders_eq)
    display_insight("hist_eq", build_histogram_with_leaders_eq, hist_summary_eq, llm, "Insights for EQ Histogram")
    st.plotly_chart(hist_fig_eq, use_container_width=True)

st.markdown("---")
st.subheader("Personality types", help="This chart shows the distribution of Leaders by their personality types.")
with st.container():
    display_insight("pie", plot_typology_distribution, type_summary, llm, "Insights for Pie Chart")
    st.plotly_chart(type_pie, use_container_width=True)

with st.container():
    st.subheader("Polar Chart by Typology", help="This chart shows the distribution of EQ or LIS scores for each dashboard.")
    selected_metric = st.selectbox("Choose a metric", ["LIS", "EQ"], help="Select the metric to visualize in the polar chart.")
    selected_leaders_type = st.multiselect("Highlight specific leaders", df['Leader'].unique(), key="type_leaders", help="Select one or more leaders to highlight them on the chart.")
    fig, stats = build_typology_bar_chart(df_filtered, metric=selected_metric, highlight_leaders=selected_leaders_type)
    display_insight("polar_chart_dropdown", build_typology_bar_chart, stats, llm, f"Insights for {selected_metric} by Typology")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.header("What are the Skills Gap?")
st.caption("ℹ️ This section analyzes the skills gap in the organization.")
st.markdown("---")

st.subheader("Top Performers by Skill", help="This chart shows the top performers for each skill.")

with st.container():
    unique_skills = sorted([skill for skill in filtered_melted["Skill"].unique()])
    if unique_skills:
        selected_skill = st.selectbox("Select Skill", unique_skills, key="skill_selector", help="Select a skill to visualize top performers.")
        top_fig, top_performers_df = plot_top_performers_by_skill(filtered_melted, skill=selected_skill)
        #st.plotly_chart(top_fig, use_container_width=True)
        display_insight("top_performers_by_skill", plot_top_performers_by_skill, top_performers_df, llm, "Insights for top performers by skill")
        st.plotly_chart(top_fig, use_container_width=True)

st.markdown("---")
st.subheader("Strongest and Weakest Skills", help="This chart shows the strongest and weakest skills for each leader.")
fig_strong, fig_weak, top_skills_df, low_skills_df = plot_strongest_and_weakest_skills(filtered_melted)

col1, col2 = st.columns(2)
with col1:
    display_insight("strongest_skills", plot_strongest_and_weakest_skills, top_skills_df, llm, "Insights for top skills")
    st.plotly_chart(fig_strong, use_container_width=True)

with col2:
    display_insight("weakest", plot_strongest_and_weakest_skills, low_skills_df, llm, "Insights for weakest skills")
    st.plotly_chart(fig_weak, use_container_width=True)

st.markdown("---")
st.subheader("Radar Chart", help="This chart shows the performance of leaders accross their skills, compared to the average of their group.")
unique_dashboards = sorted(df_filtered["# Dashboard"].unique())

if unique_dashboards:
    selected_dashboard = st.selectbox("Select Dashboard", unique_dashboards, help="Select a dashboard group to visualize the radar chart." )
    dashboard_leaders = sorted(df_filtered[df_filtered["# Dashboard"] == selected_dashboard]["Leader"].unique())
    selected_leader = st.selectbox("Select Leader", dashboard_leaders, help="Select a leader to visualize their performance.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Average Key Skills", value=df_filtered[df_filtered["Leader"] == selected_leader]["Key Skills"].values[0])
    with col2:
        st.metric("Average Useful Skills", value=df_filtered[df_filtered["Leader"] == selected_leader]["Key Skills"].values[0])
    with col3:
        st.metric("Average Supplemental Skills", value=df_filtered[df_filtered["Leader"] == selected_leader]["Key Skills"].values[0])
    fig_radar, radar_stats = radar_chart_plotly(selected_dashboard, selected_leader, df_filtered, skills_mapping)
    display_insight("radar", radar_chart_plotly, radar_stats, llm, "Insights for Radar Chart")
    st.plotly_chart(fig_radar, use_container_width=True)
else:
    st.write("No data available for the selected filters.")

st.markdown("---")
st.header("How to Bridge the skill gap")
st.caption("ℹ️ This section shows the training clusters and recommended resources to bridge the skill gap.")
st.markdown("---")

st.subheader("Training Clusters", help="This chart shows the distribution of training clusters by skill.")
with st.container():
    fig_training, summary_stat_training = plot_training_buckets_per_skill(filtered_training)
    display_insight("training_clusters", plot_training_buckets_per_skill, summary_stat_training, llm, "Insights for training clusters")
    st.plotly_chart(fig_training, use_container_width=True)

st.markdown("---")

selected_leader = st.selectbox("Select a Leader", df["Leader"].unique())
personal_data = resource[resource["Leader"] == selected_leader]
st.metric("Total Recommended Resources", len(personal_data))
st.metric("Distinct Skills Targeted", personal_data["Skill"].nunique())
st.metric("Skills Below Threshold", personal_data.loc[personal_data["Below Threshold"] == True, "Skill"].nunique())

fig_clusters, resource_summary = plot_recommended_resources_by_skill(personal_data, selected_leader)
display_insight("recommended_resources", plot_recommended_resources_by_skill, resource_summary, llm, "Insights for recommended resources")
st.plotly_chart(fig_clusters, use_container_width=True)

st.markdown("---")
st.subheader("Recommended Resources by Category")

with st.container():
    unique_resources = sorted([res for res in filtered_resources["Resource Type"].unique() if res != "Action Step"])
    if unique_resources:
        selected_resource = st.selectbox("Select Resource", unique_resources)
        top_res_fix, top_res = plot_top_resources_by_category(filtered_resources, selected_resource)
        display_insight("top_res", plot_top_resources_by_category, top_res, llm, "Insights for top Chart")
        st.plotly_chart(top_res_fix, use_container_width=True)


