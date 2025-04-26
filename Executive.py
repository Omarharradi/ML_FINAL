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
    plot_lis_by_typology
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
#df_filtered, selected_dashboards, selected_positions, selected_individuals, selected_type = dynamic_sidebar_filters(df)
#filtered_resources=get_filtered_df(resource, selected_dashboards, selected_positions, selected_individuals, selected_type)
training=pd.read_csv('Training Clusters.csv')
#filtered_training=get_filtered_df(training, selected_dashboards, selected_positions, selected_individuals, selected_type)
#grouped = filtered_resources.groupby(['Leader', 'Skill']).agg(
 #   Score=('Score', 'last'),
  #  Below_Threshold_Count=('Below Threshold', 'last'),
   # Skill_category=('Skill Category', 'last'),
    #Dashboard=('# Dashboard', 'last'),
    #Position=('Position', 'last'),
#).reset_index().rename(columns={'Dashboard': '# Dashboard'})


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
#filtered_melted=get_filtered_df(melted_df, selected_dashboards, selected_positions, selected_individuals, selected_type)


def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()
with st.sidebar:
        # --- Logo section with base64 HTML ---
        nesma_base64 = get_image_base64("assets/nesma.png")
        ivy_base64 = get_image_base64("assets/Ivy Logo copy.png")

        st.markdown(f"""
            <div style="padding-top: 10px; padding-bottom: -5px;">
                <div style="display: flex; justify-content: start; align-items: center; gap: 10px;">
                    <img src="data:image/png;base64,{nesma_base64}" alt="Nesma Logo" width="120" style="vertical-align: middle;">
                    <img src="data:image/png;base64,{ivy_base64}" alt="Ivy Logo" width="120" style="vertical-align: middle;">
                </div>
            </div>
        """, unsafe_allow_html=True)

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
#st.sidebar.markdown("---")
st.sidebar.markdown("Chat Assistant", help="Ask questions about the data and get insights.")

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





# Build HTML table
html = """
<style>
.table-container {
    height: 300px;
    overflow-y: auto;
    border: 1px solid #ddd;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-family: Arial, sans-serif;
}
th, td {
    padding: 8px 12px;
    border: 1px solid #ddd;
    text-align: left;
}
a {
    color: #1f77b4;
    text-decoration: none;
}
a:hover {
    text-decoration: underline;
}
</style>
<div class="table-container">
<table>
    <thead>
        <tr>
            <th>Leader</th>
            <th>Position</th>
            <th>Dashboard Link</th>
        </tr>
    </thead>
    <tbody>
"""

for _, row in df.iterrows():
    html += f"""
        <tr>
            <td>{row['Leader']}</td>
            <td>{row['Position']}</td>
            <td><a href="{row['Link']}" target="_blank">Open Dashboard</a></td>
        </tr>
    """

html += """
    </tbody>
</table>
</div>
"""



## ppt insights

st.title("LEADERSHIP DEVELOPMENTPROGRAM (LDP) PILOT")
st.markdown("This tool displays interactive visualizations based on overall leadership competency levels.")

st.header("Access to All Leaders")
components.html(html, height=350, scrolling=True)
st.markdown("---")



import streamlit as st
from PIL import Image
import base64

st.header("Nesma & Partners LDNA ")

st.markdown("**Measured & Included in Current Scope**")
# --- Load Image ---
dna_image = Image.open("LDNA - LDP (3).png")

# Encode image to base64
with open("LDNA - LDP (3).png", "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode()

# Custom styled container with top and bottom arrows and labels
st.markdown(f"""
    <style>
    .custom-container {{
        background-color: #ffffff;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }}

    .dna-image {{
        width: 100%;
        max-width: 900px;
    }}

    .label-row, .upper-label-row {{
        display: flex;
        justify-content: space-around;
        color: #1d2742;
        font-weight: 300;
        flex-wrap: wrap;
        margin-left: 50px;
        margin-right: 50px;
    }}

    .label-row {{
        margin-top: 10px;
    }}

    .upper-label-row {{
        margin-bottom: 10px;
    }}

    .label {{
        flex: 1;
        text-align: center;
        font-size: 12px;
        min-width: 100px;
        padding: 5px;
    }}
    </style>

    <div class="custom-container">
        <div class="upper-label-row">
            <div class="label">Leadership Index Score (LIS)<div class="arrow">↑</div></div>
            <div class="label">Soft Skills Proficiency<div class="arrow">↑</div></div>
            <div class="label">Leadership Style Profile<div class="arrow">↑</div></div>
            <div class="label">Training Load Index<div class="arrow">↑</div></div>
            <div class="label">Engagement Index<div class="arrow">↑</div></div>
            <div class="label">Emotional Intelligence (EQ)<div class="arrow">↑</div></div>
        </div>
        <img src="data:image/png;base64,{encoded_image}" class="dna-image" />
        <div class="label-row">
            <div class="label">↓<br>Mentorship Readiness</div>
            <div class="label">↓<br>Strenght of Development Culture</div>
            <div class="label">↓<br>Recognition Practices</div>
            <div class="label">↓<br>Retentien Trends</div>
            <div class="label">↓<br>Alignment Hiring & Desired Leadership Traits</div>
            <div class="label">↓<br>Alignment Onboarding & Desired Leadership Traits</div>
            <div class="label">↓<br>Leadership Style Fit</div>
            <div class="label">↓<br>Leader/Employee Cultural Fit</div>
        </div>
    </div>
""", unsafe_allow_html=True)

st.markdown("**Expanded & Excluded from Current Scope**")



#6
st.markdown("---")

st.subheader("📊 Executive Summary: Understanding the leadershiplandscape of N&P")


# First Row
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Leaders Fit Their Roles", value="86%", delta="Strong or Very Strong Fit")
    st.caption("Based on their Key, Useful, and Supplemental Skills.")

with col2:
    st.metric(label="Avg Leadership Index (LIS)", value="78%")
    st.caption("Combines emotional intelligence and all job-relevant skills.")

# Second Row
col3, col4 = st.columns(2)

with col3:
    st.metric(label="Recommended EQ Training", value="96%")
    st.caption("Other flagged areas: Communication, Accountability.")

with col4:
    st.metric(label='"Mentoring" Leaders', value="35%")
    st.caption('This typology showed the highest leadership index (LIS) in our study.')
st.markdown("---")


# 28
st.subheader("Results: Leadership Index (LIS) & EQ")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Avg Leadership Index Score (LIS)", value="78%")
    st.caption("Cohort LIS above baseline of 70; majority are role-aligned.")

with col2:
    st.metric(label="Avg EQ Score", value="72%")
    st.caption("Developmental; 48 in intermediate tier.")

st.markdown("---")


# 29

st.subheader("Results: Avg Skill Scores & Leadership Styles")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Overall Avg Score (All Leadership Skills)", value="74%")
    st.caption("Strongest average skill was Goal Setting.")

with col2:
    st.metric(label='"Mentoring" Leadership Style', value="35%")
    st.caption("High people-centricity; asset for mentorship culture.")

df_sorted = df.sort_values(by='LIS', ascending=False)

type_pie1, type_summary1 = plot_typology_distribution(df)

st.plotly_chart(type_pie1, use_container_width=True, key="type_pie1")


st.markdown("---")



# 31
import streamlit as st
import matplotlib.pyplot as plt

# Section Header & Descriptions
st.subheader("Q1. Are Leaders in the Right Roles?")
st.markdown("**Leadership Index Score (LIS) Benchmarks / 100**")


# Render 3 identical pies for demo (replace each with unique data)
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Below 70")
    st.dataframe(below70)

with col2:
    st.subheader("70-84")
    st.dataframe(between_70_84)
with col3:
    st.subheader("85+")
    st.dataframe(above85)
st.markdown("---")


# 32

st.subheader("Leadership Role Fit Analysis")


# Create a new row with two columns for the other metrics
col1, col2 = st.columns(2)

with col1:
    st.metric(label="Leaders Scoring Well Above Avg", value="7")
    st.caption("Over 88.42 LIS")
    st.dataframe(above85)  # Replace 'above85' with your actual DataFrame variable

with col2:
    st.metric(label="Growth Needed Leaders", value="8")
    st.caption("Under 67.33 LIS")
    st.dataframe(above85)  # Replace 'above85' with the appropriate DataFrame variable if different


st.markdown("### How Is the Leadership Index Score (LIS) Calculated?")
st.code("LIS = (EQ × 0.40) + (KS × 0.30) + (US × 0.21) + (SS × 0.09)", language="python")

st.markdown("---")


st.markdown("### How Did the Whole Group Score?")

# Row: Key Skills, Useful Skills, Supplemental Skills
col4, col5, col6 = st.columns(3)

with col4:
    st.metric(label="Key Skills", value="84.3")
    st.caption("Highest performance area — strong foundational leadership competencies.")

with col5:
    st.metric(label="Useful Skills", value="80.2")
    st.caption("Supporting competencies — solid performance across leadership roles.")

with col6:
    st.metric(label="Supplemental Skills", value="81.1")
    st.caption("Contextual skills — unexpectedly strong performance indicators.")


st.subheader("Mentoring Typologies Drive the Highest Leadership Index Score (LIS")

fig2=plot_lis_by_typology(df)

st.plotly_chart(fig2)

st.subheader("The Skills Where N&P Excels the Most")
# Create three equal-width columns
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Goal Setting**")
    st.write("Exceptional competency with only 1 person requiring additional training.")

with col2:
    st.markdown("**Calculated Risk Taking**")
    st.write("Strong strategic decision-making with minimal training needs (1 person).")

with col3:
    st.markdown("**Supportiveness**")
    st.write("Excellence in team support and collaboration with just 1 person recommended for development.")

st.markdown("---")

import streamlit as st

st.subheader("Q2. What are the Skill Gaps?")

st.header("Key Findings")
st.markdown("- **100%** of leaders were identified with at least one flagged skill gap")
st.markdown("- On average, each leader had **7** targeted development areas")
st.markdown("- Majority of gaps found in **Key and Useful skill categories**")


st.markdown("---")

# Row: Critical Training Priorities
st.header("Critical Training Priorities Identified Across the Group")
st.write("Highlighting Immediate Development Focus Areas")

# Create a row of columns for the training priorities
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("**96%**")
    st.write("Leaders Need EQ Training")

with col2:
    st.markdown("**65%**")
    st.write("Leaders Need Accountability Training")

with col3:
    st.markdown("**46%**")
    st.write("Leaders Need Communication Training")

with col4:
    st.markdown("**33%**")
    st.write("Leaders Need Change Management Training")

st.subheader("Developmental Distribution of EQ: Widespread Need Development")

fig=plot_eq_leader_skills()
st.plotly_chart(fig, use_container_width=True)

# 30
    
st.subheader("Results: Engagement Behavior & Development Areas per Leader")

col1, col2 = st.columns(2)

with col1:
    st.metric(label="Engagement", value="54%")
    st.caption("Based on completion metrics and response time — 85 out of 100 leaders completed both discovery tools.")

with col2:
    st.metric(label="Avg Development Areas per Leader", value="7")
    st.caption("Most skill gaps identified in EQ, Communication, Accountability & Change Management.")


### TOOL

