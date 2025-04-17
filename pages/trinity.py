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
    plot_top_performers_by_skill,
    build_histogram_with_leaders
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
df['Overall Results']= round((df['Overall Results'] / 155) * 100)
df['EQ']=df['Overall Results']
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
df_filtered, selected_dashboards, selected_positions, selected_individuals, selected_type = dynamic_sidebar_filters(df)
filtered_resources=get_filtered_df(resource, selected_dashboards, selected_positions, selected_individuals, selected_type)
grouped = filtered_resources.groupby(['Leader', 'Skill']).agg(
    Score=('Score', 'last'),
    Below_Threshold_Count=('Below Threshold', 'last'),
    Skill_category=('Skill Category', 'last'),
    Dashboard=('# Dashboard', 'last'),
    Position=('Position', 'last'),
).reset_index().rename(columns={'Dashboard': '# Dashboard'})


df=df_filtered.copy()

dashboard_filter =df_filtered['Dashboard Number'].unique()
position_filter = df_filtered['Position'].unique()
leader_filter = df_filtered['Leader'].unique()

if len(dashboard_filter) == 12:
    subtitle = "All Dashboards"
else:
    subtitle = f"Dashboard: {', '.join(dashboard_filter)}"

# Shared annotation and margin
annotation = [
    dict(
        text=subtitle,
        x=0.5,
        y=1.08,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12),
        align="center"
    )
]
shared_layout = dict(
    title_font_size=18,
    annotations=annotation,
    margin=dict(t=80)  # Must be the same for both plots
)

st.title("Leadership Dashboard Analysis")

# ===============================
# 1. LIS Distribution - Boxplot & Histogram
# ===============================
st.subheader("LIS Score Distribution")

col1, col2 = st.columns(2)

with col1:
    box_fig = px.box(df, y='LIS', title="Boxplot of LIS Scores")
    box_fig.update_layout(**shared_layout)
    st.plotly_chart(box_fig, use_container_width=True)




# ===============================
# 5. Better Visualization of Typology Distribution
# ===============================
st.subheader("Distribution of Leadership Typologies")

if 'Typology 1' in df.columns:
    typology_counts = df['Typology 1'].value_counts().reset_index()
    typology_counts.columns = ['Typology', 'Count']
with col2:
    fig_typ_dist = px.pie(typology_counts, names='Typology', values='Count', title='Leadership Typology Distribution')
    fig_typ_dist.update_layout(**shared_layout)
    st.plotly_chart(fig_typ_dist, use_container_width=True)


with st.container():
    hist_fig=build_histogram_with_leaders(df_filtered)
    hist_fig.update_layout(**shared_layout)
    st.plotly_chart(hist_fig, use_container_width=True)




# ===============================
# 2. Performance by Skill Buckets
# ===============================
st.subheader("Performance on Skill Buckets")

melted = df.melt(
    id_vars=['Leader'],
    value_vars=['Key Skills', 'Necessary Skills', 'Beneficial Skills'],
    var_name='Skill Type',
    value_name='Score'
)

bucket_fig = px.box(melted, x='Skill Type', y='Score', title='Performance by Skill Buckets')
bucket_fig.update_layout(**shared_layout)
st.plotly_chart(bucket_fig, use_container_width=True)

# ===============================
# 3. Insights by Leadership Typology
# ===============================
if 'Typology 1' in df.columns:
    st.subheader("Insights by Leadership Type (Typology 1)")
    selected_type = st.multiselect("Select a Leadership Typology", df['Typology 1'].dropna().unique())

    filtered = df[df['Typology 1'].isin(selected_type)]

    if not filtered.empty:
        st.markdown(f"### Average Scores for {selected_type}")
        avg_scores = filtered[['LIS', 'Key Skills', 'Necessary Skills', 'Beneficial Skills', 'EQ']].mean().round(2).to_frame(name="Average Score")
        st.dataframe(avg_scores)
        st.dataframe(filtered[['Leader', 'Typology 1']])

        st.markdown("### LIS Distribution for Selected Typology")
        fig_typo = px.histogram(filtered, x='LIS', nbins=20, title=f"LIS Scores for {selected_type}", color_discrete_sequence=['indigo'])
        st.plotly_chart(fig_typo, use_container_width=True)
    else:
        st.warning("No data found for the selected typology.")
else:
    st.warning("Typology 1 column not found in the data.")



# ===============================
# 4. Score of Other Skills When a Skill is Selected
# ===============================
st.subheader("Compare Distribution of Selected Skills")

# Select one or multiple skills from numerical columns (excluding some metadata columns)
numeric_skills = [col for col in df.columns if col not in ['Leader', 'LIS', 'Typology 1'] and df[col].dtype in ['int64', 'float64']]
selected_skills = st.multiselect("Select Skills to Visualize", numeric_skills)

# Generate boxplots only if skills are selected
if selected_skills:
    skill_df = df[['Leader'] + selected_skills].dropna()
    melted_df = skill_df.melt(id_vars='Leader', value_vars=selected_skills, var_name='Skill', value_name='Score')

    fig_selected_skills = px.box(
        melted_df, x='Skill', y='Score',
        title="Boxplot of Selected Skills",
        points='all'
    )
    fig_selected_skills.update_layout(**shared_layout)
    st.plotly_chart(fig_selected_skills, use_container_width=True)


# ===============================
# 6. Correlation of LIS with Typology (Boxplot)
# ===============================
st.subheader("LIS Score by Leadership Typology")

if 'Typology 1' in df.columns:
    fig_corr = px.box(df, x='Typology 1', y='LIS', title="LIS Score Across Leadership Typologies", points='all')
    fig_corr.update_layout(**shared_layout)
    st.plotly_chart(fig_corr, use_container_width=True)

st.subheader("Strongest Skills (Average Score)")

# Weakest skills by average score
Strongest = (
    filtered_resources.groupby("Skill")["Score"]
    .mean()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_strong = px.bar(
    Strongest,
    x="Skill",
    y="Score",
    title="Top 10 Strongest Skills by Average Score",
    labels={"Score": "Average Score"},
    text_auto=True
)
fig_strong.update_layout(**shared_layout)
st.plotly_chart(fig_strong, use_container_width=True)

st.subheader("Weakest Skills (Average Score)")

# Weakest skills by average score
weakest_skills = (
    filtered_resources.groupby("Skill")["Score"]
    .mean()
    .sort_values()
    .head(10)
    .reset_index()
)

fig_weakest = px.bar(
    weakest_skills,
    x="Skill",
    y="Score",
    title="Top 10 Weakest Skills by Average Score",
    labels={"Score": "Average Score"},
    text_auto=True
)
fig_weakest.update_layout(**shared_layout)
st.plotly_chart(fig_weakest, use_container_width=True)

# Resource Type vs Skill Category
st.subheader("Resource Type vs Skill Category")

res_type_vs_skill_cat = filtered_resources.groupby(["Resource Type", "Skill Category"]).size().reset_index(name="Count")
fig_res_type_cat = px.bar(
    res_type_vs_skill_cat,
    x="Resource Type",
    y="Count",
    color="Skill Category",
    barmode="group",
    title="Resource Type vs Skill Category"
)
fig_res_type_cat.update_layout(**shared_layout)
st.plotly_chart(fig_res_type_cat, use_container_width=True)

# Allocation of Resources Based on Skills
st.subheader("Allocation of Resources Based on Skills")

skill_alloc = filtered_resources["Skill"].value_counts().reset_index()
skill_alloc.columns = ["Skill", "Resource Count"]

fig_alloc = px.bar(
    skill_alloc,
    x="Skill",
    y="Resource Count",
    title="Number of Resources Allocated per Skill",
    text_auto=True
)
fig_alloc.update_layout(**shared_layout)
st.plotly_chart(fig_alloc, use_container_width=True)



import plotly.express as px
import plotly.graph_objects as go

fig = px.box(df, x="Dashboard Number", y="LIS", points="all", color="Dashboard Number")
fig.add_hline(y=60, line_dash="dot", line_color="red")
fig.add_hline(y=70, line_dash="dot", line_color="orange")
fig.add_hline(y=85, line_dash="dot", line_color="green")
fig.update_layout(title="LIS Distribution by Dashboard", yaxis_title="LIS")

st.plotly_chart(fig, use_container_width=True)






fig = px.scatter(df, x="LIS", y="Key Skills", size="EQ", color="Dashboard Number",
                 hover_name="Leader", title="Bubble Plot: LIS vs Key Skills vs EQ")

st.plotly_chart(fig, use_container_width=True)



import seaborn as sns
import matplotlib.pyplot as plt

# Melt the DataFrame so each skill becomes a row under a single "Skill" column
melted_df = df.melt(id_vars=["# Dashboard"], 
                    value_vars=[
                        'Emotional Identification, Perception, and Expression', 
                        'Emotional Facilitation of Thought', 
                        'Emotional Understanding', 
                        'Emotional Management',
                        'Emotional Self-awareness',
                        'Awareness of Strengths and Limitations',
                        'Comfort with Emotions',
                        "Recognition of Other's Emotions",
                        'Rumination',
                        'Problem-Solving',
                        'Positive Mindset',
                        'Emotional Reflection',
                        'Emotional Integration',
                        'Conflict-Resolution Knowledge',
                        'Empathy',
                        'Social Insight',
                        'Self-Control',
                        'Resilience/Hardiness',
                        'Coping Skills',
                        'Self-Motivation',
                        'Striving',
                        'Emotional Selectivity',
                        'Adaptable Social Skills',
                        'Conflict-Resolution Behavior'
                    ],
                    var_name="Skill",
                    value_name="Score")

# Pivot the melted data
heatmap_data = melted_df.pivot_table(values="Score", index="Skill", columns="# Dashboard", aggfunc="mean")

# Plot heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="coolwarm", ax=ax, cbar_kws={'label': 'Average Score'})
plt.title("Average Skill Scores by Dashboard")
plt.xticks(rotation=45)
st.pyplot(fig)
 

import plotly.express as px
import plotly.graph_objects as go

# Step 1: Create LIS Band
def get_lis_band(lis):
    if lis < 70:
        return "<70"
    elif 70 <= lis < 75:
        return "70–75"
    elif 75 <= lis < 80:
        return "75–80"
    else:
        return "80+"

df["LIS Band"] = df["LIS"].apply(get_lis_band)

# Step 2: Define Skill Performance Buckets
def skill_bucket(score):
    if score < 70:
        return "Red (<70)"
    elif score < 80:
        return "Orange (70–79)"
    else:
        return "Green (80+)"

df["Skill Bucket"] = df["Key Skills"].apply(skill_bucket)

# Step 3: Aggregate
grouped = df.groupby(["LIS Band", "Skill Bucket"]).size().reset_index(name="Count")

# Optional sort LIS Bands
lis_band_order = ["<70", "70–75", "75–80", "80+"]
grouped["LIS Band"] = pd.Categorical(grouped["LIS Band"], categories=lis_band_order, ordered=True)
grouped = grouped.sort_values("LIS Band")

# Step 4: Plot using Plotly
fig = px.bar(
    grouped,
    x="LIS Band",
    y="Count",
    color="Skill Bucket",
    title="Leaders by LIS Band and Key Skills Performance",
    color_discrete_map={
        "Red (<70)": "#d62728",
        "Orange (70–79)": "#ff7f0e",
        "Green (80+)": "#2ca02c"
    },
    category_orders={"LIS Band": lis_band_order},
)

fig.update_layout(
    barmode="stack",
    xaxis_title="LIS Band",
    yaxis_title="Number of Leaders",
    legend_title="Skill Bucket"
)

st.plotly_chart(fig, use_container_width=True)
