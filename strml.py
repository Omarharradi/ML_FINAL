import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
import json
from plotly.subplots import make_subplots
import base64



# Set page configuration as the very first command
st.set_page_config(page_title="Leadership Competency Viewer", layout="wide")

st.markdown(
    """
    <style>
    /* Reduce the top padding of the main content container */
    .block-container {
        padding-top: 0.2rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

#hide_streamlit_style = """
#            <style>
#            /* Hide Streamlit hamburger menu */
#            #MainMenu {visibility: hidden;}
#            /* Hide Streamlit footer */
#            footer {visibility: hidden;}
#            footer:after {display: none;}
#            /* Optionally, hide the Streamlit header (if desired) */
#            header {visibility: hidden;}
#            </style>
#            """
#st.markdown(hide_streamlit_style, unsafe_allow_html=True)


# --- Data Setup (adjust paths as needed) ---
df = pd.read_csv('LDP_summary_anonymized.csv')
with open("skills_mapping_renamed.json", "r") as f:
    skills_mapping = json.load(f)
# --- End Data Setup ---

# Cache the filtered dataframe so that it doesn't re-run unnecessarily.
@st.cache_data(show_spinner=False)
def get_filtered_df(df, selected_dashboards, selected_positions):
    filtered = df.copy()
    if selected_dashboards:
        filtered = filtered[filtered["# Dashboard"].isin(selected_dashboards)]
    if selected_positions:
        filtered = filtered[filtered["Position"].isin(selected_positions)]
    return filtered

# Cache chart functions if desired
@st.cache_data(show_spinner=False)
def build_donut_chart(lis_data):
    mean_lis = np.mean(lis_data) if len(lis_data) > 0 else 0
    std_lis = np.std(lis_data) if len(lis_data) > 0 else 0
    std_low = np.mean(lis_data) - 1.5 * std_lis if len(lis_data) > 0 else 0
    std_high = np.mean(lis_data) + 1.5 * std_lis if len(lis_data) > 0 else 0

    leaders_meeting = np.sum((lis_data >= std_low) & (lis_data <= std_high)) if len(lis_data) > 0 else 0
    leaders_exceeding = np.sum(lis_data > std_high) if len(lis_data) > 0 else 0
    leaders_requiring_training = np.sum(lis_data < std_low) if len(lis_data) > 0 else 0

    donut_labels = ['Meeting Minimum Competency', 'Exceeding Expectations', 'Requiring Training']
    donut_values = [leaders_meeting, leaders_exceeding, leaders_requiring_training]
    donut_colors = ['#5c9acc', '#f4a300', '#e63946']
    
    fig = go.Figure(data=[go.Pie(
        labels=donut_labels,
        values=donut_values,
        hole=0.4,
        marker=dict(colors=donut_colors),
        hovertemplate="%{label}: %{value} leaders (%{percent})",
        textinfo='percent+label'
    )])
    fig.update_layout(
        title="Overall Leadership Competency Levels",
        margin=dict(t=50, b=50, l=50, r=50),
        width=700,
        height=500
    )
    return fig

@st.cache_data(show_spinner=False)
def build_histogram(lis_data):
    mean_lis = np.mean(lis_data) if len(lis_data) > 0 else 0
    std_lis = np.std(lis_data) if len(lis_data) > 0 else 0
    std_low = np.mean(lis_data) - 1.5 * std_lis if len(lis_data) > 0 else 0
    std_high = np.mean(lis_data) + 1.5 * std_lis if len(lis_data) > 0 else 0
    
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=lis_data,
        nbinsx=20,
        marker_color='darkblue',
        opacity=0.75,
        name='LIS'
    ))
    fig.add_vline(x=mean_lis, line=dict(color='gold', dash='dash'),
                   annotation_text=f'Mean: {mean_lis:.2f}', annotation_position="top right")
    fig.add_vline(x=std_low, line=dict(color='green', dash='dash'),
                   annotation_text=f'2-std below: {std_low:.2f}', annotation_position="top left")
    fig.add_vline(x=std_high, line=dict(color='green', dash='dash'),
                   annotation_text=f'2-std above: {std_high:.2f}', annotation_position="top right")
    fig.update_layout(
        title="LIS Distribution",
        xaxis_title="LIS Score",
        yaxis_title="Frequency",
        template="plotly_white",
        width=700,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig

# Radar chart function (not cached here because it depends on a single dashboard selection)
def radar_chart_plotly(dashboard, df, skills_mapping):
    # Extract skills per category for the given dashboard
    mapping = skills_mapping[dashboard]
    categories = ['Critical Skills', 'Necessary', 'Beneficial Skills']
    
    # Compute average scores per category per skill
    avg_scores = {}
    for cat in categories:
        skills = mapping.get(cat, [])
        avg_scores[cat] = [df.loc[df['# Dashboard'] == dashboard, skill].mean() 
                           for skill in skills if skill in df.columns]
    
    # Create subplots with polar type for each category
    fig = make_subplots(
        rows=1, cols=len(categories),
        specs=[[{'type': 'polar'} for _ in categories]],
        subplot_titles=[f"{dashboard} - {cat}" for cat in categories]
    )
    
    for i, cat in enumerate(categories):
        scores = avg_scores[cat]
        if not scores:
            continue  # Skip if no scores available for this category
        skills = mapping[cat]
        # Close the polygon by repeating the first score and label
        scores = np.array(scores)
        scores_closed = np.concatenate((scores, [scores[0]]))
        skills_closed = skills + [skills[0]]
        
        fig.add_trace(
            go.Scatterpolar(
                r = scores_closed,
                theta = skills_closed,
                fill='toself',
                mode='markers+lines',
                name=cat
            ),
            row=1, col=i+1
        )
        
        # Update the polar layout for this subplot
        fig.update_polars(
            dict(
                radialaxis=dict(visible=True, range=[0, max(scores_closed)*1.1]),
                angularaxis=dict(tickfont=dict(size=10))
            ),
            row=1, col=i+1
        )
    
    fig.update_layout(
        title_text=f"Radar Chart of Average Skill Scores for Dashboard {dashboard}",
        showlegend=False,
        width=350 * len(categories),
        height=500,
        margin=dict(t=100, b=50, l=50, r=50)
    )
    return fig

# --- Inject CSS for header alignment and ChatGPT-style sidebar chat ---
st.markdown(
    """
    <style>
    /* Header logos styling */
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .header-title {
        flex: 7;
    }
    .header-logo {
        flex: 1;
        text-align: right;
    }
    .header-logo img {
        max-width: 100%;
        height: auto;
    }
    /* Chat sidebar styling (ChatGPT-like) */
    .chat-container {
        background-color: #F7F7F8;
        padding: 15px;
        border-radius: 10px;
        max-height: 300px;
        overflow-y: auto;
        font-family: 'Helvetica', sans-serif;
    }
    .chat-message {
        margin-bottom: 10px;
        padding: 8px;
        border-radius: 5px;
    }
    .user-message {
        background-color: #DCF8C6;
        text-align: right;
    }
    .assistant-message {
        background-color: #FFFFFF;
        text-align: left;
    }
    </style>
    """, unsafe_allow_html=True
)


def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


st.markdown(
    f"""
    <div class="header-container">
      <div class="header-title">
         <h1>Leadership Results Overview</h1>
      </div>
      <div class="header-logo">
      </div>
      <div class="header-logo">
      </div>
    </div>
    """,
    unsafe_allow_html=True
)


st.markdown("""
This app displays interactive visualizations based on overall leadership competency levels.
""")

# --------------------------------------
# Sidebar: Filtering System, Navigation, and Chat Assistant
# --------------------------------------
with st.sidebar:
    st.header("Filters")
    selected_dashboards = st.multiselect("Select Dashboard(s)", sorted(df["# Dashboard"].unique()))
    if "Position" in df.columns:
        selected_positions = st.multiselect("Select Position(s)", sorted(df["Position"].unique()))
    else:
        selected_positions = []
    
    st.markdown("---")
    st.header("Navigation")
    page = st.radio("Go to", ["Overall Results", "Resources Summary", "Next Steps"])
    
    st.markdown("---")
    st.header("Chat Assistant (Beta)")
    chat_input = st.text_input("Your message:")
    if st.button("Send", key="chat_send"):
        st.markdown('<div class="chat-message assistant-message">Placeholder response: LLM backend not implemented.</div>', unsafe_allow_html=True)
    if chat_input:
        st.markdown(f'<div class="chat-message user-message">{chat_input}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------------
# Filter the DataFrame based on user selections (applied on all pages)
# -------------------------------
filtered_df = get_filtered_df(df, selected_dashboards, selected_positions)

# -------------------------------
# Compute Overall Leadership Metrics (LIS) using filtered_df
# -------------------------------
lis_data = filtered_df['LIS']
mean_lis = np.mean(lis_data) if len(lis_data) > 0 else 0
std_lis = np.std(lis_data) if len(lis_data) > 0 else 0
std_low = np.mean(lis_data) - 1.5 * std_lis if len(lis_data) > 0 else 0
std_high = np.mean(lis_data) + 1.5 * std_lis if len(lis_data) > 0 else 0

leaders_meeting = np.sum((lis_data >= std_low) & (lis_data <= std_high)) if len(lis_data) > 0 else 0
leaders_exceeding = np.sum(lis_data > std_high) if len(lis_data) > 0 else 0
leaders_requiring_training = np.sum(lis_data < std_low) if len(lis_data) > 0 else 0

# Data for the donut chart
donut_labels = ['Meeting Minimum Competency', 'Exceeding Expectations', 'Requiring Training']
donut_values = [leaders_meeting, leaders_exceeding, leaders_requiring_training]
donut_colors = ['#5c9acc', '#f4a300', '#e63946']

# -------------------------------
# Build Plotly Donut Chart
# -------------------------------
donut_fig = go.Figure(data=[go.Pie(
    labels=donut_labels,
    values=donut_values,
    hole=0.4,
    marker=dict(colors=donut_colors),
    hovertemplate="%{label}: %{value} leaders (%{percent})",
    textinfo='percent+label',
    textposition='inside',  # Moves text inside the donut chart
   # insidetextorientation='radial',  # Keeps text readable
   # textfont=dict(size=12)  
)])
donut_fig.update_layout(
    title="Overall Leadership Competency Levels",
    margin=dict(t=50, b=50, l=50, r=50),
    width=700,
    height=500
)

# -------------------------------
# Build Plotly Histogram for LIS Distribution
# -------------------------------
hist_fig = go.Figure()
hist_fig.add_trace(go.Histogram(
    x=lis_data,
    nbinsx=20,
    marker_color='darkblue',
    opacity=0.75,
    name='LIS'
))
hist_fig.add_vline(x=mean_lis, line=dict(color='gold', dash='dash'),
                   annotation_text=f'Mean: {mean_lis:.2f}', annotation_position="top right")
hist_fig.add_vline(x=std_low, line=dict(color='green', dash='dash'),
                   annotation_text=f'2-std below: {std_low:.2f}', annotation_position="top left")
hist_fig.add_vline(x=std_high, line=dict(color='green', dash='dash'),
                   annotation_text=f'2-std above: {std_high:.2f}', annotation_position="top right")
hist_fig.update_layout(
    title="LIS Distribution",
    xaxis_title="LIS Score",
    yaxis_title="Frequency",
    template="plotly_white",
    width=700,
    height=400,
    margin=dict(t=50, b=50, l=50, r=50)
)

# -------------------------------
# Build Plotly Polar Bar Chart for Average Overall Results per Dashboard
# -------------------------------
avg_overall = filtered_df.groupby("# Dashboard")["LIS"].mean().reset_index()
fig_polar = px.bar_polar(
    avg_overall,
    r="LIS",
    theta="# Dashboard",
    color="# Dashboard",
    template="plotly_white",
    title="Average Overall LIS by Dashboard (Polar Bar Chart)",
    color_discrete_sequence=px.colors.qualitative.Bold
)
fig_polar.update_layout(
    margin=dict(l=50, r=50, t=100, b=50),
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, avg_overall["LIS"].max() * 1.1]
        )
    )
)

# -------------------------------
# Build Plotly Box Plot for Overall Results by Dashboard
# -------------------------------
fig_box = px.box(
    filtered_df,
    x="# Dashboard",
    y="Overall Results",
    title="Box Plot of EQ by Dashboard",
    template="plotly_white",
    color="# Dashboard",
    color_discrete_sequence=px.colors.qualitative.Pastel
)

# -------------------------------
# Main Page Content
# -------------------------------
if page == "Overall Results":
    st.header("Overall Results")
    # First row: Donut chart and Histogram side by side
    col1_row, col2_row = st.columns(2)
    with col1_row:
        st.plotly_chart(donut_fig, use_container_width=True)
    with col2_row:
        st.plotly_chart(hist_fig, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Additional Analysis")
    # Second row: Polar bar chart and Box plot side by side
    col3_row, col4_row = st.columns(2)
    with col3_row:
        st.plotly_chart(fig_polar, use_container_width=True)
    with col4_row:
        st.plotly_chart(fig_box, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Radar Chart")
    # Third row: Radar chart
    # Create a selectbox for the dashboard used in the radar chart.
    # If the filter results in multiple dashboards, we allow the user to pick one.
    unique_dashboards = sorted(filtered_df["# Dashboard"].unique())
    if unique_dashboards:
        selected_radar = st.selectbox("Select Dashboard for Radar Chart", unique_dashboards)
        radar_fig = radar_chart_plotly(selected_radar, filtered_df, skills_mapping)
        st.plotly_chart(radar_fig, use_container_width=True)
    else:
        st.write("No data available for the selected filters.")

elif page == "Resources Summary":
    st.header("Resources Summary")
    st.markdown("""
    **Resources Summary** is a placeholder page.
    
    Here you can add:
    
    - Detailed summaries of training materials
    
    """)

elif page == "Next Steps":
    st.header("Next Steps")
    st.markdown("""
    **Next Steps** is a placeholder page.
    
    Suggested next steps:
    
    1. Analyze the data trends and identify areas of improvement.
    2. Develop targeted training programs based on competency gaps.

    
    *Add additional actionable items and recommendations here.*
    """)
