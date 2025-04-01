# utils.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st  
import plotly.express as px

@st.cache_data(show_spinner=False)
def get_filtered_df(df, selected_dashboards, selected_positions):
    filtered = df.copy()
    if selected_dashboards:
        filtered = filtered[filtered["# Dashboard"].isin(selected_dashboards)]
    if selected_positions:
        filtered = filtered[filtered["Position"].isin(selected_positions)]
    return filtered


@st.cache_data(show_spinner=False)
def build_donut_chart(lis_data):
    ''' 
    LIS Stands for Leadership Index Score which is a weighted score of critical skills necessary skills and beneficial skills to have a standardised metric to compare all individuals
    EQ assesses Emotional Intelligence which is one of the most important skills that all leaders are assessed in.
    '''
    mean_lis = np.mean(lis_data)
    std_lis = np.std(lis_data)
    std_low = mean_lis - 1.5 * std_lis
    std_high = mean_lis + 1.5 * std_lis

    leaders_meeting = np.sum((lis_data >= std_low) & (lis_data <= std_high))
    leaders_exceeding = np.sum(lis_data > std_high)
    leaders_requiring_training = np.sum(lis_data < std_low)

    labels = ['Meeting Minimum Competency', 'Exceeding Expectations', 'Requiring Training']
    values = [leaders_meeting, leaders_exceeding, leaders_requiring_training]
    colors = ['#5c9acc', '#f4a300', '#e63946']

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors),
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
    ''' 
    LIS Stands for Leadership Index Score which is a weighted score of critical skills necessary skills and beneficial skills to have a standardised metric to compare all individuals
    EQ assesses Emotional Intelligence which is one of the most important skills that all leaders are assessed in.
    '''
    mean_lis = np.mean(lis_data)
    std_lis = np.std(lis_data)
    std_low = mean_lis - 1.5 * std_lis
    std_high = mean_lis + 1.5 * std_lis

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
                  annotation_text=f'1.5-std below: {std_low:.2f}', annotation_position="top left")
    fig.add_vline(x=std_high, line=dict(color='green', dash='dash'),
                  annotation_text=f'1.5-std above: {std_high:.2f}', annotation_position="top right")
    fig.update_layout(
        title="Leadership Index Score (LIS) Distribution",
        xaxis_title="Leadership Index Score (LIS) Score",
        yaxis_title="Frequency",
        template="plotly_white",
        width=700,
        height=400,
        margin=dict(t=50, b=50, l=50, r=50)
    )
    return fig

@st.cache_data(show_spinner=False)
def radar_chart_plotly(dashboard, df, skills_mapping):
    ''' 
    LIS Stands for Leadership Index Score which is a weighted score of critical skills necessary skills and beneficial skills to have a standardised metric to compare all individuals
    EQ assesses Emotional Intelligence which is one of the most important skills that all leaders are assessed in.
    '''
    mapping = skills_mapping[dashboard]
    categories = ['Critical Skills', 'Necessary', 'Beneficial Skills']

    avg_scores = {}
    for cat in categories:
        skills = mapping.get(cat, [])
        avg_scores[cat] = [df.loc[df['# Dashboard'] == dashboard, skill].mean()
                           for skill in skills if skill in df.columns]

    fig = make_subplots(
        rows=1, cols=len(categories),
        specs=[[{'type': 'polar'} for _ in categories]],
        subplot_titles=[f"{dashboard} - {cat}" for cat in categories]
    )

    for i, cat in enumerate(categories):
        scores = avg_scores[cat]
        if not scores:
            continue
        skills = mapping[cat]
        scores = np.array(scores)
        scores_closed = np.concatenate((scores, [scores[0]]))
        skills_closed = skills + [skills[0]]

        fig.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=skills_closed,
                fill='toself',
                mode='markers+lines',
                name=cat
            ),
            row=1, col=i+1
        )
        fig.update_polars(
            dict(
                radialaxis=dict(visible=True, range=[0, max(scores_closed) * 1.1]),
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

@st.cache_data(show_spinner=False)
def build_polar_chart(df):
    ''' 
    LIS Stands for Leadership Index Score which is a weighted score of critical skills necessary skills and beneficial skills to have a standardised metric to compare all individuals
    EQ assesses Emotional Intelligence which is one of the most important skills that all leaders are assessed in.
    '''
    avg_overall = df.groupby("Dashboard Number")["LIS"].mean().reset_index()

    fig = px.bar_polar(
        avg_overall,
        r="LIS",
        theta="Dashboard Number",
        color="Dashboard Number",
        template="plotly_white",
        title="Average Overall Leadership Index Score (LIS) by Dashboard (Polar Bar Chart)",
        color_discrete_sequence=px.colors.qualitative.Bold
    )

    fig.update_layout(
        margin=dict(l=50, r=50, t=100, b=50),
        polar=dict(
            radialaxis=dict(visible=True, range=[0, avg_overall["LIS"].max() * 1.1])
        )
    )
    return fig

@st.cache_data(show_spinner=False)
def build_box_plot(df):
    ''' 
    LIS Stands for Leadership Index Score which is a weighted score of critical skills necessary skills and beneficial skills to have a standardised metric to compare all individuals
    EQ assesses Emotional Intelligence which is one of the most important skills that all leaders are assessed in.
    '''
    fig = px.box(
        df,
        x="Dashboard Number",
        y="Overall Results",
        title="Box Plot of EQ by Dashboard",
        template="plotly_white",
        color="Dashboard Number",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    return fig

def get_insights_chart(lis_data, source_code, llm):
    insights_template = PromptTemplate(
        input_variables=["Dataframe", "PlotCode"],
        template="""You are a data analyst helping decision-makers understand data trends.

    Step 1: Analyze the PlotCode and DataFrame to understand what the chart is visualizing. Do NOT include or describe the code itself.

    Step 2: Return your output in **Markdown** using the following structure:

    **Chart Description**
    \n
    A brief paragraph that explains what the chart is visualizing and why it's relevant.

    **Chart Insights**
    - Use concise bullet points to highlight key patterns, trends, or outliers in the data.
    - Make the language natural and insightful, not robotic.

    **Chart Actionables**
    - Suggest meaningful next steps or considerations based on the data.
    - Actionables should feel strategic and helpful to someone using the chart to make decisions.

    Important Guidelines:
    - If a KPI, variable, or concept is not clearly defined in the DataFrame or PlotCode, do NOT guess its meaning. It may represent proprietary or internal data.
    - Never fabricate interpretations — only base your insights on what is explicitly observable from the data and plot logic.
    - Be clear, thoughtful, and avoid assumptions.

    Only return the final Markdown-formatted insight (no explanations, no code).

    DataFrame:
    {Dataframe}

    PlotCode:
    {PlotCode}
    """
    )

    #llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
    chain = insights_template | llm | StrOutputParser()
    response = chain.invoke({"Dataframe": lis_data, "PlotCode": source_code})
    return response