
# utils.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st  
import plotly.express as px
import os
import base64




def get_filtered_df(df, selected_dashboards, selected_positions, selected_individuals, selected_type):
    filtered = df.copy()
    if selected_dashboards:
        filtered = filtered[filtered["# Dashboard"].isin(selected_dashboards)]
    if selected_positions:
        filtered = filtered[filtered["Position"].isin(selected_positions)]
    if selected_individuals:
        filtered = filtered[filtered["Leader"].isin(selected_individuals)]
    if selected_type:
        filtered = filtered[filtered["Typology 1"].isin(selected_type)]
    return filtered


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
    
    leaders_summary = {
    "meeting_standard": np.sum((lis_data >= std_low) & (lis_data <= std_high)),
    "exceeding_standard": np.sum(lis_data > std_high),
    "requiring_training": np.sum(lis_data < std_low)
    }

       # Original full data
    full_labels = ['Demonstrating Role <br>Proficiency', 'Surpassing Role Proficiency', 'Requiring Training']
    full_values = [leaders_meeting, leaders_exceeding, leaders_requiring_training]
    full_colors = ['#5c9acc', '#f4a300', '#e63946']

    # Filter out 0% categories
    filtered_labels = []
    filtered_values = []
    filtered_colors = []
    for label, value, color in zip(full_labels, full_values, full_colors):
        if value > 0:
            filtered_labels.append(label)
            filtered_values.append(value)
            filtered_colors.append(color)

    fig = go.Figure(data=[go.Pie(
        labels=filtered_labels,
        values=filtered_values,
        hole=0.4,
        marker=dict(colors=filtered_colors),
        hovertemplate="%{label}: %{value} leaders (%{percent})",
        textinfo='percent+label',
        textposition='outside',
        rotation=100
    )])

    fig.update_layout(
        title="Overall Leadership Competency Levels",
        margin=dict(t=80, b=80, l=50, r=50),
        width=700,
        height=400,
        showlegend=False  # <-- this line removes the legend
    )

    return fig, leaders_summary


def build_histogram(lis_data):
    ''' 
    LIS Stands for Leadership Index Score which is a weighted score of critical skills necessary skills and beneficial skills to have a standardised metric to compare all individuals
    EQ assesses Emotional Intelligence which is one of the most important skills that all leaders are assessed in.
    '''
    mean_lis = np.mean(lis_data)
    std_lis = np.std(lis_data)
    std_low = mean_lis - 1.5 * std_lis
    std_high = mean_lis + 1.5 * std_lis

    below_std = np.sum(lis_data < std_low)
    above_std = np.sum(lis_data > std_high)

    # Summary dictionary
    summary_stats = {
        "mean": mean_lis,
        "std_dev": std_lis,
        "1.5_std_below": std_low,
        "1.5_std_above": std_high,
        "min_value": np.min(lis_data),
        "max_value": np.max(lis_data),
        "count": len(lis_data),
        "below_1.5_std": below_std,
        "above_1.5_std": above_std
    }
    

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
    return fig, summary_stats

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
def radar_chart_plotly(dashboard, leader, df, skills_mapping):
    '''
    Compare average dashboard skill scores and individual leader's skill scores.
    If leader is "None", only show the dashboard average scores.
    '''
    mapping = skills_mapping.get(dashboard, {})
    categories = ['Critical Skills', 'Necessary', 'Beneficial Skills']

    summary_stats = {
        "dashboard": dashboard,
        "leader": leader,
        "skill_scores": {}
    }

    color_palette = px.colors.qualitative.Pastel
    category_colors = {
        "Average": color_palette[0],
        "Leader": color_palette[1]
    }

    fig = make_subplots(
        rows=1, cols=len(categories),
        specs=[[{'type': 'polar'} for _ in categories]],
        subplot_titles=[f"{cat}" for cat in categories],
    )

    radial_range = [0, 100]
    tickvals = [0, 40, 80]

    for i, cat in enumerate(categories):
        skills = mapping.get(cat, [])
        if not skills:
            continue

        # Average dashboard scores
        avg_scores = [df.loc[df['# Dashboard'] == dashboard, skill].mean() for skill in skills if skill in df.columns]
        
        # If a leader is selected, get their scores; otherwise, set leader scores to None
        if leader != "None":
            leader_scores = [df.loc[(df['# Dashboard'] == dashboard) & (df['Leader'] == leader), skill].values[0]
                             if skill in df.columns else None for skill in skills]
        else:
            leader_scores = [None] * len(skills)  # No leader selected, show no leader scores

        summary_stats["skill_scores"][cat] = {
            "average": avg_scores,
            "leader": leader_scores
        }

        # Format skill names for radial labels
        formatted_skills = [
            skill.replace(" ", "<br>").replace("-", "<br>") if "&" not in skill
            else skill.replace(" &", "&").replace("-", "<br>").replace(" ", "<br>").replace("&", " &")
            for skill in skills
        ]
        skills_closed = formatted_skills + [formatted_skills[0]]

        # Close the loop for radar plot
        avg_scores_closed = np.concatenate((avg_scores, [avg_scores[0]]))
        leader_scores_closed = np.concatenate((leader_scores, [leader_scores[0]]))

        # Add average trace
        fig.add_trace(
            go.Scatterpolar(
                r=avg_scores_closed,
                theta=skills_closed,
                fill='toself',
                mode='lines+markers',
                name=f"Dashboard Avg",
                marker=dict(color=category_colors["Average"])
            ),
            row=1, col=i+1
        )

        # Only add the leader trace if a leader is selected
        if leader != "None":
            fig.add_trace(
                go.Scatterpolar(
                    r=leader_scores_closed,
                    theta=skills_closed,
                    fill='toself',
                    mode='lines+markers',
                    name=f"{leader} (Leader)",
                    marker=dict(color=category_colors["Leader"])
                ),
                row=1, col=i+1
            )

        # Update radar settings
        fig.update_polars(
            dict(
                radialaxis=dict(visible=True, range=radial_range, tickvals=tickvals, showline=True),
                angularaxis=dict(
    rotation=50,
    tickfont=dict(size=8),
    direction="clockwise"
)
            ),
            row=1, col=i+1
        )

    fig.update_layout(
        title_text=f"Radar Chart: {leader} vs Dashboard Avg ({dashboard})" if leader != "None" else f"Radar Chart: {dashboard} (No Leader)",
        showlegend=True,
        width=350 * len(categories),
        height=400,
        margin=dict(t=80, b=0, l=60, r=100),
        legend=dict(
    orientation="h",        # horizontal legend
    yanchor="bottom",
    y=-0.2,
    xanchor="center",
    x=0.5,
    title_text='',          # remove default legend title
    font=dict(size=10)
)
    )

    return fig, summary_stats



def build_polar_chart(df):
    ''' 
    LIS Stands for Leadership Index Score which is a weighted score of critical skills necessary skills and beneficial skills to have a standardised metric to compare all individuals
    EQ assesses Emotional Intelligence which is one of the most important skills that all leaders are assessed in.
    '''
    avg_overall = df.groupby("Dashboard Number")["LIS"].mean().reset_index()
    avg_overall_dict=avg_overall.to_dict(orient='records')

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
            radialaxis=dict(
                visible=True,
                range=[0, 100],  # Set fixed range
                tickvals=[90],  # Only show tick at 100
                ticktext=["90"],  # Display '100' only
                showline=False
            )
        )
    )
    return fig, avg_overall_dict

def build_box_plot(df):
    ''' 
    LIS Stands for Leadership Index Score which is a weighted score of critical skills necessary skills and beneficial skills to have a standardised metric to compare all individuals
    EQ assesses Emotional Intelligence which is one of the most important skills that all leaders are assessed in.
    '''

    summary_stats = df.groupby("Dashboard Number")["Overall Results"].describe().reset_index()

    fig = px.box(
        df,
        x="Dashboard Number",
        y="Overall Results",
        title="Box Plot of EQ by Dashboard",
        template="plotly_white",
        color="Dashboard Number",
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    return fig, summary_stats

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
    
    Important Guidelines:
    - If a KPI, variable, or concept is not clearly defined in the DataFrame or PlotCode, do NOT guess its meaning. It may represent proprietary or internal data.
    - Never fabricate interpretations — only base your insights on what is explicitly observable from the data and plot logic.
    - Be clear, thoughtful, and avoid assumptions.

    Only return the final Markdown-formatted insight.

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



# utils.py


import os
import json
import pandas as pd
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA, LLMChain
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
def initialize_pipeline(json_path="result.json"):
    """
    Initializes the pipeline by:
      - Loading the JSON data and converting it to a DataFrame.
      - Creating Document objects for RAG.
      - Setting up the LLM, embeddings, vector store, retriever, and chains.
      - Creating a classifier chain and a unified ask() function.
      
    Returns:
      ask (function): A function that accepts a query string and returns an answer.
    """
    # -----------------------------
    # Load & Prepare JSON Data
    # -----------------------------
    with open(json_path, "r") as f:
        data = json.load(f)

    records = []
    docs = []

    for id_, entry in data.items():
        flat = {"ID": id_}
        flat.update(entry)
        records.append(flat)

        # Create a text version for RAG documents
        text = f"ID: {id_}\n"
        for k, v in entry.items():
            if isinstance(v, list):
                text += f"{k}: {', '.join(map(str, v))}\n"
            else:
                text += f"{k}: {v}\n"
        docs.append(Document(page_content=text))

    df = pd.DataFrame(records)
    df=pd.read_csv('LDP_summary.csv')

    # -----------------------------
    # LLM & Embeddings
    # -----------------------------
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest")
    embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # -----------------------------
    # RAG Setup
    # -----------------------------
    persist_dir = "./chroma_db"
    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        # Load existing vector store
        vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embedding, collection_name="ldp_docs")
    else:
        # Create a new vector store and persist it
        vectorstore = Chroma.from_documents(docs, embedding, collection_name="ldp_docs", persist_directory=persist_dir)
        vectorstore.persist()
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

    # Custom prompt for the RAG chain
    context_prompt = PromptTemplate.from_template("""
You are a data assistant helping analyze leadership assessment data.

Use the provided context to answer the question below. The context includes information on individuals, dashboards, and their skills.

Context:
{context}

Question:
{question}

Answer by combining insights across individuals. Focus on patterns, averages, and summaries. If no useful context is found, say "Not enough information."
""")
    llm_chain = LLMChain(llm=llm, prompt=context_prompt)

    # Using StuffDocumentsChain to combine documents into context
    from langchain.chains.combine_documents.stuff import StuffDocumentsChain as CombineStuffDocumentsChain
    stuff_chain = CombineStuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)


    rag_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=stuff_chain,
        memory=memory
    )

    # -----------------------------
    # Pandas Agent Setup
    # -----------------------------
    pandas_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=df,
        verbose=True,
        allow_dangerous_code=True
    )

    # -----------------------------
    # Classifier to Route Query
    # -----------------------------
    classifier_prompt = PromptTemplate.from_template("""
You are a smart classifier. Given a user question, decide if it should be handled using:

- "structured" → if it's about rankings, filters, math, comparisons, scores
- "semantic" → if it's about summaries, meaning, types of skills, descriptions

Return ONLY the word "structured" or "semantic".

Question: {query}
""")
    classifier_chain = LLMChain(llm=llm, prompt=classifier_prompt)

    # -----------------------------
    # Unified Ask Function
    # -----------------------------
    def ask(query: str) -> str:
        route = classifier_chain.run({"query": query}).strip().lower()
        print(f"[Routing → {route}]")
        
        if route == "structured":
            # Run the pandas agent to get the raw output including its chain trace.
            raw_result = pandas_agent.run(query)
            # Use an LLM chain to rephrase the full chain trace into a natural response.
            rephrase_prompt = PromptTemplate.from_template("""
You are an expert data analyst and storyteller. Below is the full trace of how a pandas agent processed a query, including its thoughts, actions, and the final answer.

Full Agent Trace:
{raw_result}

The user's original question was:
"{query}"

Based on this trace, generate a natural, complete sentence that states the final answer along with all relevant details. 
In particular, if the agent identified a person with the highest LIS and a numerical score, include the person's full name, the exact LIS score, and any other useful context.
For example, instead of just saying "William Smullen", your answer should be like: "William Smullen achieved the highest LIS of 92.615."
""")
            rephrase_chain = LLMChain(llm=llm, prompt=rephrase_prompt)
            natural_result = rephrase_chain.run({"raw_result": raw_result, "query": query})
            
            return natural_result.strip()
        elif route == "semantic":
            return rag_chain.run(query)
        else:
            return "Sorry, I couldn't confidently classify your question."

    return ask


import inspect
import streamlit as st

def display_insight(unique_key, build_func, lis_data, llm, expander_title=None):
    """
    Generates and displays insights for a given chart using session state.

    Parameters:
    - unique_key (str): A unique identifier for the chart (e.g., "donut", "hist", etc.).
    - build_func (function): The chart function whose source code will be used.
    - lis_data: Data to be passed to the get_insights_chart function.
    - llm: The LLM instance to generate insights.
    - expander_title (str, optional): Title for the expander. Defaults to "Insights for {unique_key}".
    """
    # Ensure that the session state dictionary exists
    if "insights" not in st.session_state:
        st.session_state["insights"] = {}

    # Generate insight when the button is pressed
    if st.button("🧠 Explain this graph", key=f"explain_{unique_key}"):
        source_code = inspect.getsource(build_func)
        with st.spinner("Generating insights..."):
            response = get_insights_chart(lis_data, source_code, llm)
        st.session_state["insights"][unique_key] = response

    # Display the saved insight, if it exists
    if unique_key in st.session_state["insights"]:
        title = expander_title if expander_title else f"Insights for {unique_key}"
        with st.expander(title, expanded=True):
            st.info(st.session_state["insights"][unique_key])


def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()


def dynamic_sidebar_filters(df):
    """Render dynamic sidebar filters and return the filtered DataFrame + current selections."""
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

        st.markdown("---")

        st.header("Filters")

        # 1. Dashboard filter (no dependencies at this point)
        all_dashboards = sorted(df["# Dashboard"].dropna().unique())
        selected_dashboards = st.multiselect("Select Dashboard(s)", all_dashboards)

        # 2. Filter DataFrame based on selected dashboards (if any)
        df_dash_filtered = df[df["# Dashboard"].isin(selected_dashboards)] if selected_dashboards else df

        # 3. Position filter based on dashboards
        position_options = sorted(df_dash_filtered["Position"].dropna().unique()) if "Position" in df.columns else []
        selected_positions = st.multiselect("Select Position(s)", position_options)

        # 4. Further filter DataFrame based on positions
        df_pos_filtered = df_dash_filtered[df_dash_filtered["Position"].isin(selected_positions)] if selected_positions else df_dash_filtered

        # 5. Individual filter based on dashboards + positions
        individual_options = sorted(df_pos_filtered["Leader"].dropna().unique()) if "Leader" in df.columns else []
        selected_individuals = st.multiselect("Select Leader(s)", individual_options)

        df_type_filtered = df_dash_filtered[df_dash_filtered["Leader"].isin(selected_individuals)] if selected_individuals else df_dash_filtered

        # 5. Individual filter based on dashboards + positions
        type_options = sorted(df_type_filtered["Typology 1"].dropna().unique()) if "Typology 1" in df.columns else []
        selected_type = st.multiselect("Select Type(s)", type_options)

    # --- Final filtering: apply all selected filters together ---
    df_filtered = df.copy()

    if selected_dashboards:
        df_filtered = df_filtered[df_filtered["# Dashboard"].isin(selected_dashboards)]
    if selected_positions:
        df_filtered = df_filtered[df_filtered["Position"].isin(selected_positions)]
    if selected_individuals:
        df_filtered = df_filtered[df_filtered["Leader"].isin(selected_individuals)]
    if selected_type:
        df_filtered = df_filtered[df_filtered["Typology 1"].isin(selected_type)]

    return df_filtered, selected_dashboards, selected_positions, selected_individuals, selected_type





# insight_charts.py


def plot_resource_type_distribution(df):
    """
    Plots a stacked bar chart showing the distribution of resource types per skill category.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'Skill Category' and 'Resource Type' columns.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): Plotly figure object.
    """
    resource_type_by_category = df.groupby(['Skill Category', 'Resource Type']).size().reset_index(name='Count')
    fig = px.bar(resource_type_by_category, x='Skill Category', y='Count', color='Resource Type',
                 title='Distribution of Resource Types per Skill Category', barmode='stack')
    return fig, resource_type_by_category

def plot_top_skills(df, top_n=10):
    """
    Plots a bar chart of the top N most frequently targeted skills.

    Parameters:
    - df (pd.DataFrame): DataFrame containing a 'Skill' column.
    - top_n (int): Number of top skills to display (default is 10).

    Returns:
    - fig (plotly.graph_objs._figure.Figure): Plotly figure object.
    """
    skill_leader_counts = (
        df.groupby('Skill')['Leader']
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .reset_index()
        .rename(columns={'Leader': 'Leader Count'})
    )

    fig = px.bar(
        skill_leader_counts,
        x='Skill',
        y='Leader Count',
        title=f'Top {top_n} Skills by Number of Unique Leaders Assessed',
        labels={'Leader Count': 'Number of Leaders'},
    )
    return fig, skill_leader_counts




def plot_resource_distribution_by_category(df):
    """
    Plots a pie chart showing the distribution of total resources per skill category.

    Parameters:
    - df (pd.DataFrame): DataFrame containing 'Skill Category' and 'Resource Text' columns.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): Plotly figure object.
    """
    skill_cat_resources = df.groupby('Skill Category')['Resource Text'].count().reset_index(name='Total Resources')
    fig = px.pie(skill_cat_resources, names='Skill Category', values='Total Resources',
                 title='Resource Distribution by Skill Category')
    return fig, skill_cat_resources



def plot_top_resources_by_category(df, category, top_n=8):
    # Filter by the selected resource category
    filtered_df = df[df['Resource Type'] == category]

    if filtered_df.empty:
        print(f"No resources found for category: {category}")
        return

    # Count most frequent Resource Texts
    top_resources = (
        filtered_df['Resource Text']
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    top_resources.columns = ['Resource Text', 'Count']  # Explicitly rename columns

    # Plot with Plotly
    fig = px.bar(
        top_resources,
        x='Count',
        y='Resource Text',
        orientation='h',
        title=f"Top {top_n} Most Recurrent Resources in '{category}'",
        height=500
    )

    fig.update_layout(yaxis=dict(autorange='reversed'))  # highest at top
    return fig, top_resources


def plot_leaders_below_threshold(group, top_n=10, selected_skill=None):
    """
    Plots a vertical bar chart of the top N leaders with the most skills below threshold.

    Parameters:
    - group (pd.DataFrame): DataFrame containing 'Leader', 'Below_Threshold_Count', and 'Skill'.
    - top_n (int): Number of top leaders to display (default is 10).
    - selected_skill (str, optional): Specific skill to filter by. If None, shows all skills.

    Returns:
    - fig (plotly.graph_objs._figure.Figure): Plotly figure object.
    """
    
    # Filter by the selected skill (if provided)
    if selected_skill:
        filtered_group = group[group['Skill'] == selected_skill]
    else:
        filtered_group = group

    # Get leaders with skills below threshold
    leaders_below = filtered_group[filtered_group['Below_Threshold_Count'] == True]['Leader'].value_counts().reset_index()
    leaders_below.columns = ['Leader', 'Below Threshold Count']

    # Get the top N leaders with the most skills below threshold
    leaders_below = leaders_below.head(top_n)

    # Create the vertical bar chart
    fig = px.bar(leaders_below, x='Leader', y='Below Threshold Count',
                 title=f'Top {top_n} Leaders with the Most Skills Below Threshold',
                 color='Below Threshold Count')
    
    # Update layout to make the chart vertical
    fig.update_layout(
        xaxis_title='Leader',
        yaxis_title='Below Threshold Count',
        xaxis={'tickangle': 45},  # Rotate x-axis labels for better readability
        height=500
    )
    
    return fig, leaders_below




import pandas as pd
import plotly.express as px

def plot_top_performers_by_skill(df, skill=None, top_n=5, bar_color="#636EFA"):
    """
    Plots a clean bar chart showing the top N unique performers for a selected skill.

    Parameters:
    - df (pd.DataFrame): DataFrame containing at least 'Leader', 'Skill', and 'Score' columns.
    - skill (str or None): Specific skill to filter by. If None, no plot is returned.
    - top_n (int): Number of top performers to display (default is 5).
    - bar_color (str): Hex color code for the bars (default is Plotly's gray-blue).

    Returns:
    - fig (plotly.graph_objs._figure.Figure): Plotly figure object.
    - top_performers_df (pd.DataFrame): DataFrame of top performers.
    """
    if not skill:
        print("Please select a skill.")
        return None, pd.DataFrame()

    filtered_df = df[df['Skill'] == skill]

    if filtered_df.empty:
        print(f"No data found for skill: {skill}")
        return None, pd.DataFrame()

    # Ensure unique leaders by taking the max score per leader
    unique_leader_scores = (
        filtered_df.groupby('Leader', as_index=False)['Score']
        .max()
        .sort_values(by='Score', ascending=False)
        .head(top_n)
    )

    fig = px.bar(
        unique_leader_scores,
        x='Leader',
        y='Score',
        title=f"Top {top_n} Performers in '{skill}'",
        text='Score'
    )

    fig.update_traces(marker_color=bar_color, textposition='outside')
    fig.update_layout(
        yaxis_title='Score',
        xaxis_title='Leader',
        uniformtext_minsize=8,
        uniformtext_mode='hide',
        plot_bgcolor='white',
        title_x=0.5,
        margin=dict(t=60, b=40),
    )

    return fig, unique_leader_scores





import pandas as pd
import numpy as np
import plotly.graph_objects as go

def build_histogram_with_leaders(df):
    '''
    Expects a DataFrame with columns: 'LIS' and 'Leader'
    '''
    lis_data = df['LIS']
    leaders = df['Leader']
    
    # Stats
    mean_lis = np.mean(lis_data)
    std_lis = np.std(lis_data)
    std_low = mean_lis - 1.5 * std_lis
    std_high = mean_lis + 1.5 * std_lis

    # Bin the LIS data manually into 20 bins
    bin_counts, bin_edges = np.histogram(lis_data, bins=20)
    bin_labels = [f"{int(left)}–{int(right)}" for left, right in zip(bin_edges[:-1], bin_edges[1:])]

    # Assign each row to a bin
    df['LIS_bin'] = pd.cut(df['LIS'], bins=bin_edges, labels=bin_labels, include_lowest=True, ordered=False)

    # Group by bin: count and leader names
    grouped = df.groupby('LIS_bin').agg({
        'Leader': lambda x: ', '.join(sorted(set(x))),
        'LIS': 'count'
    }).reset_index().rename(columns={'LIS': 'Count'})

    # Build bar plot with hover info
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=grouped['LIS_bin'],
        y=grouped['Count'],
        customdata=grouped['Leader'],
        marker_color='darkblue',
        hovertemplate='<b>LIS Range:</b> %{x}<br>' +
                      '<b>Count:</b> %{y}<br>' +
                      '<b>Leaders:</b> %{customdata}<extra></extra>'
    ))


    fig.update_layout(
        title="Leadership Index Score (LIS) Distribution",
        xaxis_title="LIS Score Range",
        yaxis_title="Number of People",
        template="plotly_white",
        width=800,
        height=450,
        margin=dict(t=50, b=50, l=50, r=50)
    )

    # Summary stats
    summary_stats = {
        "mean": mean_lis,
        "std_dev": std_lis,
        "1.5_std_below": std_low,
        "1.5_std_above": std_high,
        "min_value": np.min(lis_data),
        "max_value": np.max(lis_data),
        "count": len(lis_data),
        "below_1.5_std": np.sum(lis_data < std_low),
        "above_1.5_std": np.sum(lis_data > std_high)
    }

    return fig
