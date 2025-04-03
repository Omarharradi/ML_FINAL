
# utils.py

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st  
import plotly.express as px
import os




def get_filtered_df(df, selected_dashboards, selected_positions):
    filtered = df.copy()
    if selected_dashboards:
        filtered = filtered[filtered["# Dashboard"].isin(selected_dashboards)]
    if selected_positions:
        filtered = filtered[filtered["Position"].isin(selected_positions)]
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
    full_labels = ['Meeting Minimum <br>Competency', 'Exceeding Expectations', 'Requiring Training']
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

def radar_chart_plotly(dashboard, df, skills_mapping):
    ''' 
    LIS Stands for Leadership Index Score which is a weighted score of critical skills necessary skills and beneficial skills to have a standardized metric to compare all individuals.
    EQ assesses Emotional Intelligence which is one of the most important skills that all leaders are assessed in.
    '''
    mapping = skills_mapping.get(dashboard, {})
    categories = ['Critical Skills', 'Necessary', 'Beneficial Skills']

    avg_scores = {}
    summary_stats = {"dashboard": dashboard, "skill_scores": {}}

    # Define unique colors for each category (up to 3 categories in your case)
    color_palette = px.colors.qualitative.Pastel
    category_colors = {
        cat: color_palette[i % len(color_palette)]
        for i, cat in enumerate(categories)
    }

    avg_scores = {}
    for cat in categories:
        skills = mapping.get(cat, [])
        avg_scores[cat] = [df.loc[df['# Dashboard'] == dashboard, skill].mean()
                           for skill in skills if skill in df.columns]

    fig = make_subplots(
        rows=1, cols=len(categories),
        specs=[[{'type': 'polar'} for _ in categories]],
        subplot_titles=[f"{cat}" for cat in categories],
    )

    # Set consistent radial axis settings
    radial_range = [0, 100]
    tickvals = [0, 20, 40, 60, 80, 100]
    ticktext = ["0", "20", "40", "60", "80", "100"]

    for i, cat in enumerate(categories):
        scores = avg_scores[cat]
        if not scores:
            continue
        skills = mapping[cat]
        scores = np.array(scores)
        scores_closed = np.concatenate((scores, [scores[0]]))
        formatted_skills = [
            skill.replace(" ", "<br>").replace("-", "<br>") if "&" not in skill
            else skill.replace(" &", "&").replace("-", "<br>").replace(" ", "<br>").replace("&", " &")
            for skill in skills
        ]
        skills_closed = formatted_skills + [formatted_skills[0]]

        # Dynamically set rotation angle based on number of edges
        n_skills = len(skills)
        rotation = 45 if n_skills == 4 else 0 if n_skills == 5 else 0

        fig.add_trace(
            go.Scatterpolar(
                r=scores_closed,
                theta=skills_closed,
                fill='toself',
                mode='markers+lines',
                name=cat,
                marker=dict(color=category_colors[cat])
            ),
            row=1, col=i+1
        )

        fig.update_polars(
            dict(
                radialaxis=dict(
                    visible=True,
                    range=radial_range,
                    tickvals=tickvals,
                    ticktext=ticktext,
                    showline=True
                ),
                angularaxis=dict(
                    rotation=rotation,
                    tickfont=dict(size=10)
                )
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

