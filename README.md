# Machine Learning II - Group Assignment
This repository serves as the main deliverable of the Group Assignment for Machine Learning II of IE University's Master of Business Analytics and Big Data.

## Group members
_Group 11: Laura Silva | Afonso Vaz dos Santos | Ignacio Amigó | Omar Harradi | Lucas Ihnen_

The streamlit application is deployed in the following link:

https://mlfinal-7bnkw3eebcjkiszxpvpmnk.streamlit.app/

## Files included in this repository

### Streamlit App
The application is deployed from the Streamlit code on the file `ivy.py`.

This file has a supporting python file that contains all the auxiliary functions for building charts, calling the LLM and interacting with different components on `utils.py`.

Also, the `requirements.txt` file needed for proper deployment in Streamlit Cloud. 

Also, there is a folder containing the chromadb for our RAG implementation, as well as a notebook called `rag.ipynb` that created it.

### Datasets
The proprietary data for this application is anonymized and untraceable to the assessed people. 

It contains the following datasets:
- LDP_summary_anonymized.csv: CSV file containing all the assesment results
- skills_mapping_renamed.json: Explanatory JSON of the assesment and skills involved
- result.json: The assessment result, but in JSON format

### Presentation
We also included in the repository the final presentation `G11 - Machine Learning - Final Project.pdf`



