# Add SQLite fix at the very top
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import pandas as pd
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

# Set OpenAI API key from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# App title and description
st.title("LP Lead Scoring")
st.markdown("Score potential LPs based on fit with your fund")

# Initialize session state for results
if 'results_displayed' not in st.session_state:
    st.session_state['results_displayed'] = False

# VC Fund Information Collection
with st.form("fund_info"):
    st.subheader("Fund Information")
    fund_name = st.text_input("Fund Name")
    fund_size = st.number_input("Target Fund Size ($M)", min_value=0)
    stage = st.selectbox("Investment Stage", ["Pre-Seed", "Seed", "Series A", "Series B+", "Growth"])
    thesis = st.text_area("Investment Thesis")
    focus_areas = st.multiselect(
        "Focus Areas",
        ["Enterprise SaaS", "AI/ML", "Fintech", "Healthcare", "Consumer", "Climate"]
    )
    geographical_focus = st.multiselect(
        "Geographical Focus",
        ["North America", "Europe", "Asia", "Latin America", "Global"]
    )
    submit_button = st.form_submit_button("Save Fund Info")

# LP Database Upload
st.subheader("LP Database Upload")
uploaded_file = st.file_uploader("Upload LP database (CSV)", type="csv")

if uploaded_file is not None and not st.session_state['results_displayed']:
    try:
        # Read and process LP data
        df = pd.read_csv(uploaded_file)
        
        # Create RAG system
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Prepare LP documents
        documents = []
        for _, row in df.iterrows():
            content = f"""
            LP Name: {row['name']}
            Type: {row['type']}
            AUM: {row['aum']}
            Investment Focus: {row['investment_focus']}
            Previous Investments: {row['previous_investments']}
            Geographic Preference: {row['geographic_preference']}
            """
            documents.append(content)
        
        # Create embeddings and vector store with persistence
        embeddings = OpenAIEmbeddings()
        texts = text_splitter.create_documents(documents)
        
        # Create vectorstore with simplified configuration
        vectorstore = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
        
        # Create query from fund information
        query = f"""
        Find LPs that would be interested in:
        - A {fund_size}M {stage}-stage fund
        - Focusing on {', '.join(focus_areas)}
        - Investing in {', '.join(geographical_focus)}
        - With thesis: {thesis}
        """
        
        # Retrieve and rank LPs
        similar_docs = vectorstore.similarity_search(query, k=5)
        
        # Display results
        st.markdown("### Top LP Matches")
        for doc in similar_docs:
            st.text(doc.page_content)
            st.divider()
        
        # Set flag to prevent duplicate display
        st.session_state['results_displayed'] = True
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

# Add reset button
if st.button("Reset Results"):
    st.session_state['results_displayed'] = False
    st.rerun()
