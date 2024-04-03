import streamlit as st
from data_use.document_loaders.pdf import S2ORCPDFLoader
import json
from pathlib import Path
import tiktoken
import tempfile
import time
# from langchain_community.document_loaders import PyMuPDFLoader

from langchain.document_loaders import PyMuPDFLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import (
    NLTKTextSplitter,
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)

from data_use.indexes import vectorstore as dvectorstore
from data_use.document_loaders import pdf
from data_use import text_splitter as ts
from data_use.ranking import sentence_prob_for_texts



def extract_data_use(text: str):
    with st.spinner('Extracting data use...'):
        time.sleep(2)
        st.write("Data use extracted from the passage: ")


passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval, Input: ",
    query_instruction="Represent the question for retrieving relevant passage, Input: ",
)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)


st.title("Data Use Demo App")

st.write("This is a simple app to preview initial progress in the data citation work.")

# query = st.text_input("Enter a query", "Was data or dataset used? Was data or dataset collected and analyzed?")
query = st.text_input("Was data or dataset such as survey, satellite imagery, indicators, etc., mentioned in the passage?")

pdf_file = st.file_uploader("Upload a PDF", type="pdf")

# Create a text splitter
text_splitter = NLTKTextSplitter(chunk_size=512, chunk_overlap=0)

if pdf_file is not None:
    # Create a temporary file and write the content to it
    temp_file = tempfile.NamedTemporaryFile(delete=True, suffix=".pdf")
    temp_file.write(pdf_file.getvalue())
    print(temp_file.name)

    # Load the PDF content
    # In the meantime, we can use the PyMuPDF library to load the PDF content.
    # content = S2ORCPDFLoader(pdf_file).load()
    st.markdown("## Results")

    with st.spinner('Loading and processing the document...'):
        content = pdf_file.read()
        documents = PyMuPDFLoader(str(temp_file.name)).load_and_split(text_splitter=text_splitter)
        max_docs = len(documents)
        time.sleep(2)

    with st.spinner('Currently creating vectors...'):
        # Create a vectorstore index
        index = dvectorstore.VectorstoreIndexCreator(
            embedding=passage_embeddings,
        ).from_documents(documents)


    with st.spinner('Shortlisting passages...'):
        shortlist = index.vectorstore.similarity_search(query, k=min(20, max_docs))
        shortlist = [rs.page_content for rs in shortlist]
        time.sleep(2)

    with st.spinner('Ranking passages...'):
        ranked = sorted(zip(sentence_prob_for_texts(shortlist), shortlist), reverse=True)[:10]
        time.sleep(2)

    for i, rs in enumerate(ranked, 1):
        # Collapsible section
        with st.expander(f"Passage {i} - Score ({rs[0]:.4f})", expanded=False):
            st.write(f"{rs[1]}")

            # st.button("Extract Data Use", key=f"extract_{i}", on_click=extract_data_use, args=(rs[1],))

    # # Apply reranking

    # # Get the path to the temporary file
    # temp_file_path = Path(temp_file.name)

    # # Load the PDF and split it into sentences
    # loader = pdf.S2ORCPDFLoader(temp_file_path)
    # docs = loader.load()
    # documents = du.aggregate_documents(docs)

    # # Create a vectorstore index
    # index = dvectorstore.VectorstoreIndexCreator(
    #     embedding=passage_embeddings,
    # ).from_documents(documents)

    # shortlist = index.vectorstore.similarity_search(query)
    # shortlist = [rs.page_content for rs in shortlist]

    # # Apply reranking
