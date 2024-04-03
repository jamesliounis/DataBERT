#!/usr/bin/python3

# ==============BUILDING AN INTERACTIVE APP===============

import streamlit as st
# from DataBERT_pipeline import BERTClassifier
# from DocumentProcessingPipeline import DocumentProcessor
import nltk
from textraer.textraer import DocumentProcessor
nltk.download('punkt')

# Setting a page configuration with a title and icon.
st.set_page_config(page_title='DataBERT', page_icon=':bar_chart:')

# Define the function to set the background color and additional styles.
def set_background_color_and_styles():
    # Use colors from the World Bank's branding for the background.
    main_bg_color = "#0049A5"  # This is a shade of blue.
    main_text_color = "#FFFFFF"  # White text color for better contrast.
    
    # Custom CSS to set the background color and center the content.
    st.markdown(f"""
        <style>
        .stApp {{
            background-color: {main_bg_color};
            color: {main_text_color};
        }}
        /* Center the logo image */
        .stImage {{
            margin: auto;
        }}
        /* Additional styles if needed */
        </style>
        """, unsafe_allow_html=True)

# Call the function to set the background color and apply styles.
set_background_color_and_styles()

# Display the logo image at the top of the page and center it.
st.image('logo.png', width=250, use_column_width=True)  # Adjust width as needed

# Initialize your BERTClassifier and DocumentProcessor
document_processor = DocumentProcessor(organization='org', json_cache_dir='cache_dir')


def classify_sentences_from_pdf(pdf_url):
    """
    Classifies sentences extracted from a PDF document identified by its URL and filters out
    sentences classified with a positive label, indicating they contain a dataset mention.

    This function performs three main steps:
    1. It extracts text from the PDF located at the given URL, segmenting the content into sentences.
    2. It classifies each extracted sentence using a pre-defined classification model within `DocumentProcessor`.
    3. It filters and ranks the positively labeled sentences by their confidence scores in descending order.

    Args:
        pdf_url (str): The URL of the PDF document to process.

    Returns:
        list of tuples: Each tuple contains a positively labeled sentence and its associated confidence score,
        sorted by confidence in descending order.
    """
    st.write('Extracting text from the PDF document...')
    sentences = document_processor.get_doc_from_url(pdf_url, mode='sent')
    
    st.write('Classifying sentences...')
    classification_results = document_processor.classify(sentences)
    
    st.write('Filtering and ranking positive labels by confidence...')
    positive_classified_sentences = [
        (sentences[i], result[0]['score'])
        for i, result in enumerate(classification_results)
        if result and result[0].get('label') == 'LABEL_1'
    ]
    positive_classified_sentences.sort(key=lambda x: x[1], reverse=True)
    return positive_classified_sentences




# Streamlit UI
#st.title('Where Data Meets Discovery')
st.markdown("<h1 style='text-align: center;'>Where Data Meets Discovery</h1>", unsafe_allow_html=True)


pdf_url = st.text_input('Enter the URL of your PDF document:')

if pdf_url:
    with st.spinner('Processing...'):
        classified_sentences = classify_sentences_from_pdf(pdf_url)
        if classified_sentences:
            st.write('Sentences containing dataset mentions ranked by confidence:')
            for sentence, confidence in classified_sentences:
                st.write(f"{sentence} - Confidence: {confidence:.2f}")
        else:
            st.write('No dataset mentions detected in the document.')
