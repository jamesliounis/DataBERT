# Import the necessary library
import requests
from DataBERT_pipeline import DocumentProcessor

# Assuming the DocumentProcessor class is already defined as per your script.

# Initialize the DocumentProcessor with hypothetical arguments.
# Replace 'your_organization' with your actual Hugging Face organization name
# and specify a valid directory for 'json_cache_dir'.
document_processor = DocumentProcessor(
    organization='your_organization',
    json_cache_dir='./json_cache',
    tokenizer_model='bert-base-uncased'
)

# URL of the paper to process
pdf_url = 'https://arxiv.org/pdf/1706.03762.pdf'

# Use the DocumentProcessor to extract text and classify the content of the paper.
# This example extracts text by chunks (full pages).
try:
    # Download and extract text from the paper.
    document_text_chunks = document_processor.get_doc_from_url(pdf_url, mode='chunk')

    # Now, you could print out these chunks or proceed to classify them.
    # Here's how you might print the first chunk:
    print("Displaying the first extracted chunk of text:\n")
    print(document_text_chunks[0][:1000])  # Displaying the first 1000 characters for brevity.

    # If you want to proceed with classification (dummy example as classification needs a fine-tuned model):
    # Remember, 'classify' in this context might not provide meaningful results unless using a properly trained model.
    print("\nClassifying the first extracted chunk of text:\n")
    classification_results = document_processor.classify(document_text_chunks[:1])  # Classify the first chunk as an example.
    print(classification_results)

except Exception as e:
    print(f"An error occurred: {str(e)}")

