import datasets
from transformers import BertTokenizer
import tempfile
import requests
import PyPDF2
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
from DataBERT_pipeline import BertClassifier

class DocumentProcessor:
    def __init__(
        self, organization, json_cache_dir, tokenizer_model="bert-base-uncased"
    ):
        """
        Initializes the DocumentProcessor with a specific tokenizer model and organization details.

        Args:
        organization (str): Name of the organization for dataset pushing.
        json_cache_dir (str): Directory to cache processed documents.
        tokenizer_model (str): Pretrained BERT tokenizer model.
        """
        self.organization = organization  # Organization name for dataset publishing.
        self.json_cache_dir = json_cache_dir  # Directory to store cached documents.
        # Initialize a BERT tokenizer for text processing.
        self.tokenizer = BertTokenizer.from_pretrained(
            tokenizer_model, do_lower_case=True
        )

    def extract_text(self, pdf_path, mode):
        """
        Extracts text from a PDF file, either in chunks (full page) or by sentences.

        Args:
        pdf_path (str): Path to the PDF file to be processed.
        mode (str): The mode of text extraction - 'chunk' for page-wise, 'sent' for sentence-wise.

        Returns:
        list: A list of extracted text chunks or sentences.
        """
        content = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            # Iterate through PDF pages.
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()  # Extract text from the current page.
                if mode == "chunk":
                    content.append(text)  # Append the whole text as one chunk.
                elif mode == "sent":
                    # Tokenize and append each sentence separately.
                    sentences = sent_tokenize(text)
                    content.extend(sentences)
        return content

    def get_doc_from_url(self, pdf_url, mode="chunk"):
        """
        Downloads a PDF document from a given URL and extracts text based on the specified mode.

        Args:
        pdf_url (str): URL of the PDF to download and process.
        mode (str): The mode of text extraction - 'chunk' or 'sent'.

        Returns:
        list: A list of extracted text chunks or sentences.

        Raises:
        Exception: If there is an issue with downloading the PDF.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
            response = requests.get(pdf_url, stream=True)
            if response.status_code == 200:
                # Write the content of the PDF to a temporary file.
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_pdf.write(chunk)
                temp_pdf.seek(0)  # Reset file pointer.
                # Extract text from the downloaded PDF.
                return self.extract_text(temp_pdf.name, mode)
            else:
                raise Exception(
                    f"Error downloading the file: Status code {response.status_code}"
                )

    def create_dataset(self, documents, mode="chunk"):
        """
        Creates and pushes a dataset from a list of document contents to Hugging Face Hub.

        Args:
        documents (list): List of document contents to be included in the dataset.
        mode (str): Mode of dataset creation, can be 'chunk' or 'sent'.

        Raises:
        AssertionError: If the mode is not 'chunk' or 'sent'.
        """
        assert mode in ["chunk", "sent"]

        dataset = {}
        # Construct the dataset from document contents.
        for i, content in enumerate(tqdm(documents)):
            dataset[i] = {"content": content}

        # Create and push the dataset to Hugging Face Hub.
        data = datasets.Dataset.from_dict(dataset)
        data.push_to_hub(
            f"{self.organization}/wb-prwp-{mode}",
            private=True,
            commit_message=f"Add {mode} new dataset.",
        )

    def classify(self, documents):
        """
        Classifies a list of documents using a fine-tuned BERT model.
        Pipeline is in separate file: DataBERT_pipeline.py

        Args:
        model_path (str): Path to the fine-tuned BERT model directory.
        documents (list of str): Documents to classify. 
        Note: In our case, we apply the function directly to individual sentences. 
        The iterative approach just further enforces modularity that we may need in future versions. 

        Returns:
        list: A list containing classification results for each document.
        """
        bert_classifier = (
            BertClassifier()
        )  # Initialize the BertClassifier with the fine-tuned model.
        classifications = []

        # Iterate over all documents and predict their classes.
        for doc in tqdm(documents, desc="Classifying documents"):
            classification = bert_classifier.predict(doc)
            classifications.append(classification)

        return classifications
