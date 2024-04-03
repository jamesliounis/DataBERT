import tempfile
import requests
import PyPDF2
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
import datasets
from transformers import pipeline


class DocumentProcessor:
    def __init__(self, organization, json_cache_dir, tokenizer_model="bert-base-uncased", model_path="jamesliounis/DataBERT"):
        """
        Initializes the DocumentProcessor with specific configurations for text processing and dataset management.

        Args:
            organization (str): The organization's name for pushing datasets.
            json_cache_dir (str): Directory to cache processed document texts.
            tokenizer_model (str): The tokenizer model identifier for text processing.
            model_path (str): The path to the fine-tuned model for classification.
        """
        self.organization = organization
        self.json_cache_dir = json_cache_dir
        self.classifier_pipeline = pipeline("text-classification", model=model_path, tokenizer=tokenizer_model)

    def extract_text(self, pdf_path, mode):
        """
        Extracts text from a specified PDF file, either in full-page chunks or sentence by sentence.

        Args:
            pdf_path (str): The path to the PDF file from which text will be extracted.
            mode (str): The extraction mode ('chunk' for full pages or 'sent' for sentences).

        Returns:
            List[str]: A list of extracted text segments (chunks or sentences).
        """
        content = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                text = page.extract_text()
                if mode == "chunk":
                    content.append(text)
                elif mode == "sent":
                    sentences = sent_tokenize(text)
                    content.extend(sentences)
        return content

    def get_doc_from_url(self, pdf_url, mode="chunk"):
        """
        Downloads and extracts text from a PDF document at a given URL.

        Args:
            pdf_url (str): The URL of the PDF to download and process.
            mode (str): The extraction mode ('chunk' or 'sent').

        Returns:
            List[str]: A list of text segments extracted from the document.

        Raises:
            Exception: If there is an error in downloading the PDF.
        """
        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
            response = requests.get(pdf_url, stream=True)
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_pdf.write(chunk)
                temp_pdf.seek(0)
                return self.extract_text(temp_pdf.name, mode)
            else:
                raise Exception(f"Failed to download PDF, status code: {response.status_code}")

    def create_dataset(self, documents, mode="chunk"):
        """
        Creates a dataset from a list of documents and pushes it to the Hugging Face Hub.

        Args:
            documents (List[str]): The documents to include in the dataset.
            mode (str): The mode of dataset creation ('chunk' or 'sent').

        Raises:
            AssertionError: If the provided mode is invalid.
        """
        assert mode in ["chunk", "sent"], "Mode must be either 'chunk' or 'sent'."

        dataset = datasets.Dataset.from_dict({"content": documents})
        dataset.push_to_hub(f"{self.organization}/wb-prwp-{mode}", private=True, commit_message=f"Add {mode} dataset.")


    def classify(self, documents):
        """
        Classifies a list of documents using the initialized text classification pipeline.

        This method processes each document through the pipeline to obtain classification results, leveraging
        the BERT model specified during the class initialization.

        Args:
            documents (list of str): The documents to be classified.

        Returns:
            list: A list containing classification results for each document, including labels and scores.
        """
        classifications = []

        # Iterate over all documents and apply the classification pipeline.
        for doc in tqdm(documents, desc="Classifying documents"):
            # The pipeline returns a list of result objects; we take the first one as each document is processed individually.
            result = self.classifier_pipeline(doc)
            classifications.append(result)

        return classifications




