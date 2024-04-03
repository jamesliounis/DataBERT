# DataBERT Application

The DataBERT application leverages [DataBERT](https://huggingface.co/jamesliounis/DataBERT), our fine-tuned BERT model to detect dataset mentions within textual data. This application is particularly useful for identifying potential dataset references in large documents or corpuses. The following guide outlines the components of the DataBERT application and provides instructions on how to use it.

## Components

The DataBERT application comprises several scripts, each serving a unique purpose within the system:

1. `DataBERT_pipeline.py`:

This script contains the core functionalities to fine-tune and deploy a BERT-based machine learning model for the task of identifying dataset mentions in text. It includes two main classes:

- `TrainBERTClassifier`: Handles the training and evaluation of the BERT model, providing functionalities to preprocess text data, fine-tune the model, evaluate its performance, and push the fine-tuned model to Hugging Face Hub.
- `BERTClassifier`: Utilizes the fine-tuned model to classify new text inputs, predicting whether they contain mentions of datasets and providing confidence scores for these predictions. This is the class leveraged by the application. 

2. `DocumentProcessor.py`

This module is responsible for processing documents to extract text, which is then analyzed by the BERT model. Key functionalities include:

- Extracting text from PDF files, supporting both page-wise and sentence-wise extraction.
- Downloading and processing PDF documents from URLs.
- Classifying extracted text segments using the fine-tuned BERT model to identify dataset mentions.

3. `app.py`

A Streamlit-based web application script that provides a user-friendly interface to interact with the DataBERT model. Users can input the URL of a PDF document, and the app will display sentences identified as containing dataset mentions, ranked by confidence.

4. `Dockerfile`

Defines a Docker container that encapsulates the DataBERT application, ensuring consistent deployment across different environments. The Dockerfile outlines the steps to set up the application, install dependencies, and launch the Streamlit app.

## Using the App

To use the DataBERT Streamlit application, you can pull the Docker image from Docker Hub and run it on your local machine or a server. Below are the steps to get started:

1. **Pull the Docker image**

Pull the `databert_app` image from Docker Hub:

```bash
docker pull jamesliounis/databert_app
```

2. **Run the Docker Container**

Launch the container, mapping the container's port 8501 to a local port:

```bash
docker run -p 8501:8501 jamesliounis/databert_app
```
This command runs the container in the foreground. If you prefer to run it in the background, you can add the `-d` flag.

3. **Access the Streamlit App**

Once the container is running, access the Streamlit app by navigating to [http://localhost:8501](http://localhost:8501) in your web browser. Here, you can interact with the DataBERT application:

- Enter the URL of a PDF document in the provided input field. For example, you could try: https://documents1.worldbank.org/curated/en/099503103052424337/pdf/IDU1a6ed49c91591914f4e198a21bde2d796969f.pdf
- Submit the URL, and the application will process the document, identifying and ranking sentences based on their likelihood of containing dataset mentions.
- The results will be displayed directly in the web interface, with sentences sorted by confidence scores.




