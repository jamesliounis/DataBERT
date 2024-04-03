# Automated BERT-based Text Classifier

This documentation covers two primary classes that form an end-to-end automated pipeline using the BERT (Bidirectional Encoder Representations from Transformers) model for text classification: `TrainBERTClassifier` and `BERTClassifier`.

## TrainBERTClassifier

The `TrainBERTClassifier` class is designed for fine-tuning the BERT model on a specific text classification task. It includes methods for data preprocessing, training, evaluation, and pushing the fine-tuned model to the Hugging Face Hub.

### Key Methods

- `__init__`: Initializes the model with necessary parameters such as model name, number of labels, etc.
- `preprocess`: Processes a list of texts into BERT's input format, converting texts into sequences of IDs and attention masks.
- `train`: Trains the BERT model on the provided dataset, evaluating it on a validation set, and computes accuracy, precision, recall, and specificity metrics.
- `push_to_huggingface_hub`: Pushes the fine-tuned model to the Hugging Face Hub for easy sharing and deployment.

## Usage

### Training DataBERT

1. **Initialize the Classifier**

You can initialize the `TrainBERTClassifier` with desired training parameters. Here's an example:

```python
from bert_classifier import TrainBERTClassifier

classifier = TrainBERTClassifier(
    model_name="bert-base-uncased",
    num_labels=2,
    max_length=32,
    batch_size=16,
    learning_rate=5e-7,
    epsilon=1e-8,
    epochs=2000,
    patience=3,
    tolerance=1e-4,
    optimizer_type='AdamW'
)
```

**Parameters**:

- `model_name` (str): The name or path of the pre-trained BERT model. Default is "bert-base-uncased". You can change this to any BERT model variant like "bert-large-uncased" depending on your specific needs or experimentations for better performance.

- `num_labels` (int): The number of labels for the classification task. Default is 2, which corresponds to binary classification. Modify this based on the number of categories in your specific classification task.

- `max_length` (int): The maximum length of the input sequences. Default is 32. This should be adjusted based on the average or maximum sentence length in your dataset to optimize both performance and computational efficiency.

- `batch_size` (int): The size of the batches for training the model. Default is 16. A larger batch size requires more memory but can lead to faster training. Adjust this based on your hardware capabilities and dataset size.

- `learning_rate` (float): The learning rate used by the optimizer. Default is 5e-7. This is a critical parameter that might need tuning based on your dataset characteristics and model performance.

- `epsilon` (float): The epsilon parameter for the optimizer, preventing any division by zero in the implementation. Default is 1e-8. This is a standard value for many optimizers and typically does not need adjustment.

- `epochs` (int): The number of training epochs, i.e., how many times the model should iterate over the entire dataset. Default is 2000. This parameter should be set based on the convergence behavior of your model: too few epochs might underfit, while too many can lead to overfitting.

- `patience` (int): The number of epochs to wait for improvement on the validation set before early stopping. Default is 3. Adjust this to allow more or fewer epochs for potential improvement, which can be particularly useful for larger datasets or more complex models.

- `tolerance` (float): The minimum improvement in validation loss required to reset the patience counter. Default is 1e-4. This defines what constitutes an "improvement," preventing early stopping from halting training on minor fluctuations.

- `optimizer_type` (str): The type of optimizer to use. Default is 'AdamW'. Other options like 'SGD', 'RMSprop', 'Adam', 'Adagrad', and 'Adadelta' allow you to experiment with different optimization strategies to find the best one for your data and model.

**Usage Notes**:

The `model_name`, `num_labels`, and `optimizer_type` parameters allow you to adapt the classifier to various datasets and classification problems.

The `max_length`, `batch_size`, `learning_rate`, `epsilon`, `epochs`, `patience`, and `tolerance` parameters offer fine-tuning capabilities to optimize model training and performance.

It is essential to experiment with different parameter settings to identify the most effective configuration for your specific task and dataset.

2. **Prepare your data**

Organize your text data and labels for training and validation. For example:

```python
# List of sentences
train_texts = ['This is the first text', 'Here is another one']
train_labels = [0, 1]

# List of sentences
val_texts = ['This text is for validation', 'Another validation text']
val_labels = [0, 1]
```



3. **Train the model**

Use the `train` method to fine-tune the BERT model on your data:

```python
classifier.train(train_texts, train_labels, val_texts, val_labels)
```

4. **Evaluate the Model**
:
After training, the model's performance metrics for the validation set will be printed automatically, including accuracy, precision, recall, specificity, and ROC-AUC.

5. **Save the Model**:

Save your trained model and tokenizer for later use:

```python
classifier.save_model_and_tokenizer('path/to/save/directory')
```



## BERTClassifier

The BERTClassifier class is intended for loading a fine-tuned BERT model and using it to classify new text instances. It can load models directly from the Hugging Face Hub.

### Key Methods
- `__init__`: Initializes the classifier with the path to a fine-tuned model and tokenizer.
- `preprocess`: Converts a given text into BERT's format, outputting token IDs and attention masks.
- `predict`: Generates a prediction for a given text, outputting whether it contains a dataset mention or not.

### Example Usage

One way you may use the BERTClassifier is the way we used it to classify PWRP sentences. Given that it encapsulates the entire inference process from pre-processsing to importing model weights from HuggingFace, you can construct a very simple function that can perform inference for you as follows (make sure that you are running the script from the same directory as classifier!):

```python
from DataBERT_pipeline import BERTClassifier

def dataset_classifier(input_sentence):
  bert_classifier = BertClassifier()
  prediction = bert_classifier.predict(input_sentence)
  return prediction
```


# Document Processing Class

The `DocumentProcessor` class is a versatile tool designed to facilitate the extraction, processing, and classification of text data from PDF documents, particularly in the context of analyzing World Bank Policy Research Working Papers (PWRP). It is related to the last part of the modelling process that we have outlined in this [notebook](https://github.com/avsolatorio/data-use/blob/build_training_data/notebooks/modelling/FineTuneBERT.ipynb). This class serves as a crucial component in the larger pipeline that aims to identify dataset mentions within these documents.

## Features and Functionalities

- **Text Extraction**: Supports extracting text from PDF documents either page-wise ('chunk' mode) or sentence-wise ('sent' mode).
- **Document Downloading**: Capable of downloading PDF documents from specified URLs and extracting their text content directly.
- **Dataset Creation**: Facilitates the aggregation of extracted text into structured datasets, which can then be uploaded to the Hugging Face Hub for further use or sharing.
- **Text Classification**: Integrates with our fine-tuned BERT model to classify extracted texts (or sentences) regarding their relevance to dataset mentions.

## Using it in the Project

The `DocumentProcessor` class is instrumental in automating the extraction and initial processing of text data from PWRP PDFs. It significantly reduces manual effort and provides a standardized method for preparing the text data for subsequent analysis steps. The class's ability to interface with the Hugging Face Hub ensures that datasets can be shared and accessed conveniently. The integration with the `BertClassifier` from `DataBERT_pipeline.py` allows for an automated, AI-driven approach to classify text based on the presence of dataset mentions.

## Example Usages

### Initializing the DocumentProcessor

```python
from DocumentProcessingPipeline import DocumentProcessor

# Initialize the processor for an organization with a specific cache directory.
processor = DocumentProcessor("my_organization", "./json_cache", tokenizer_model="bert-base-uncased")
```

### Extract text by sentences from a local PDF file.
```python
sentences = processor.extract_text("path/to/local/file.pdf", "sent")
```

### Download a PDF from a URL and extract text in chunks.
```python
chunks = processor.get_doc_from_url("https://example.com/document.pdf", "chunk") # or mode = "sent"
```

### Creating and Pushing a Dataset to Hugging Face Hub
```python
# Where 'documents' is a list of extracted text contents.
processor.create_dataset(documents, mode="sent")
```
### Given a list of documents or sentences, classify each using the fine-tuned BERT model.
```python
results = processor.classify(documents)
```



