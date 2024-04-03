# Introduction

**Bidirectional Encoder Representations from Transformers (BERT)** is a groundbreaking method in natural language processing (NLP) that has revolutionized how machines understand human language. Developed by researchers at Google, BERT is based on the Transformer architecture and utilizes deep learning techniques to process words in relation to all the other words in a sentence, contrary to traditional methods that look at words sequentially. This bidirectional approach allows BERT to capture the context of a word more effectively, leading to significant improvements in a variety of NLP tasks, such as question answering, named entity recognition, and sentiment analysis.

The objective of our project is to leverage the powerful capabilities of BERT for a specific, critical NLP task: identifying mentions of datasets within textual content. In the realm of research, especially within fields that heavily depend on data, such as economics, life sciences, and social sciences, recognizing references to datasets is crucial. These mentions can provide insights into the data sources researchers are utilizing, foster data sharing, and enhance reproducibility in scientific research. The goal is to eventually construct a centralized database of dataset mentions. 

By fine-tuning BERT on our specially consolidated training data (for which details can be found [here](https://github.com/avsolatorio/data-use/blob/build_training_data/notebooks/ranking/README.md)), our goal is to develop a robust classifier capable of accurately distinguishing sentences that contain dataset mentions from those that do not. The ability to automatically detect dataset references can significantly benefit researchers, librarians, and data curators by streamlining the process of linking research outcomes with the underlying data, thereby advancing the frontiers of open science and data-driven research.

# Fine-Tuning BERT model on our consolidated training data

In this notebook, we fine-tune BERT on our consolidated training data in order to obtain a reliable classifier that can distinguish between sentences that mention a dataset and sentences that do not. 

We tokenize the data using the BERT tokenizer `bert-base-uncased`, and encode with specific parameters:

```python
return tokenizer.encode_plus(
        input_text,  # The text to be tokenized and encoded
        add_special_tokens=True,  # Add special tokens (e.g., [CLS], [SEP])
        max_length=32,  # Maximum length of the tokenized sequence
        padding="max_length",  # Pad sequences to the maximum length
        truncation=True,  # Truncate sequences exceeding the maximum length
        return_attention_mask=True,  # Generate attention masks
        return_tensors="pt",  # Return PyTorch tensors
    )
```

Splitting the data between `train` and `val`, we are careful to have a balanced amount of positive and negative samples by stratifying around the labels. 

## Evaluating our model 

To assess the performance of our fine-tuned BERT model, we employ several key metrics that are standard in the field of classification tasks. These metrics provide us with a comprehensive understanding of our model's capabilities. 

- **Accuracy**: This metric assesses the overall correctness of the model predictions, measuring the ratio of correctly predicted observations to the total observations.

  $$\text{Accuracy} = \frac{\text{Number of correct predictions}}{\text{Total number of predictions}} = \frac{TP + TN}{TP + TN + FP + FN}$$

- **Precision**: Also known as the positive predictive value, precision measures the ratio of correctly predicted positive observations to the total predicted positives. It is crucial for minimizing false positives.

  $$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP) + False Positives (FP)}}$$

- **Recall**: Also known as sensitivity, recall measures the ratio of correctly predicted positive observations to all observations in the actual class. It is vital for minimizing false negatives.

  $$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP) + False Negatives (FN)}}$$

- **Specificity**: This metric measures the ratio of correctly predicted negative observations to all observations in the actual negative class. It helps in assessing the model's ability to identify true negatives.

  $$\text{Specificity} = \frac{\text{True Negatives (TN)}}{\text{True Negatives (TN) + False Positives (FP)}}$$

### Implications and Limitations of the Metrics

- **Accuracy** provides a general sense of the model's performance but can be misleading in cases of imbalanced datasets. High accuracy might not necessarily indicate a well-performing model if one class significantly outweighs the other.

- **Precision** is particularly important when the cost of false positives is high. In the context of our project, a high precision means that when our model identifies a sentence as having a dataset mention, it is likely correct. However, focusing solely on precision might lead to overlooking relevant dataset mentions (increasing false negatives).

- **Recall** becomes critical when it is crucial to capture as many positives as possible. For dataset identification, a high recall means we are successfully capturing most dataset mentions. Nonetheless, optimizing only for recall can result in many false positives, lowering the overall quality of our predictions.

- **Specificity** is key when it is important to validate the true negatives. While not always the primary focus for every classification task, in the context of identifying dataset mentions, high specificity ensures that sentences without dataset mentions are rarely misclassified.


## Training

We train the model for 2 epochs, using a batch size of 16 as recommended by the [original BERT paper](https://arxiv.org/pdf/181004805.pdf). The training was carried out with scripts from the PyTorch libray, using a T4 GPU available on Google Colab. 

We obtain the following results:

```python
Epoch:  50%|█████     | 1/2 [01:36<01:36, 96.14s/it]
	 - Train loss: 0.0586
	 - Validation Accuracy: 0.9869
	 - Validation Precision: 0.9995
	 - Validation Recall: 0.9761
	 - Validation Specificity: 0.9990

Epoch: 100%|██████████| 2/2 [03:10<00:00, 95.11s/it]
	 - Train loss: 0.0280
	 - Validation Accuracy: 0.9846
	 - Validation Precision: 0.9885
	 - Validation Recall: 0.9844
	 - Validation Specificity: 0.9855
```

### Results

At the end of the first epoch, our model shows excellent performance across all metrics. The high precision (99.95%) indicates that almost all positive predictions were correct, while the high recall (97.61%) suggests the model was also able to identify the majority of actual positives. The specificity is similarly high, indicating few false positives. However, by the second epoch, even though the training loss has decreased, indicating better performance on the training set, we observe a slight decrease in validation accuracy, precision, and specificity. This could be an early sign of overfitting, where the model performs better on the training data but slightly worse on unseen data.

In conclusion, the slight decrease in validation metrics from the first to the second epoch, despite lower training loss, suggests that the model might be starting to overfit the training data. The high precision and recall values indicate that the model is highly effective but might be beginning to memorize the training data, especially given the drop in precision and specificity.


## Performing Inference

The model weights are saved in a [Shared Drive](https://drive.google.com/drive/u/0/folders/1DwY2dQYABso7JVzTRcUJHThiFyb-Bj7P), but are not also available on [HuggingFace](https://huggingface.co/jamesliounis/DataBERT). This last step was made in order to streamline the process of reading in the model weights, which can now be done in a few lines:

```python
# Load model directly
from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case = True)
model = BertForSequenceClassification.from_pretrained("jamesliounis/DataBERT")
```

We then set up the model for classification purposes, and test it out. 

This entire pipeline of training the model, saving the weights to HuggingFace and then using it for inference has been made into a robust script which encapsulates the entire process and is available [here](https://github.com/avsolatorio/data-use/blob/build_training_data/scripts/DataBERT/DataBERT_pipeline.py), along with relevant documentation. 



### Using a BERT model to Analyze World Bank Policy Research Working Papers (PWRP)

In our project, we aim to analyze the World Bank's Policy Research Working Papers (PWRP) to identify mentions of datasets within these documents. Our process involves several key steps:

1. **Accessing the PWRP Database**: We use a pre-defined [API](https://drive.google.com/file/d/164IErzEVdl4cKenBXFenlWJfLKpusJvu/view?usp=share_link) to access metadata for 50 different PWRP documents. This API provides us with essential information such as the authors' names, paper titles, and URLs for accessing the full texts.

2. **Text Extraction**: To extract text from the PWRP documents, we employ a document processing pipeline, which can be reviewed [here](https://github.com/avsolatorio/data-use/tree/build_training_data/scripts). This script is responsible for downloading PDF documents from the provided URLs and extracting their text content. The complete extraction and classification process is encapsulated in [this pipeline](https://github.com/avsolatorio/data-use/blob/build_training_data/scripts/DataBERT/DocumentProcessingPipeline.py), with accompanying documentation for reference.

3. **Sentence-wise Classification**: Once we have extracted the full text from the PDFs, we split this content into individual sentences, resulting in approximately 5,000 sentence records from 50 papers. Subsequently, we apply our fine-tuned BERT classifier to each sentence to determine whether it mentions a dataset. This classification process yields an annotated dataset, `pwrd_annotated_50.xlsx`, which is available [here](https://docs.google.com/spreadsheets/d/1--LfSC3AljJL5eLMBbbPOZHb5ZMTawkk/edit?usp=share_link&ouid=104530792956854202310&rtpof=true&sd=true).

### Catching Specific Errors

We manually verify roughly 200 entries and look at specific examples of incorrectly classified sentences:

**False-Positives**:

- "The papers carry the \nnames of the authors and should be cited accordingly."
- "\nFrom a methodological point of view, this paper aims to cover the entire universe of knowledge products \ngenerated by the World Bank while at the same time encompassing the various correlates of suc cess \nexplored by previous studies."
- "\nThe analysis in the paper i s conducted  for knowledge- related documents completed between July 1st, \n2015, and June 30, 2019."

The model struggles with strong research contexts. Verbs like "papers", "authors", "analysis" seem to influence it quite heavily in detecting datasets. 

**True-Positives**:

- "\nUsing citation data from the Google Scholar platform , this study revealed  the heterogen eous quality of \nthe research  products  selected ."
- "\nIn another study, a survey of senior World Bank operational staff was used to assess their awareness \nabout research  products , such as those produced by DEC  (Ravallion  and Wagstaff 2011)."

Very strong nuance here, no dataset really mentioned but definitely data and a data source. TP or FP?


**False-Negatives** (or not??):

- "\nA comprehensi ve database along those lines was assemble d for this paper  by matching information \navailable in three independent digital platforms : \n• The Image Bank  (IB) is a document filing system where all entries are classified based on their \nsensitivity, which determines who inside or outside the World Bank can retrieve them."
- \n• The Business Intelligence  Warehouse  (WH) is a digital platform  organized around budget codes that is \naccessible only internally.
- "', '13 \n • Finally, the Open Knowledge Repository  (OKR) makes all World Bank publications av ailable to the \npublic  at large ."

Datasets are clearly mentioned here, but not actively used in the research. TBD how we score these. 


### Human in the Loop Process

After utilizing the BERT model to annotate the PWRP sentences, we engage in a 'human in the loop' process:

1. **Data Review and Augmentation**: We manually review the annotated sentences to validate the model's accuracy. By identifying and confirming correctly classified instances, we can select high-quality examples to augment our existing training dataset. This augmentation process enhances the diversity and coverage of our training data, making the model more robust and reliable for future analyses.

The next iteration of our model will have to focus on additional nuances in the training data which we will implement based on the data points that we manually identified. For example:
- Generate training data which has a strong research context but no dataset to teach the model the distinction between contexts and actual dataset mentions.
- Further work on the nuances between dataset mentions but no use.

2. **Model Re-training**: With the expanded training dataset, we re-train our BERT model to incorporate the new examples. This iterative process of augmentation and re-training helps improve the model's performance and adaptability, especially concerning the identification of dataset mentions within the specific context of World Bank research papers.

By continuously refining our model through these steps, we enhance its precision and utility in the field of automated text analysis, particularly in identifying dataset mentions within academic and research texts.



