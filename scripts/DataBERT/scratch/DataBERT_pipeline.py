from transformers import BertForSequenceClassification, BertTokenizer
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import torch
import numpy as np
import tempfile
from huggingface_hub import HfFolder, Repository
import os
from sklearn.metrics import roc_auc_score
import copy
import tempfile
import requests
import PyPDF2
from nltk.tokenize import sent_tokenize
from tqdm.auto import tqdm
import datasets
from databert import BERTClassifier

# ================ FINE-TUNE BERT ======================


# FINE-TUNE BERT


class TrainBERTClassifier:
    def __init__(
        self,
        model_name="bert-base-uncased",
        num_labels=2,
        max_length=32,
        batch_size=16,
        learning_rate=5e-7,
        epsilon=1e-8,
        epochs=2000,
        patience=3,
        tolerance=1e-4,
        optimizer_type="AdamW",
    ):
        
        """
        Initializes the BERT Classifier for training and evaluation.

        Args:
        model_name (str): Name or path of the pre-trained model.
        num_labels (int): Number of labels for classification.
        max_length (int): Maximum length of the input sequence.
        batch_size (int): Batch size for training and evaluation.
        learning_rate (float): Learning rate for the optimizer.
        epsilon (float): Epsilon value for the optimizer.
        epochs (int): Number of training epochs.
        """

        self.optimizer_type = optimizer_type
        self.patience = patience
        self.tolerance = tolerance
        self.best_val_roc_auc = 0.0  # Initialize the best ROC-AUC score
        self.epochs_no_improve = (
            0  # Initialize the count for epochs without improvement
        )
        self.best_model_state = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.epochs = epochs

    def preprocess(self, texts):
        """
        Preprocesses a list of texts for BERT model.

        Args:
        texts (list of str): List of texts to be preprocessed.

        Returns:
        TensorDataset: A dataset of input IDs and attention masks.
        """
        input_ids = []
        attention_masks = []

        for text in texts:
            encoding = self.tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_attention_mask=True,
                return_tensors="pt",
            )
            input_ids.append(encoding["input_ids"])
            attention_masks.append(encoding["attention_mask"])
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        return TensorDataset(input_ids, attention_masks)

    def select_optimizer(self, parameters):
        """
        Selects the optimizer based on the specified optimizer type.

        Args:
        parameters: Model parameters to optimize.

        Returns:
        Optimizer: The selected optimizer.
        """
        if self.optimizer_type == "AdamW":
            return torch.optim.AdamW(
                parameters, lr=self.learning_rate, eps=self.epsilon
            )
        elif self.optimizer_type == "SGD":
            return torch.optim.SGD(
                parameters, lr=self.learning_rate
            )
        elif self.optimizer_type == "RMSprop":
            return torch.optim.RMSprop(
                parameters, lr=self.learning_rate, eps=self.epsilon
            )
        elif self.optimizer_type == "Adam":
            return torch.optim.Adam(
                parameters, lr=self.learning_rate, eps=self.epsilon
            )
        elif self.optimizer_type == "Adagrad":
            return torch.optim.Adagrad(
                parameters, lr=self.learning_rate
            )
        elif self.optimizer_type == "Adadelta":
            return torch.optim.Adadelta(
                parameters, lr=self.learning_rate
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}")

    @staticmethod
    def flat_accuracy(preds, labels):
        """
        Calculates the accuracy of the predictions based on the comparison with true labels.

        Args:
        preds: Numpy array of predictions.
        labels: Numpy array of actual labels.

        Returns:
        float: Accuracy score.
        """
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def b_tp(self, preds, labels):
        """
        Calculates the number of true positives.

        Args:
        preds (list or array): Predicted labels.
        labels (list or array): Actual labels.

        Returns:
        int: Number of true positives.
        """
        return sum((pred == 1) and (label == 1) for pred, label in zip(preds, labels))

    def b_fp(self, preds, labels):
        """
        Calculates the number of false positives.

        Args:
        preds (list or array): Predicted labels.
        labels (list or array): Actual labels.

        Returns:
        int: Number of false positives.
        """
        return sum((pred == 1) and (label == 0) for pred, label in zip(preds, labels))

    def b_tn(self, preds, labels):
        """
        Calculates the number of true negatives.

        Args:
        preds (list or array): Predicted labels.
        labels (list or array): Actual labels.

        Returns:
        int: Number of true negatives.
        """
        return sum((pred == 0) and (label == 0) for pred, label in zip(preds, labels))

    def b_fn(self, preds, labels):
        """
        Calculates the number of false negatives.

        Args:
        preds (list or array): Predicted labels.
        labels (list or array): Actual labels.

        Returns:
        int: Number of false negatives.
        """
        return sum((pred == 0) and (label == 1) for pred, label in zip(preds, labels))

    def b_metrics(self, preds, labels):
        """
        Calculates binary classification metrics including accuracy, precision, recall, and specificity.

        Args:
        preds (array): Model's probability predictions before applying threshold.
        labels (array): Actual labels.

        Returns:
        tuple: A tuple containing the accuracy, precision, recall, and specificity.
        """
        preds = np.argmax(preds, axis=1).flatten()
        labels = labels.flatten()

        tp = self.b_tp(preds, labels)
        tn = self.b_tn(preds, labels)
        fp = self.b_fp(preds, labels)
        fn = self.b_fn(preds, labels)

        b_accuracy = (tp + tn) / len(labels)
        b_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        b_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        b_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        return b_accuracy, b_precision, b_recall, b_specificity

    def compute_roc_auc(self, preds, labels):
        """
        Computes the ROC-AUC score.

        Args:
        preds (np.array): Raw model predictions (logits or probabilities).
        labels (np.array): Actual labels.

        Returns:
        float: ROC-AUC score.
        """
        # Convert softmax output to positive class probabilities for ROC-AUC calculation
        preds_proba = np.exp(preds)[:, 1] / np.sum(np.exp(preds), axis=1)
        roc_auc = roc_auc_score(labels, preds_proba)
        return roc_auc

    def train(self, train_texts, train_labels, val_texts, val_labels):
        """
        Trains and evaluates the BERT model.

        Args:
        train_texts (list of str): Training texts.
        train_labels (list of int): Training labels.
        val_texts (list of str): Validation texts.
        val_labels (list of int): Validation labels.
        """
        train_dataset = self.preprocess(train_texts)
        val_dataset = self.preprocess(val_texts)

        train_labels = torch.tensor(train_labels)
        val_labels = torch.tensor(val_labels)

        train_dataset = TensorDataset(
            train_dataset.tensors[0], train_dataset.tensors[1], train_labels
        )
        val_dataset = TensorDataset(
            val_dataset.tensors[0], val_dataset.tensors[1], val_labels
        )

        train_dataloader = DataLoader(
            train_dataset,
            sampler=RandomSampler(train_dataset),
            batch_size=self.batch_size,
        )
        validation_dataloader = DataLoader(
            val_dataset,
            sampler=SequentialSampler(val_dataset),
            batch_size=self.batch_size,
        )

        
        # Initialize the optimizer
        optimizer = self.select_optimizer(self.model.parameters())

        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, eps=self.epsilon
        )

        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0

            for batch in train_dataloader:
                batch = tuple(b.to(self.device) for b in batch)
                b_input_ids, b_input_mask, b_labels = batch
                self.model.zero_grad()
                outputs = self.model(
                    b_input_ids,
                    token_type_ids=None,
                    attention_mask=b_input_mask,
                    labels=b_labels,
                )
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
            avg_train_loss = total_loss / len(train_dataloader)
            print(f"Epoch {epoch + 1}")
            print(f"Training loss: {avg_train_loss}")

            self.model.eval()
            eval_accuracy = 0
            eval_metrics = np.zeros(
                4
            )  # To store accuracy, precision, recall, specificity
            nb_eval_steps = 0

            logits_all = []
            label_ids_all = []
            
            for batch in validation_dataloader:
                batch = tuple(b.to(self.device) for b in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = self.model(
                        b_input_ids, token_type_ids=None, attention_mask=b_input_mask
                    )
                logits = outputs.logits.detach().cpu().numpy()
                label_ids = b_labels.to("cpu").numpy()

                logits_all.append(logits)
                label_ids_all.append(label_ids)

                tmp_eval_accuracy = self.flat_accuracy(logits, label_ids)
                eval_accuracy += tmp_eval_accuracy
                metrics = self.b_metrics(logits, label_ids)
                eval_metrics += np.array(metrics)
                nb_eval_steps += 1

            # Concatenate all logits and labels from the validation loop
            logits_all = np.concatenate(logits_all, axis=0)
            label_ids_all = np.concatenate(label_ids_all, axis=0)

            # Compute ROC-AUC after processing all validation batches
            roc_auc = self.compute_roc_auc(logits_all, label_ids_all)
            print(f"Validation ROC-AUC: {roc_auc}")

            
            print(f"Validation Accuracy: {eval_accuracy / nb_eval_steps}")
            print(f"Validation Precision: {eval_metrics[1] / nb_eval_steps}")
            print(f"Validation Recall: {eval_metrics[2] / nb_eval_steps}")
            print(f"Validation Specificity: {eval_metrics[3] / nb_eval_steps}")

            if roc_auc > self.best_val_roc_auc + self.tolerance:
                self.best_val_roc_auc = roc_auc
                self.epochs_no_improve = 0
                self.best_model_state = copy.deepcopy(
                    self.model.state_dict()
                )  # Save the best model state
            else:
                self.epochs_no_improve += 1
                print(
                    f"No improvement in validation ROC-AUC for {self.epochs_no_improve} consecutive epochs."
                )

                if self.epochs_no_improve >= self.patience:
                    print("Early stopping triggered.")
                    if self.best_model_state is not None:
                        print("Restoring best model weights!")
                        self.model.load_state_dict(
                            self.best_model_state
                        )  # Restore the best model state
                    break

    def save_model_and_tokenizer(self, save_directory):
        """
        Saves the trained model and tokenizer to the specified directory.

        Args:
        save_directory (str): The directory to save the model and tokenizer.
        """
        # Ensure the save directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model
        model_save_path = os.path.join(save_directory, "model")
        self.model.save_pretrained(model_save_path)

        # Save the tokenizer
        tokenizer_save_path = os.path.join(save_directory, "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_save_path)

        print(f"Model and tokenizer saved in {save_directory}")


    def push_to_huggingface_hub(self, commit_message="Push model to Hugging Face Hub"):
        """
        Pushes the fine-tuned model to the Hugging Face Hub, prompting the user for their username and access token.

        Args:
        commit_message (str): Commit message for the repository.
        """
        # Prompt the user for their Hugging Face username and access token

        hf_username = input("Enter your Hugging Face username: ")
        hf_access_token = input("Enter your Hugging Face access token: ")

        # Create a temporary directory to store model files

        temp_dir = tempfile.mkdtemp()

        # Save the model to the temporary directory

        self.model.save_pretrained(temp_dir)

        # Define the desired model name on the Hugging Face Hub

        model_name_on_hub = input("Enter your desired model name on Hugging Face Hub: ")

        # Initialize a repository in the temporary directory, set to push to the Hugging Face Hub

        repo = Repository(
            local_dir=temp_dir,
            clone_from=f"{hf_username}/{model_name_on_hub}",
            use_auth_token=hf_access_token,
        )

        # Push the model to the Hugging Face Hub

        repo.push_to_hub(commit_message=commit_message)



# =================== CLASSIFYING WITH TRAINED MODEL ====================


class BERTClassifier:
    class BERTClassifier:
        def __init__(self, model_path="jamesliounis/DataBERT", tokenizer_name="bert-base-uncased"):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model = BertForSequenceClassification.from_pretrained(model_path).to(self.device)
            self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

        def preprocess(self, input_texts):
            """
            Preprocesses a list of input texts for the BERT model.

            Args:
            input_texts (list of str): Texts to be preprocessed.

            Returns:
            Tuple containing two tensors: one for input IDs and one for attention masks.
            """
            # Batch encoding the texts, padding to the longest sequence in the batch
            encodings = self.tokenizer(input_texts, add_special_tokens=True, max_length=32, 
                                    padding=True, truncation=True, return_tensors="pt")
            
            input_ids = encodings["input_ids"].to(self.device)
            attention_mask = encodings["attention_mask"].to(self.device)

            return input_ids, attention_mask

        def predict(self, input_texts):
            """
            Generates predictions and confidence scores for a batch of texts.

            Args:
            input_texts (list of str): Batch of texts to generate predictions for.

            Returns:
            List of tuples where each tuple contains the predicted class and the confidence score.
            """
            self.model.eval()

            input_ids, attention_mask = self.preprocess(input_texts)

            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            probabilities = F.softmax(logits, dim=1)

            # Convert logits to list of predicted class IDs and confidence scores
            predictions = []
            for idx, logit in enumerate(logits):
                predicted_class_id = logit.argmax().item()
                confidence = probabilities[idx, predicted_class_id].item()
                predictions.append((1 if predicted_class_id == 1 else 0, confidence))

            return predictions


