from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

# from transformers import pipeline
DATA_DIR = Path(__file__).parent.parent.parent / "data"

# model_path = DATA_DIR / "training" / "ranking" / "models" / "distilbert-base-cased_t1682997237"

model_path = "avsolatorio/distilbert-base-cased_t1682997237"
print(model_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# classifier = pipeline("text-classification", model=model_path)

def sentence_prob_for_texts(texts, batch=50):
    if isinstance(texts, str):
        texts = [texts]

    model.eval()
    sent_prob = []

    with torch.no_grad():
        for i in tqdm(range(0, len(texts) + batch, batch)):
            ti = texts[i:i + batch]
            if not ti:
                break

            inputs = tokenizer(ti, truncation=True, padding=True, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            o = model(**inputs)
            sent_prob.append(softmax(o.logits.cpu(), axis=1))

            torch.cuda.empty_cache()

    sent_prob_arr = np.vstack(sent_prob)[:, 1]

    return sent_prob_arr


def make_predictions_from_df(df, model, batch=50):

    def _preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)

    test_inputs = df.apply(_preprocess_function, axis=1)
    data_sent_prob = []

    with torch.no_grad():
        model.eval()

        for i in tqdm(range(0, len(test_inputs) + batch, batch)):
            ti = test_inputs.iloc[i:i + batch]
            if ti.empty:
                break

            ti = ti.tolist()

            ti = {k: v.to(model.device) for k, v in data_collator(ti).items()}
            o = model(**ti)

            data_sent_prob.append(softmax(o.logits.cpu(), axis=1))

            torch.cuda.empty_cache()

    data_sent_prob_arr = np.vstack(data_sent_prob)[:, 1]

    return data_sent_prob_arr


def rank_texts(texts, scores):
    texts = np.array(texts)
    scores = np.array(scores)

    idx = np.argsort(scores)[::-1]
    return texts[idx].tolist()
