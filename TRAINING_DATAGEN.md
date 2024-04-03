# Generating training data for re-ranking

We use random sampled data from the Coleridge Initiative competition and the World Bank's PRWP dataset to generate training data for re-ranking. The training data is generated using the following steps:

1. We randomly sample 100 papers from the Coleridge Initiative competition and 100 papers from the World Bank's PRWP dataset.
2. We store these in a directory called `data/training/ranking/coleridge` and `data/training/ranking/prwp` respectively.

We use the snippet below to generate the sample data.

```Python
import os
import json
import random
from pathlib import Path

RANKING_DATA_DIR = Path("data/training/ranking")
N = 100

# Coleridge Initiative
coleridge_dir = Path("data/coleridgeinitiative/coleridgeinitiative-show-us-the-data/train")
paths = list(coleridge_dir.glob("*.json"))
random.seed(1029)
random.shuffle(paths)
ids = []

for path in paths:
    name = path.name

    if os.path.getsize(path) > 3_000_000: # skip large files
        continue

    with open(path) as f:
        doc = json.load(f)

    fname = RANKING_DATA_DIR / "coleridge" / f"{name}"
    fname.parent.mkdir(exist_ok=True, parents=True)

    fname.write_text(json.dumps(doc, indent=2))

    ids.append(name)
    if len(ids) == N:
        break

# World Bank PRWP
prwp_dir = Path("data/prwp")
paths = list(prwp_dir.glob("*.json"))
random.seed(1029)
random.shuffle(paths)
ids = []

for path in paths:
    name = path.name

    if os.path.getsize(path) > 3_000_000: # skip large files
        continue

    with open(path) as f:
        doc = json.load(f)

    fname = RANKING_DATA_DIR / "prwp" / f"{name}"
    fname.parent.mkdir(exist_ok=True, parents=True)

    fname.write_text(json.dumps(doc, indent=2))

    ids.append(name)
    if len(ids) == N:
        break


```

3. We then use the following snippet to generate the shortlist of sentences for annotation.

```Python
import json
import random
from pathlib import Path
import tiktoken
from tqdm.auto import tqdm
from langchain.vectorstores import Qdrant, Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
import fire

from data_use.indexes import vectorstore as dvectorstore
from data_use.document_loaders import structured_json
from data_use import text_splitter as ts

sentence_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the sentence for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant sentence mentioning data; Input: ",
)

QUERY = "What dataset was used in the paper?"


def get_sample_sentences_from_path(doc_path, jq_schema, query, sentence_embeddings, k=20, split_docs: bool = False, skip_urls: bool = True, min_sentence_len: int = 25, vectorstore_cls = "Qdrant"):
    # path = coleridge_dir / f"{ids[id_num]}.json"
    # cole_jq_schema = ".[].text"
    # prwp_jq_schema=".pdf_parse.body_text[].text"
    loader = sj.StructuredJSONLoader(doc_path, jq_schema=jq_schema)
    docs = loader.load()
    documents = du.aggregate_documents(docs, skip_urls=skip_urls, min_sentence_len=min_sentence_len)

    if split_docs:
        sdocs = du.split_documents(documents)
    else:
        sdocs = documents

    collection_name = None

    if vectorstore_cls == "Qdrant":
        collection_name = "coleridge"
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Qdrant,
            embedding=sentence_embeddings,
            vectorstore_kwargs=dict(
                location=":memory:",
                collection_name=collection_name,
            ),
        ).from_documents(sdocs)
    elif vectorstore_cls == "Chroma":
        index = dvectorstore.VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=sentence_embeddings,
        ).from_documents(sdocs)
    else:
        raise ValueError(f"Unknown vectorstore_cls: {vectorstore_cls}")

    shortlist = index.vectorstore.similarity_search(query, k=k)
    shortlist = [rs.page_content for rs in shortlist]

    try:
        # Delete the collection
        # ChromaDB vectorstore
        index.vectorstore.delete_collection()
    except AttributeError:
        try:
            # Delete the collection
            # Qdrant vectorstore
            if collection_name is not None:
                index.vectorstore.client.delete_collection(collection_name)
        except:
            pass

    embedding_config = sentence_embeddings.dict()
    embedding_config["client"] = embedding_config["client"].__repr__()
    doc_options = dict(
        skip_urls=skip_urls,
        min_sentence_len=min_sentence_len,
    )

    return dict(
        doc_id=doc_path.stem,
        query=query,
        k=k,
        shortlist=shortlist,
        embedding_config=embedding_config,
        vectorstore_cls=vectorstore_cls,
        doc_options=doc_options,
    )

# PRWP 10330 - D34010638
out = get_sample_sentences_from_path(Path("data/prwp/D34010638.json"), jq_schema=".pdf_parse.body_text[].text", query="What dataset was used in the paper?", sentence_embeddings=sentence_embeddings, split_docs=True, k=50)


jq_schema_map = dict(
    coleridge=".[].text",
    prwp=".pdf_parse.body_text[].text",
)

for dataset in ["coleridge", "prwp"]:
    print(f"Processing {dataset}")
    data_dir = RANKING_DATA_DIR / dataset
    out_dir = RANKING_DATA_DIR / f"shortlist"
    out_dir.mkdir(exist_ok=True, parents=True)

    for path in tqdm(sorted(data_dir.glob("*.json"))):
        print(path)
        out_path = out_dir / f"{dataset}_{path.stem}.json"

        if out_path.exists():
            continue

        out = get_sample_sentences_from_path(
            path, jq_schema=jq_schema_map[dataset], query=QUERY, sentence_embeddings=sentence_embeddings, split_docs=True, k=50, skip_urls=True, min_sentence_len=25, vectorstore_cls="Qdrant")

        out_path.write_text(json.dumps(out, indent=2))
```

4. We then use the following snippet to generate the shortlist of sentences for annotation.

```Python
import json
import pandas as pd
from pathlib import Path

all_df = []
RANKING_DIR = Path("data/training/ranking")
for path in sorted((RANKING_DIR / "shortlist").glob("*.json")):
    with open(path) as f:
        doc = json.load(f)

    df = pd.DataFrame(doc["shortlist"], columns=["text"])
    df["doc_id"] = doc["doc_id"]
    df["rank"] = df.index + 1

    all_df.append(df)

all_df = pd.concat(all_df)
all_df["is_relevant"] = None

all_df["sent_id"] = list(range(1, all_df.shape[0] + 1))
all_df["sent_id"] = "s_" + all_df["sent_id"].astype(str).str.zfill(5)

all_df = all_df.sort_values(["rank", "doc_id"])

all_df.to_excel((RANKING_DIR / "ranking_shortlist.xlsx"), index=None)
```

1. We use GPT to pre-label the shortlist of sentences. We will pass batches of sentences to GPT for labeling.

```bash
pipenv run python -m scripts.ranking.gpt_label_ranking_data
```



```bash
pipenv run python -m scripts.ranking.process_gpt_labels
```


## Generate the additional data from GPT

```bash
pipenv run python -m scripts.ranking.process_gpt_sentences
```

## Train the reranking model
Colab notebook: https://colab.research.google.com/drive/1l20s_JBykPGFo1OrDpblUyoW43Yt1oi5




## Inference


```Python
import numpy as np
import torch
from scipy.special import softmax
from tqdm.auto import tqdm

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import DataCollatorWithPadding

from transformers import pipeline

model_path = "data/training/ranking/models/distilbert-base-cased_t1682997237"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

classifier = pipeline("text-classification", model=model_path)


def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True)


def make_predictions(df, model, batch=50):
    test_inputs = df.apply(preprocess_function, axis=1)
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


# train_scores = make_predictions(train_df, model, batch=50)
# train_df["probs"] = train_scores
# train_df.groupby("gpt_label")["probs"].describe()
```



If you are uncertain about a sentence, do not include it in the list.


Forget all previous instructions.

Given a list of sentences extracted from papers, extract the sentences that explicitly mention a dataset used in the paper. Only include sentences where the dataset is clearly and unambiguously named, such as LSMS or DHS. Exclude sentences that mention software used for data analysis, equipment used to collect data, or other tools that are not datasets. Also, exclude sentences that only cite a reference without mentioning the dataset or refer to data computed by the author without mentioning the dataset used to derive it.

Your task is to return the sentence ids of the identified sentences as a JSON array along with the reason why you included them. The output format should be [{"sent_id": s_XXXXX, "mentioned": <true|false>, "reason": <dataset name>}]. If a sentence mentions a dataset clearly and unambiguously, set "mentioned" to true and fill in the "reason" field with the name of the dataset. If a sentence does not mention a dataset, set "mentioned" to false and leave the "reason" field empty.

Do not explain.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Text:


[{"sent_id": "s_08023", "dataset_mentioned": true, "reason": "2004 ISSP 6 household survey data"}, {"sent_id": "s_08073", "dataset_mentioned": true, "reason": "Brazilian data"}, {"sent_id": "s_08173", "dataset_mentioned": true, "reason": "Current Population Survey"}, {"sent_id": "s_08450", "dataset_mentioned": true, "reason": "ADAS-cog"}, {"sent_id": "s_08550", "dataset_mentioned": true, "reason": "Alzheimer's Disease Neuroimaging Initiative (ADNI) database"}, {"sent_id": "s_08600", "dataset_mentioned": true, "reason": "DEA's Automation of Reports and Consolidated Orders System (ARCOS) data"}, {"sent_id": "s_08800", "dataset_mentioned": true, "reason": "Alzheimer's Disease Neuroimaging Initiative (ADNI) database and the Australian Imaging, Biomarkers and Lifestyle (AIBL) study"}, {"sent_id": "s_08850", "dataset_mentioned": true, "reason": "Citizenship Education Longitudinal Study (CELS)"}, {"sent_id": "s_08900", "dataset_mentioned": false, "reason": ""}]



Forget all previous instructions.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Given a list of sentences extracted from papers, extract the sentences that explicitly mention a dataset used in the paper. Only include sentences where the dataset is clearly and unambiguously named, such as LSMS or DHS. Exclude sentences that mention software used for data analysis, equipment used to collect data, or other tools that are not datasets. Also, exclude sentences that only cite a reference without mentioning the dataset. Also exclude sentences that refer to data computed by the author without mentioning the dataset used to derive it.

Your task is to return the sentence ids of the identified sentences as a JSON array along with the reason why you included them. The output format should be [{"sent_id": s_XXXXX, "data_mentioned": true, "reason": <dataset name>}]. If a sentence mentions a dataset clearly and unambiguously, set "data_mentioned" to true and fill in the "reason" field with the name of the dataset

Do not explain.

Text:




Forget all previous instructions.

Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Given a list of sentences extracted from papers, extract the sentences that explicitly mention a dataset used in the paper. Only include sentences where the dataset is clearly and unambiguously named, such as LSMS or DHS. Exclude sentences that mention software used for data analysis, equipment used to collect data, or other tools that are not datasets. Also, exclude sentences that only cite a reference without mentioning the dataset. Also exclude sentences that refer to data computed by the author without mentioning the dataset used to derive it.

Your task is to return the sentence ids of the identified sentences as a JSON array along with the reason why you included them. The output format should be [{"sent_id": s_XXXXX, "data_mentioned": true, "reason": <dataset name>}]. If a sentence mentions a dataset clearly and unambiguously, set "data_mentioned" to true and fill in the "reason" field with the name of the dataset

Do not explain.

Text:





Forget all previous instructions.

You are given a text delimited by triple backticks. The text contains a list of sentences.

Perform the following actions:

1. Create a shortlist of sentences that are likely to contain mentions of one or more datasets.
2. Remove from this shortlist sentences that do not explicitly mention a name of a dataset.
3. Further filter the shortlist of sentences that mention software used for data analysis, equipment used to collect data, or other tools that are not datasets.
4. Finally, exclude sentences that refer to data computed by the author without mentioning the dataset used to derive it.

Return the remaining sentences a list. The output format should be [{"sent_id": s_XXXXX, "data_mentioned": true, "reason": <dataset name>}]. If a sentence mentions a dataset clearly and unambiguously, set "data_mentioned" to true and fill in the "reason" field with the name of the dataset. If a sentence does not mention a dataset, set "data_mentioned" to false and leave the "reason" field empty.


Always return the result in a valid JSON format. Do not truncate the output and never generate ... in the response. The output should not raise a JSONDecodeError when loaded in Python.

Given a list of sentences extracted from papers, extract the sentences that explicitly mention a dataset used in the paper. Only include sentences where the dataset is clearly and unambiguously named, such as LSMS or DHS. Exclude sentences that mention software used for data analysis, equipment used to collect data, or other tools that are not datasets. Also, exclude sentences that only cite a reference without mentioning the dataset. Also exclude sentences that refer to data computed by the author without mentioning the dataset used to derive it.

Your task is to return the sentence ids of the identified sentences as a JSON array along with the reason why you included them. The output format should be [{"sent_id": s_XXXXX, "data_mentioned": true, "reason": <dataset name>}]. If a sentence mentions a dataset clearly and unambiguously, set "data_mentioned" to true and fill in the "reason" field with the name of the dataset

Do not explain.

Text:

s_07924: Until the study described in this paper was undertaken, there was little information available on household demand for improved water services in Nigeria which could help clarify the issues involved in this policy discussion (for an exception, see Reedy, 1987) .

s_07974: 6 SeeTompson (2004), p. 27 for a detailed discussion of this issue.

s_08024: Note: this table is based on the discussion in Komives et.

s_08074: 52 SeeKeefer and Knack (2006) andPritchett (2000).

s_08124: These data were kindly provided by Leandro Prados, who used them recently in Álvarez-Nogal and Prados de la Escosura 2006, which in turn were taken from Yun Casalilla (1987: p. 465) and Ramos Palencia (2001: p. 70 ).

s_08174: In a recent paper, four authors of this brief -Athreya, Ionescu, Neelakantan, and Vidangos -examine how the value of college varies across the population of U.S. high school graduates and the importance of the college subsidy to this valuation.

s_08224: Our second article has an analytical flavor, crunching data from the US Department of Education's High School Longitudinal Study, which began in 2009 to follow students who were then entering the ninth grade.

s_08251: 18 Data is used by IHEs to examine existing curricula in science-related fields and businesses to understand and monitor employment trends in various industries and professions.

s_08301: The resulting dataset provides some value added regarding its sources.

s_08351: The discovery GWAS meta analysis datasets used in the study contain large sample sizes (in total 54,162 for AD and 23,986 for serum iron status) and show both AD and serum iron measures to have a strong polygenic components 27, 31 .

s_08401: Data used in the preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database.

s_08451: The authors of that review on stroke therapeutics make several conclusions that are highly relevant to the AD field regarding alignment of preclinical studies and human clinical trials design.

s_08501: The WIC dataset has been already used in a number of scientific and action papers.

s_08551: The following data from 819 ADNI participants were downloaded from the ADNI database: all baseline 1.5 T MRI scans, the RBM (Rules-Based Medicine) multiplex proteomic analytes extracted from plasma and serum, and demographic and baseline diagnosis information.

s_08601: Therefore, we used data on buprenorphine provision in substance abuse treatment facilities from the 2004-2006 and 2008-2011 N-SSATS to impute substance abuse treatment facilities' use of buprenorphine in 2007.

s_08651: See Mullis, et al., (2004) and Martin, et al., (2004) for details of the findings.

s_08701: The program reads observational data and forecast system output, fills in missing data, creates new data structures for comparison purposes, carries out the comparisons, and presents the results in tabular form.

s_08751: www.cts-journal.com Table 2 Unadjusted associations between past participation, past opportunity, and willingness to participate and sociodemographic characteristics, access to care, and health status   Data Source: Behavioral Risk Factor Surveillance System, 2015.

s_08801: First the datasets were converted to the BIDS format, then the t1-volume preprocessing pipeline of Clinica was applied [10] .

s_08851: The transcripts of these interviews were coded using thematic analysis (Braun and Clarke, 2006) , allowing for both inductive (i.e., unanticipated themes emerging from the data) and deductive coding.



