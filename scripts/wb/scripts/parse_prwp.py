import datasets
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import fire

from wb.wb import pdf
from wb.wb import text_splitter as ts

HF_ORGANIZATION = "avsolatorio"
PDF_DIR = "scripts/wb/data/pdfs"
DOCS_DIR = "scripts/wb/data/docs"
JSON_CACHE_DIR = "scripts/wb/data/json"
EXPAND_EQUATIONS = True
REMOVE_CITATIONS = False
TOKENIZER = "BAAI/bge-base-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)


def get_doc(pdf_path):
    loader = pdf.S2ORCPDFLoader(pdf_path)
    docs = loader.load(json_dir=JSON_CACHE_DIR, expand_equation=EXPAND_EQUATIONS, remove_citations=REMOVE_CITATIONS)

    du = ts.DataUseTextSplitter(tokenizer=tokenizer, separator="\n\n", chunk_size=512)
    docs = du.aggregate_documents(docs)

    return docs


def get_sent_doc(pdf_path):
    loader = pdf.S2ORCPDFLoader(pdf_path)
    docs = loader.load(json_dir=JSON_CACHE_DIR, expand_equation=EXPAND_EQUATIONS, remove_citations=REMOVE_CITATIONS)

    du = ts.DataUseTextSplitter(tokenizer=tokenizer, separator="\n\n", chunk_size=512)  # The separator and chunk_size are not used.
    docs = du.explode_documents(docs)

    return docs


def create_dataset(mode: str = "chunk"):
    assert mode in ["chunk", "sent"]

    dataset = {}
    curr_count = 0

    pdf_dir = Path(PDF_DIR)
    for pdf_file in tqdm(sorted(pdf_dir.glob("*.pdf"))):
        # Running this will save the JSON files in the JSON_CACHE_DIR
        try:
            if mode == "chunk":
                docs = get_doc(pdf_file)
            else:
                docs = get_sent_doc(pdf_file)
        except Exception as e:
            print(f"Failed to parse {pdf_file}.")
            continue

        for doc in docs:
            l = doc.dict()

            # Flatten the metadata
            l = {"page_content": l["page_content"],  **l["metadata"]}

            for k, v in l.items():
                if k not in dataset:
                    # Initialize the list and fill with None
                    # for the previous entries.
                    dataset[k] = [None] * curr_count

                dataset[k].append(v)

            curr_count += 1

    data = datasets.Dataset.from_dict(dataset, split="train")
    data.push_to_hub(f"{HF_ORGANIZATION}/wb-prwp-{mode}", private=True, commit_message=f"Add {mode} new dataset.")

if __name__ == "__main__":
    # pipenv run python scripts/wb/scripts/parse_prwp.py create_dataset --mode sent
    fire.Fire(create_dataset)
