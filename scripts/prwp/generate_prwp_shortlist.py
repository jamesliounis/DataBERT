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
from data_use import ranking


sentence_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the sentence for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant sentence mentioning data; Input: ",
)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)

jq_schema_map = dict(
    coleridge=".[].text",
    prwp=".pdf_parse.body_text[].text",
)


def build_shortlist(doc_path: Path, query: str, dataset: str, k: int = 25, skip_urls: bool = True, min_sentence_len: int = 25, vectorstore_cls = "Qdrant"):
    loader = structured_json.StructuredJSONLoader(doc_path, jq_schema=jq_schema_map[dataset])
    docs = loader.load()
    documents = du.aggregate_documents(docs, skip_urls=skip_urls, min_sentence_len=min_sentence_len)
    sdocs = du.split_documents(documents)
    # sdocs = documents

    collection_name = None

    if vectorstore_cls == "Qdrant":
        collection_name = dataset
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
    shortlist = [{"rank": i, "text": rs.page_content} for i, rs in enumerate(shortlist, 1)]

    sent_probs = ranking.sentence_prob_for_texts([i["text"] for i in shortlist])
    sent_probs = sent_probs.astype(float).round(8).tolist()

    shortlist = [{**sh, **{"prob": sc}} for sh, sc in zip(shortlist, sent_probs)]
    shortlist = sorted(shortlist, key=lambda x: x["prob"], reverse=True)

    # ranked_shortlist = ranking.rank_texts(shortlist, sent_probs)
    # sent_probs = sent_probs.astype(float).round(8).tolist()

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


def main(dataset: str, num_docs: int = 100, k: int = 25, force: bool = False):
    assert jq_schema_map.get(dataset) is not None, f"jq_schema not registered for dataset: {dataset}"

    # Set up paths
    # proj_root = Path("/Users/avsolatorio/WBG/ChatPRWP")
    proj_root = Path("~/WBG/data-use/")
    proj_root = proj_root.expanduser()
    data_dir = proj_root / "data" / dataset
    json_source_dir = data_dir / "raw"

    assert json_source_dir.exists(), f"json_source_dir does not exist: {json_source_dir}"

    shortlist_dir = data_dir / "shortlist"
    shortlist_dir.mkdir(parents=True, exist_ok=True)

    RANKING_DATA_DIR = proj_root / "data" / "training" / "ranking" / dataset

    # query = "Was data or dataset used in this sentence?"
    query = "What dataset was used in the paper?"  # Query used to generate the ranking shortlist

    json_list = sorted(json_source_dir.glob("*.json"))
    random.seed(1029)
    random.shuffle(json_list)

    # Get the num_docs most recent PRWP documents
    _docs = []

    for json_path in tqdm(json_list):
        if json_path.exists():

            # Generate the shortlist of snippets for each document that likely contains the answer to the question.
            # Then, store the shortlist in a file.
            shortlist_fname = shortlist_dir / f"{json_path.stem}.shortlist.json"

            ranking_file = RANKING_DATA_DIR / f"{json_path.stem}.json"

            if ranking_file.exists():
                # Make sure we exclude documents that have been used for training
                # the ranking model.
                if shortlist_fname.exists():
                    shortlist_fname.unlink()
                continue

            if not shortlist_fname.exists() or force:
                try:
                    json_data = build_shortlist(json_path, query, dataset, k=k, skip_urls=True, min_sentence_len=25, vectorstore_cls="Qdrant")
                    shortlist_fname.write_text(json.dumps(json_data, indent=2))
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    print(f"Error in {json_path}: {e}")
                    continue

            _docs.append(json_path)

        if len(_docs) >= num_docs:
            break


if __name__ == "__main__":
    # python -m scripts.prwp.generate_prwp_shortlist --dataset=prwp --num_docs=100 --k=50 --force
    fire.Fire(main)
