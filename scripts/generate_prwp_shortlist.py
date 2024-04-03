import json
from pathlib import Path
import tiktoken
from langchain.embeddings import HuggingFaceInstructEmbeddings
import fire

from data_use.indexes import vectorstore as dvectorstore
from data_use.document_loaders import pdf
from data_use import text_splitter as ts

passage_embeddings = HuggingFaceInstructEmbeddings(
    embed_instruction="Represent the passage for retrieval optimized for answering questions on the mention of data; Input: ",
    query_instruction="Represent the question for retrieving the most relevant passage mentioning data; Input: ",
)

enc = tiktoken.encoding_for_model("gpt-3.5-turbo")
du = ts.DataUseTextSplitter(tokenizer=enc, separator="\n\n", chunk_size=512)

# Set up paths
# proj_root = Path("/Users/avsolatorio/WBG/ChatPRWP")
proj_root = Path("~/ChatPRWP")
proj_root = proj_root.expanduser()
data_dir = proj_root / "data"
pdf_dir = data_dir / "pdf"
shortlist_dir = data_dir / "shortlist"
shortlist_dir.mkdir(parents=True, exist_ok=True)

cached_json_dir = data_dir / "s2orc-output_dir" / "prwp"

metadata_fname = data_dir / "raw" / "prwp_metadata_full.json"
metadata = json.loads(metadata_fname.read_text())


def build_shortlist(doc_path: Path, query: str, k: int = 3, expand_equation: bool = False, remove_citations: bool = False, skip_urls: bool = True):
    loader = pdf.S2ORCPDFLoader(doc_path)
    docs = loader.load(json_dir=cached_json_dir, expand_equation=expand_equation, remove_citations=remove_citations)
    documents = du.aggregate_documents(docs, skip_urls=skip_urls)

    index = dvectorstore.VectorstoreIndexCreator(
        embedding=passage_embeddings,
    ).from_documents(documents)

    shortlist = index.vectorstore.similarity_search(query, k=k)
    shortlist = [rs.page_content for rs in shortlist]

    try:
        # Delete the collection
        index.vectorstore.delete_collection()
    except:
        pass

    embedding_config = passage_embeddings.dict()
    embedding_config["client"] = embedding_config["client"].__repr__()
    doc_options = dict(
        expand_equation=expand_equation,
        remove_citations=remove_citations,
        skip_urls=skip_urls,
    )

    return dict(
        doc_id=doc_path.stem,
        query=query,
        k=k,
        shortlist=shortlist,
        embedding_config=embedding_config,
        doc_options=doc_options,
    )


def main(num_docs: int = 100, k: int = 3):
    query = "Was data or dataset used? Was data or dataset collected?"

    # Get the num_docs most recent PRWP documents
    prwp_docs = []

    for doc_id in sorted(metadata, key=lambda x: metadata[x].get("datestored", metadata[x].get("docdt", metadata[x].get("last_modified_date", ""))), reverse=True):
        pdf_path = pdf_dir / f"{doc_id}.pdf"
        if pdf_path.exists():

            # Generate the shortlist of snippets for each document that likely contains the answer to the question.
            # Then, store the shortlist in a file.
            shortlist_fname = shortlist_dir / f"{pdf_path.stem}.shortlist.json"

            if not shortlist_fname.exists():
                try:
                    json_data = build_shortlist(pdf_path, query, k=k, expand_equation=False, remove_citations=False, skip_urls=True)
                    shortlist_fname.write_text(json.dumps(json_data, indent=2))
                except KeyboardInterrupt:
                    raise KeyboardInterrupt
                except Exception as e:
                    print(f"Error in {pdf_path}: {e}")
                    continue

            prwp_docs.append(pdf_path)

        if len(prwp_docs) >= num_docs:
            break


if __name__ == "__main__":
    # python -m scripts.generate_prwp_shortlist --num_docs=100 --k=3
    fire.Fire(main)
