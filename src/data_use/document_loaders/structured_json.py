from typing import List, Union, Any
from pathlib import Path
import json
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document


class StructuredJSONLoader(BaseLoader):
    """Loads a JSON file and references a schema provided to load the text into
    documents.

    Example:
    [{"text": ...}, {"text": ...}, {"text": ...}] -> schema = .[].text
    {"key": [{"text": ...}, {"text": ...}, {"text": ...}]} -> schema = .key[].text
    ["", "", ""] -> schema = .[]
    """

    def __init__(self, file_path: Union[str, Path], jq_schema: str):
        """Initialize with file path."""
        try:
            import jq  # noqa:F401
        except ImportError:
            raise ValueError(
                "jq package not found, please install it with " "`pipenv install jq`"
            )

        self.file_path = Path(file_path).resolve()
        self.jq_schema = jq.compile(jq_schema)

    def load(self) -> List[Document]:
        """Load given path as spans extracted from the PDF using doc2json."""

        body_text = self.jq_schema.input(json.loads(self.file_path.read_text())).all()
        docs = []

        for i, text in enumerate(body_text, 1):
            doc = dict()
            doc["page_content"] = text

            doc["metadata"] = dict(
                source=self.file_path.as_posix(),
                seq_num=i,
            )

            docs.append(Document(**doc))

        return docs


if __name__ == "__main__":
    pass

"""
loader = sj.StructuredJSONLoader(coleridge_dir / f"{ids[3]}.json", jq_schema=".[].text")
docs = loader.load()
documents = du.aggregate_documents(docs)

index = dvectorstore.VectorstoreIndexCreator(
    embedding=passage_embeddings,
).from_documents(documents)

shortlist = index.vectorstore.similarity_search(query, k=4)
print("\n=====\n".join([o.page_content for o in shortlist]))
"""