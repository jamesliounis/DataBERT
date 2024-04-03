import os
import json
import tempfile
from pathlib import Path
from typing import List, Union, Any
from langchain.docstore.document import Document
from langchain.document_loaders.pdf import BasePDFLoader


class S2ORCPDFLoader(BasePDFLoader):
    """Loads a PDF with pypdf and chunks at character level.

    Loader also stores page numbers in metadatas.
    """

    def __init__(self, file_path: Union[str, Path], grobid_config: dict = None):
        """Initialize with file path."""
        try:
            import doc2json  # noqa:F401
        except ImportError:
            raise ValueError(
                "doc2json package not found, please install it with " "`pipenv install -e git+https://github.com/avsolatorio/s2orc-doc2json.git@fix-date-parsing#egg=doc2json`"
            )
        if isinstance(file_path, Path):
            file_path = file_path.as_posix()

        super().__init__(file_path)

        self._grobid_config = grobid_config

    def _convert_pdf_to_json(self, input_file: Union[str, Path], temp_dir: Union[str, Path] = None) -> str:
        from doc2json.grobid2json.grobid.grobid_client import GrobidClient
        from doc2json.grobid2json.tei_to_json import convert_tei_xml_file_to_s2orc_json

        if temp_dir is None:
            temp_dir = Path("/tmp/data-use_grobid").resolve()

        input_file = Path(input_file).resolve()
        assert input_file.exists()

        temp_dir.mkdir(parents=True, exist_ok=True)

        # get paper id as the name of the file
        paper_id = input_file.stem
        tei_file = temp_dir / f'{paper_id}.tei.xml'

        # process PDF through Grobid -> TEI.XML
        client = GrobidClient(self._grobid_config)
        # TODO: compute PDF hash
        # TODO: add grobid version number to output
        client.process_pdf(input_file.as_posix(), temp_dir, "processFulltextDocument")

        # process TEI.XML -> JSON
        assert tei_file.exists()
        paper = convert_tei_xml_file_to_s2orc_json(tei_file.as_posix())

        return paper.release_json()

    def load(self, json_dir: Union[str, Path] = None, expand_equation: bool = False, remove_citations: bool = False) -> List[Document]:
        """Load given path as spans extracted from the PDF using doc2json."""

        doc_json = None
        json_file = None

        if json_dir is not None:
            json_dir = Path(json_dir).resolve()
            assert json_dir.exists() and json_dir.is_dir()

            # Try to find the JSON file in the directory if available
            json_file = json_dir / f"{Path(self.file_path).stem}.json"
            if json_file.exists():
                doc_json = json.loads(json_file.read_text())

        if doc_json is None:
            # Convert PDF to JSON if JSON file is not found
            doc_json = self._convert_pdf_to_json(self.file_path)

        if json_file and not json_file.exists():
            # Save the JSON file if it doesn't exist
            json_file.write_text(json.dumps(doc_json, indent=4, sort_keys=False))

        pdf_parse = doc_json.get("pdf_parse")

        if pdf_parse is None:
            raise ValueError("PDF parse not found in JSON")

        body_text = pdf_parse.get("body_text")

        if body_text is None:
            raise ValueError("Body text not found in PDF parse")

        docs = []
        for i, span in enumerate(body_text):
            doc = dict()

            if expand_equation:
                if span['text'] == "EQUATION" and span.get("eq_spans"):
                    # If the text is "EQUATION" and there are equations, we use the equations to generate the text.
                    span['text'] = ". ".join([eq.get("raw_str") for eq in span.get("eq_spans") if eq.get("raw_str")])
            else:
                # If we don't want to expand equations, we skip the span if it's an equation.
                if span['text'] == "EQUATION":
                    continue

            if remove_citations:
                # If we want to remove citations, we skip the span if it's a citation.
                citations = span.get("cite_spans", [])
                if citations:
                    # Sort the citations by start position in descending order.
                    citations = sorted(citations, key=lambda x: x.get("start"), reverse=True)
                    for citation in citations:
                        # Remove the citation from the text.
                        span['text'] = span['text'][:citation.get("start")] + span['text'][citation.get("end"):]

            doc["page_content"] = span["text"]

            doc["metadata"] = dict(
                source=self.file_path,
                span=i,
                cite_spans=span.get("cite_spans", []),
                ref_spans=span.get("ref_spans", []),
                eq_spans=span.get("eq_spans", []),
                section=span.get("section"),
                sec_num=span.get("sec_num"),
            )

            docs.append(Document(**doc))

        return docs
