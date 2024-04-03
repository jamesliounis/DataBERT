import json
from copy import deepcopy
from typing import Any, List
from langchain.docstore.document import Document
from langchain.text_splitter import TextSplitter


class DataUseTextSplitter(TextSplitter):
    """This text splitter takes in a batch of text and splits it into sentences.
    We iterate over the sentences and group them into sets not exceeding the maximum number of tokens specified.
    """
    def __init__(self,
                 tokenizer: Any,
                 separator: str = "\n\n",
                 chunk_size: int = 500,

                  **kwargs: Any):
        super().__init__(chunk_size=chunk_size, **kwargs)

        try:
            import tiktoken
        except ImportError:
            raise ValueError(
                "Could not import tiktoken python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install tiktoken`."
            )

        try:
            from nltk.tokenize import sent_tokenize
        except ImportError:
            raise ValueError(
                "Could not import nltk python package. "
                "This is needed in order to for TokenTextSplitter. "
                "Please install it with `pip install nltk`."
            )

        self._tokenizer = tokenizer
        self._sent_tokenize = sent_tokenize
        self._separator = separator

    def is_url(self, text: str) -> bool:
        return (("http://" in text) or ("https://" in text))

    def skip(self, text, min_sentence_len, skip_urls):
        text = text.strip()

        return (text == "") or (len(text) < min_sentence_len) or (skip_urls and self.is_url(text))

    def get_token_len(self, text):
        tokens = self._tokenizer.encode(text)
        if isinstance(tokens, dict):
            # Special case for transformers tokenizers
            tokens = tokens.get("input_ids")

        return len(tokens)

    def aggregate_documents(self, documents: List[Document], skip_urls: bool = True, min_sentence_len: int = 25):
        """Aggregate documents into a list of documents that satisfies the chunk constraints.
        This is basically the reverse of the split_text method. This is useful if you have a
        list of short documents.
        """
        texts = []
        curr_token_len = 0
        metadatas = []
        doc_groups = []
        meta_groups = []

        for doc in documents:
            metadatas.append(doc.metadata)
            sentences = self._sent_tokenize(doc.page_content)

            for sent in sentences:
                sent = sent.strip()

                if self.skip(sent, min_sentence_len, skip_urls):
                    continue

                num_tokens = self.get_token_len(sent)

                # +1 for the separator
                curr_token_len += (num_tokens + (1 if texts else 0))

                if curr_token_len > self._chunk_size:
                    # Try to accurately compute the number of tokens.
                    page_content = self._separator.join(texts).strip()
                    curr_token_len = self.get_token_len(page_content)

                    # Store only if adding the current sentence will actually exceed the chunk size.
                    if (curr_token_len + num_tokens) > self._chunk_size:
                        doc_groups.append(page_content)
                        meta_groups.append({"meta_group": json.dumps(metadatas), "tokens": curr_token_len})
                        texts = []
                        metadatas = [doc.metadata]
                        curr_token_len = 0

                    curr_token_len += num_tokens

                texts.append(sent)

        if texts:
            page_content = self._separator.join(texts).strip()
            curr_token_len = self.get_token_len(page_content)
            doc_groups.append(page_content)
            meta_groups.append({"meta_group": json.dumps(metadatas), "tokens": curr_token_len})

        return [Document(page_content=doc_group, metadata=meta_group) for doc_group, meta_group in zip(doc_groups, meta_groups)]

    def explode_documents(self, documents: List[Document], skip_urls: bool = True, min_sentence_len: int = 25) -> List[Document]:
        _documents = []

        for doc in documents:
            sentences = self._sent_tokenize(doc.page_content)

            for sent_idx, sent in enumerate(sentences):
                sent = sent.strip()

                metadata = deepcopy(doc.metadata)

                metadata["url_flag"] = self.is_url(sent)
                metadata["skip_flag"] = self.skip(sent, min_sentence_len, skip_urls)
                metadata["sent_idx"] = sent_idx
                metadata["num_chars"] = len(sent)
                metadata["num_words"] = len(sent.split())
                metadata["num_tokens"] = self.get_token_len(sent)
                metadata["has_covid"] = "covid" in sent.lower()

                _documents.append(Document(page_content=sent, metadata=metadata))

        return _documents

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return super().split_documents(documents)

    def split_text(self, text: str):
        return text.split(self._separator)
