from typing import List
from langchain.docstore.document import Document
from langchain.indexes import vectorstore as lang_vectorstore


class VectorstoreIndexCreator(lang_vectorstore.VectorstoreIndexCreator):
    """Logic for creating indexes."""

    def from_documents(self, documents: List[Document]) -> lang_vectorstore.VectorStoreIndexWrapper:

        vectorstore = self.vectorstore_cls.from_documents(
            documents, self.embedding, **self.vectorstore_kwargs
        )

        return lang_vectorstore.VectorStoreIndexWrapper(vectorstore=vectorstore)
