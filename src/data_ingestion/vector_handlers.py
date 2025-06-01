from abc import ABC, abstractmethod
from typing import List, Tuple

from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.vectorstores.azuresearch import AzureSearch


class VectorStoreInterface(ABC):
    @abstractmethod
    def add_documents(self, docs: List[Document]):
        pass

    @abstractmethod
    def save(self, path: str):
        pass

    @abstractmethod
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        pass

    @abstractmethod
    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        pass

    @abstractmethod
    def similarity_search_with_neighbors(
        self, query: str, k: int = 4, window: int = 1
    ) -> List[Document]:
        raise NotImplementedError(
            "This method is not implemented for this vector store."
        )

    @abstractmethod
    def load(self, path: str):
        pass

    @abstractmethod
    def as_retriever(self, search_type: str = "similarity", **kwargs):
        pass


class FAISSAdapter(VectorStoreInterface):
    def __init__(
        self, embedding_model=None, model_name: str = "text-embedding-ada-002"
    ):
        self.embedding_model = embedding_model or OpenAIEmbeddings(model=model_name)
        self.index = None

    def add_documents(self, docs: List[Document]):
        if self.index is None:
            self.index = FAISS.from_documents(docs, self.embedding_model)
        else:
            self.index.add_documents(docs)

    def save(self, path: str):
        if self.index:
            self.index.save_local(path)

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        if self.index is None:
            raise ValueError(
                "No index available. Please add documents first or load an existing index."
            )
        return self.index.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        if self.index is None:
            raise ValueError(
                "No index available. Please add documents first or load an existing index."
            )
        return self.index.similarity_search_with_score(query, k=k)

    def similarity_search_with_neighbors(
        self, query: str, k: int = 4, window: int = 1
    ) -> List[Document]:
        if self.index is None:
            raise ValueError("Index not loaded.")

        # Top-k base results
        hits = self.index.similarity_search_with_score(query, k=k)

        # Prepare lookup from FAISS's internal docstore
        all_docs = list(self.index.docstore._dict.values())
        grouped = {}
        for doc in all_docs:
            # Ensure source metadata exists
            if "source" not in doc.metadata:
                doc.metadata["source"] = doc.metadata.get("source_chunk", "unknown")

            chunk_ref = doc.metadata.get("source_chunk")
            if not chunk_ref or "/" not in chunk_ref:
                continue
            src, idx_str = chunk_ref.rsplit("/", 1)
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            grouped.setdefault(src, {})[idx] = doc

        # Pull neighbors around each hit
        enriched = set()
        results = []

        for doc, _ in hits:
            # Ensure source metadata exists
            if "source" not in doc.metadata:
                doc.metadata["source"] = doc.metadata.get("source_chunk")

            chunk_ref = doc.metadata.get("source_chunk")
            if not chunk_ref or "/" not in chunk_ref:
                results.append(doc)
                continue

            src, idx_str = chunk_ref.rsplit("/", 1)
            try:
                center_idx = int(idx_str)
            except ValueError:
                results.append(doc)
                continue

            for offset in range(-window, window + 1):
                i = center_idx + offset
                if i in grouped.get(src, {}) and (src, i) not in enriched:
                    neighbor = grouped[src][i]
                    results.append(neighbor)
                    enriched.add((src, i))
        print("results", results)
        return results

    def load(self, path: str):
        self.index = FAISS.load_local(
            path, self.embedding_model, allow_dangerous_deserialization=True
        )

    def as_retriever(self, search_type: str = "similarity", **kwargs):
        if self.index is None:
            raise ValueError(
                "No index available. Please add documents first or load an existing index."
            )
        return self.index.as_retriever(search_type=search_type, **kwargs)

    def get_unique_documents_metadata(self) -> List[dict]:
        if self.index is None:
            raise ValueError("Index not loaded.")

        seen_sources = set()
        documents_info = []

        for doc in self.index.docstore._dict.values():
            metadata = doc.metadata
            source = metadata.get("source_doc")
            date = metadata.get("creation_date", "unknown")
            num_pages = metadata.get("num_pages", "unknown")

            if source and source not in seen_sources:
                seen_sources.add(source)
                documents_info.append(
                    {"source": source, "date": date, "num_pages": num_pages}
                )

        return documents_info


class AzureSearchAdapter(VectorStoreInterface):
    def __init__(self, azure_search: AzureSearch):
        self.store = azure_search

    def add_documents(self, docs: List[Document]):
        self.store.add_documents(docs)

    def save(self, path: str):
        # Azure Search is cloud-based, no local saving needed
        pass

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        return self.store.similarity_search(query, k=k)

    def similarity_search_with_score(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        return self.store.similarity_search_with_score(query, k=k)

    def load(self, path: str):
        # Azure Search is cloud-based, no loading needed
        pass

    def as_retriever(self, search_type: str = "similarity", **kwargs):
        return self.store.as_retriever(search_type=search_type, **kwargs)
