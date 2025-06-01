from typing import Any, Dict, List, Literal, Optional

from langchain.base_language import BaseLanguageModel
from langchain.chains.base import Chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import Document
from pydantic import PrivateAttr


class CustomRetrievalQA(Chain):
    _llm: BaseLanguageModel = PrivateAttr()
    _vector_store: Any = PrivateAttr()
    _retrieval_method: str = PrivateAttr()
    _combine_documents_chain: Any = PrivateAttr()
    _return_source_documents: bool = PrivateAttr()
    _memory: Optional[BaseChatMemory] = PrivateAttr(default=None)

    def __init__(
        self,
        llm: BaseLanguageModel,
        vector_store,
        retrieval_method: Literal["default", "with_neighbors"] = "default",
        return_source_documents: bool = True,
        memory: Optional[BaseChatMemory] = None,
    ):
        super().__init__()
        self._llm = llm
        self._vector_store = vector_store
        self._retrieval_method = retrieval_method
        self._return_source_documents = return_source_documents
        self._memory = memory
        self._combine_documents_chain = load_qa_with_sources_chain(
            llm, chain_type="stuff"
        )

    @property
    def input_keys(self) -> List[str]:
        return ["question"]

    @property
    def output_keys(self) -> List[str]:
        return ["answer", "source_documents"]

    def _get_docs(self, question: str) -> List[Document]:
        if self._retrieval_method == "with_neighbors":
            return self._vector_store.similarity_search_with_neighbors(
                question, k=4, window=1
            )
        return self._vector_store.similarity_search(question, k=4)

    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        question = inputs["question"]

        history = ""
        if self._memory:
            memory_vars = self._memory.load_memory_variables({})
            history = memory_vars.get("chat_history", "")

        docs = self._get_docs(question)
        result = self._combine_documents_chain(
            {"input_documents": docs, "question": f"{history}\n{question}"}
        )
        stripped_answer = result["output_text"].split("SOURCES:")[0].strip()
        output = {"answer": stripped_answer}
        if self._return_source_documents:
            output["source_documents"] = docs

        if self._memory:
            self._memory.save_context(
                {"question": question}, {"answer": stripped_answer}
            )

        return output
