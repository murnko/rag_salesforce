from typing import Any

from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from retrieval.retriever import CustomRetrievalQA


class MetadataTool(BaseTool):
    name: str = "metadata_tool"
    description: str = (
        "Tool to retrieve vector store metadata like document count and creation dates."
    )
    vectorstore: Any = None

    def __init__(self, vectorstore):
        super().__init__()
        self.vectorstore = vectorstore

    def _run(self, question: str) -> str:
        try:
            documents_info = self.vectorstore.get_unique_documents_metadata()
        except Exception:
            return "Metadata tool failed to retrieve information."

        if not documents_info:
            return "No document metadata found in vector store."

        doc_count = len(documents_info)
        summary_lines = [f"Indexed {doc_count} unique documents:"]
        for doc in documents_info:
            summary_lines.append(
                f"- {doc['source']} | Date: {doc.get('date')} | Pages: {doc.get('num_pages')}"
            )

        return "\n".join(summary_lines)


class RetrievalGraph:
    class GraphState(TypedDict):
        question: str
        answer: str
        source_documents: list[Document]

    def __init__(self, retriever_chain: CustomRetrievalQA):
        self.llm = ChatOpenAI(temperature=0)
        self.retriever_chain = retriever_chain
        self.vectorstore = retriever_chain._vector_store
        self.metadata_tool = MetadataTool(self.vectorstore)
        self.graph = self._build_graph()

    def _default_retriever(self, state: dict) -> dict:
        return self.retriever_chain._call({"question": state["question"]})

    def _metadata_tool_node(self, state: dict) -> dict:
        question = state["question"]

        try:
            documents_info = self.vectorstore.get_unique_documents_metadata()
        except Exception:
            raise RuntimeError("Failed to retrieve metadata from vector store")

        doc_context = "\n".join(
            f"- {doc['source']} | Date: {doc.get('date', 'N/A')} | Pages: {doc.get('num_pages', 'N/A')}"
            for doc in documents_info
        )

        system_prompt = (
            "You are an assistant with access to structured metadata "
            "about indexed earnings call documents. Based on the metadata, answer the user's question."
        )

        prompt = f"{system_prompt}\n\nMetadata:\n{doc_context}\n\nQuestion: {question}\nAnswer:"

        response = self.llm.invoke(prompt).content.strip()

        return {"question": question, "answer": response, "source_documents": []}

    def _route_question(self, question: str) -> str:
        prompt = f"""
        You are a router that decides if a question should be handled by the default retriever or a metadata tool.
        Answer with 'metadata_tool_node' for questions like:
        - "When was the most recent earnings call?"
        - "How many earnings call documents do you have indexed?"
        - "How many pages are in the most recent earnings call?"
        Otherwise, answer with 'default_retriever'.

        Question: {question}
        Only answer with 'metadata_tool_node' or 'default_retriever'.
        """
        routing_decision = self.llm.invoke(prompt).content.strip().lower()
        return (
            routing_decision
            if routing_decision in ["default_retriever", "metadata_tool_node"]
            else "default_retriever"
        )

    def _build_graph(self):
        graph_builder = StateGraph(self.GraphState)
        graph_builder.add_node("default_retriever", self._default_retriever)
        graph_builder.add_node("metadata_tool_node", self._metadata_tool_node)
        graph_builder.add_conditional_edges(
            START,
            self._route_question,
            {
                "default_retriever": "default_retriever",
                "metadata_tool_node": "metadata_tool_node",
            },
        )
        graph_builder.add_edge("default_retriever", END)
        graph_builder.add_edge("metadata_tool_node", END)
        return graph_builder.compile()

    def invoke(self, question: str) -> dict:
        return self.graph.invoke({"question": question})
