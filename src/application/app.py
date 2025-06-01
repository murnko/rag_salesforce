import os

import chainlit as cl
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores.azuresearch import AzureSearch

from config import settings
from data_ingestion.vector_handlers import AzureSearchAdapter, FAISSAdapter
from retrieval.graph_router import RetrievalGraph
from retrieval.retriever import CustomRetrievalQA

load_dotenv("src/config/secrets.env", override=True)
openai_key = os.getenv("OPENAI_API_KEY")


def load_vector_store():
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-ada-002", openai_api_key=openai_key
    )

    if settings.VECTOR_STORE.upper() == "FAISS" or not settings.VECTOR_STORE:
        faiss_adapter = FAISSAdapter(embedding_model=embedding_model)
        faiss_adapter.load("faiss.index")
        return faiss_adapter
    elif settings.VECTOR_STORE.upper() == "AZURE":
        azure_search = AzureSearch(
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
            index_name=os.getenv("AZURE_INDEX_NAME"),
            embedding_function=embedding_model.embed_query,
        )
        return AzureSearchAdapter(azure_search)
    else:
        raise ValueError(f"Invalid vector store: {settings.VECTOR_STORE}")


@cl.on_chat_start
def setup():
    vector_store = load_vector_store()
    llm = ChatOpenAI(temperature=0)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    qa_chain = CustomRetrievalQA(
        llm=llm,
        vector_store=vector_store,
        retrieval_method=settings.RETRIEVAL_METHOD,
        return_source_documents=True,
        memory=memory,
    )

    graph = RetrievalGraph(retriever_chain=qa_chain)
    cl.user_session.set("graph", graph)


@cl.on_message
async def handle_msg(msg: cl.Message):
    graph = cl.user_session.get("graph")
    response = graph.invoke(msg.content)

    if isinstance(response, dict):
        answer = response.get("answer", "")
        sources = response.get("source_documents", [])

        formatted_sources = ""
        if sources:
            formatted_sources = "\n\n**Sources used:**\n"
            for i, doc in enumerate(sources, start=1):
                source_name = doc.metadata.get("source", f"Document {i}")
                excerpt = doc.page_content.strip().replace("\n", " ")
                excerpt_preview = excerpt[:200] + ("..." if len(excerpt) > 200 else "")
                formatted_sources += f"**{source_name}**\n> {excerpt_preview}\n\n"

        await cl.Message(content=f"{answer}{formatted_sources}").send()
    else:
        await cl.Message(content=str(response)).send()
