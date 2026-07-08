import os
import asyncio
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List
from langsmith import traceable

from dotenv import find_dotenv, load_dotenv
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
)
import xml.etree.ElementTree as ET

from langchain_community.retrievers import BM25Retriever
from langchain_cohere import CohereRerank
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient


load_dotenv(find_dotenv(), override=True)


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_knowledge_base_dir() -> Path:
    configured_path = os.getenv("KNOWLEDGE_BASE_DIR")
    if configured_path:
        return Path(configured_path)
    return get_project_root() / "data_base" / "knowledge_db"


def load_documents() -> List[Document]:
    folder_path = get_knowledge_base_dir()
    if not folder_path.is_dir():
        raise RuntimeError(f"Knowledge base directory not found: {folder_path}")

    loaders = []
    xml_files = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            suffix = file_path.rsplit(".", 1)[-1].lower()
            if suffix == "pdf":
                loaders.append(PyMuPDFLoader(file_path))
            elif suffix == "md":
                loaders.append(UnstructuredMarkdownLoader(file_path))
            elif suffix == "xml":
                xml_files.append(file_path)


    documents: List[Document] = []
    for loader in loaders:
        documents.extend(loader.load())

    for file_path in xml_files:
        documents.extend(load_xml_documents(file_path))

    if not documents:
        raise RuntimeError(
            f"No PDF, Markdown, or XML documents found in knowledge base: {folder_path}"
        )

    return documents


def load_xml_documents(file_path: str) -> List[Document]:
    try:
        root = ET.parse(file_path).getroot()
    except ET.ParseError:
        return []

    source = root.attrib.get("source", "")
    url = root.attrib.get("url", "")
    document_id = root.attrib.get("id", "")
    focus = root.findtext("Focus", default="").strip()

    documents: List[Document] = []

    for qa_pair in root.findall("./QAPairs/QAPair"):
        question_node = qa_pair.find("Question")
        answer_node = qa_pair.find("Answer")

        question = (
            "".join(question_node.itertext()).strip()
            if question_node is not None
            else ""
        )
        answer = (
            "".join(answer_node.itertext()).strip()
            if answer_node is not None
            else ""
        )

        if not question or not answer:
            continue

        qid = question_node.attrib.get("qid", "") if question_node is not None else ""
        qtype = question_node.attrib.get("qtype", "") if question_node is not None else ""

        page_content = "\n".join(
            item
            for item in [
                f"Focus: {focus}" if focus else "",
                f"Question: {question}",
                f"Answer: {answer}",
            ]
            if item
        )

        documents.append(
            Document(
                page_content=page_content,
                metadata={
                    "source": file_path,
                    "dataset_source": source,
                    "url": url,
                    "document_id": document_id,
                    "focus": focus,
                    "qid": qid,
                    "qtype": qtype,
                },
            )
        )

    return documents


def split_documents(documents: List[Document]) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    return text_splitter.split_documents(documents)


def combine_docs(docs: Iterable[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def merge_and_deduplicate(*doc_lists: List[Document]) -> List[Document]:
    seen = set()
    unique_docs = []

    for docs in doc_lists:
        for doc in docs:
            key = doc.page_content.strip()
            if key not in seen:
                seen.add(key)
                unique_docs.append(doc)

    return unique_docs


@lru_cache(maxsize=1)
def get_embedding_model() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model="text-embedding-3-small")


@lru_cache(maxsize=1)
def get_milvus_client() -> MilvusClient:
    return MilvusClient(uri=os.getenv("MILVUS_URI", "http://milvus:19530"))


def batched(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

async def embed_one_batch(batch, embedding_model, sem, retries = 3):
    for attempt in range(retries):
        try:
            async with sem:
                vectors = await embedding_model.aembed_documents(batch) 
                return vectors
        except Exception as e:
            if attempt == retries - 1:
                raise e
            wait_time = 2 ** attempt
            await asyncio.sleep(wait_time)


async def embed_all(
        texts, 
        embedding_model, 
        batch_size: int, 
        max_concurrency: int,
        ):
    sem = asyncio.Semaphore(max_concurrency)

    tasks = []  

    for batch in batched(texts, batch_size):
        task = embed_one_batch(batch, embedding_model, sem)
        tasks.append(task)

    result = await asyncio.gather(*tasks)

    all_vectors = []
    for batch_vectors in result:
        all_vectors.extend(batch_vectors)

    return all_vectors


def build_milvus_collection(
    batch_size: int = 64,
    max_concurrency: int = 3,
    collection_name: str = "RAG_collection",
    force_rebuild: bool = False,
) -> None:
    client = get_milvus_client()
    embedding_model = get_embedding_model()
    split_docs = split_documents(load_documents())
    chunk_texts = [doc.page_content for doc in split_docs]

    if client.has_collection(collection_name=collection_name):
        if not force_rebuild:
            return
        client.drop_collection(collection_name=collection_name)

    dimension = len(embedding_model.embed_query("test"))
    client.create_collection(
        collection_name=collection_name,
        dimension=dimension,
    )

    # vectors = embedding_model.embed_documents(chunk_texts)
    vectors = asyncio.run(
            embed_all(
            texts=chunk_texts,
            embedding_model=embedding_model, 
            batch_size=batch_size, 
            max_concurrency=max_concurrency
            )
        )

    data = [
        {
            "id": idx,
            "vector": vectors[idx],
            "text": chunk_texts[idx],
            "subject": "agent",
        }
        for idx in range(len(vectors))
    ]

    milvus_batch_size = 100
    for start in range(0, len(data), milvus_batch_size):
        client.insert(
            collection_name=collection_name,
            data=data[start : start + milvus_batch_size],
        )


def create_vector_retriever(
    collection_name: str = "RAG_collection",
    top_k: int = 10,
):
    client = get_milvus_client()
    embedding_model = get_embedding_model()

    # if not client.has_collection(collection_name=collection_name):
    #     build_milvus_collection(collection_name=collection_name)

    if not client.has_collection(collection_name=collection_name):
        raise RuntimeError(
            f"Collection '{collection_name}' not found. Please call /index first."
        )
    
    def retrieve(query: str) -> List[Document]:
        query_vector = embedding_model.embed_query(query)
        results = client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text", "subject"],
        )

        docs = []
        for result in results[0]:
            entity = result["entity"]
            docs.append(
                Document(
                    page_content=entity.get("text", ""),
                    metadata={
                        "subject": entity.get("subject", ""),
                        "retriever": "milvus",
                        "score": result.get("distance"),
                    },
                )
            )
        return docs

    return retrieve


def create_bm25_retriever(top_k: int = 10) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(split_documents(load_documents()))
    retriever.k = top_k
    return retriever


async def rerank_one_query(query, candidate_docs, reranker, sem):
    async with sem:
        return await reranker.acompress_documents(
            documents=candidate_docs,
            query=query,
        )

async def rerank_all_queries(queries_and_docs, reranker, max_concurrency) -> List[dict[str, object]]:
    sem = asyncio.Semaphore(max_concurrency)

    tasks= []
    
    for query , candidate_docs in queries_and_docs:
        task = rerank_one_query(
            query=query,
            candidate_docs=candidate_docs,
            reranker=reranker,
            sem=sem,
        )
        tasks.append(task)
    
    raw_results = await asyncio.gather(*tasks, return_exceptions=True)

    results = []

    for (query, _), item in zip(queries_and_docs, raw_results):
        if isinstance(item, Exception):
            results.append(
                {
                    "query": query,
                    "docs": None,
                    "error": str(item),
                }
            )
        else:
            results.append(
                {
                    "query": query,
                    "docs": item,
                    "error": None,
                }
            )

    return results


    
async def rerank_documents_async(query, candidate_docs):
    reranker = CohereRerank(
        model="rerank-v3.5",
        top_n=8,
    )

    sem = asyncio.Semaphore(1)
    results = await rerank_one_query(query, candidate_docs, reranker, sem)
    
    return results

# def rerank_documents(query: str, candidate_docs: List[Document]) -> List[Document]:
#     reranker = CohereRerank(
#         model="rerank-v3.5",
#         top_n=8,
#     )
#     return reranker.compress_documents(
#         documents=candidate_docs,
#         query=query,
#     )


def collect_candidate_docs(query, vector_retriever, bm25_retriever): 
    vector_docs = vector_retriever(query)
    bm25_docs = bm25_retriever.invoke(query)

    for doc in bm25_docs:
        doc.metadata["retriever"] = "bm25"

    candidate_docs = merge_and_deduplicate(vector_docs, bm25_docs)
    return candidate_docs

async def create_hybrid_retriever(collection_name: str = "RAG_collection"):
    vector_retriever = create_vector_retriever(collection_name=collection_name)
    bm25_retriever = create_bm25_retriever()
    async def retrieve(query: str) -> List[Document]:
        vector_docs = vector_retriever(query)
        bm25_docs = bm25_retriever.invoke(query)
        for doc in bm25_docs:
            doc.metadata["retriever"] = "bm25"

        candidate_docs = merge_and_deduplicate(vector_docs, bm25_docs)
        return await rerank_documents_async(query, candidate_docs)

    return RunnableLambda(retrieve)


async def create_rag_chain(model: str, collection_name: str = "RAG_collection"):
    retriever = await create_hybrid_retriever(collection_name=collection_name)

    llm = ChatOpenAI(
        model=model,
        temperature=0,
    )

    condense_question_system_template = (
        "Use the chat history and the user's latest question to rewrite the latest "
        "question into a standalone question that can be understood without prior context. "
        "Do not answer the question. Return only the rewritten question."
    )
    condense_question_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    retrieve_docs = RunnableBranch(
        (
            lambda x: not x.get("chat_history", False),
            RunnableLambda(lambda x: x["input"]) | retriever,
        ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the retrieved context snippets to answer the question. "
        "If you do not know the answer, say that you do not know."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )

    qa_chain = (
        RunnablePassthrough.assign(context=lambda x: combine_docs(x["context"]))
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    return (
        RunnablePassthrough.assign(context=retrieve_docs)
        .assign(answer=qa_chain)
    )


async def search_local_knowledge(
    query: str,
    model: str,
    chat_history=None,
    collection_name: str = "RAG_collection",
) -> str:
    rag_chain = await create_rag_chain(
        model, 
        collection_name=collection_name,
        )
        
    result = await rag_chain.ainvoke(
        {
            "input": query,
            "chat_history": chat_history or [],
        }
    )
    return result["answer"]

@traceable(name="Generate Batch Answer", run_type="chain")
async def answer_one_query(query: str, docs: list[Document], llm) -> str:
    context = combine_docs(docs)

    prompt = f"""
        You are a question-answering assistant.
        Answer the question using the context below.
        If you do not know the answer, say that you do not know.

        Context:
        {context}

        Question:
        {query}
        """

    response = await llm.ainvoke(prompt)
    return response.content


@traceable(name="Week4 Batch Local RAG", run_type="chain")
async def search_local_knowledge_batch(
    queries: list[str],
    model: str,
    collection_name: str = "RAG_collection",
    rerank_max_concurrency: int = 3,
) -> list[dict[str, str | None]]:
    # 1. Prepare retrievers.
    vector_retriever = create_vector_retriever(collection_name=collection_name)
    bm25_retriever = create_bm25_retriever()

    # 2. Collect candidate docs for each query.
    results = [None] * len(queries)
    queries_and_docs = []
    rerank_meta = []

    for idx, query in enumerate(queries):
        try:
            candidate_docs = collect_candidate_docs(
                query=query,
                vector_retriever=vector_retriever,
                bm25_retriever=bm25_retriever,
            )
            queries_and_docs.append((query, candidate_docs))
            rerank_meta.append((idx, query))
        except Exception as e:
            results[idx] = {
                "query": query,
                "answer": None,
                "error": str(e),
            }

    # 3. Call rerank_all_queries.
    reranker = CohereRerank(
        model="rerank-v3.5",
        top_n=8,
    )

    rerank_results = await rerank_all_queries(
        queries_and_docs=queries_and_docs, 
        reranker=reranker,
        max_concurrency=rerank_max_concurrency,
        )

    # 4. Generate answers from reranked docs.

    llm = ChatOpenAI(model=model, temperature=0)

    answer_tasks = []
    answer_meta = []
    for (idx, _ ), item in zip(rerank_meta, rerank_results):
        if item["error"] is not None:
            results[idx] = {
                "query": item["query"],
                "answer": None,
                "error": item["error"],
            }
        else:
            answer_tasks.append(
                answer_one_query(item["query"], item["docs"], llm)
            )
            answer_meta.append((idx, item["query"]))

    raw_answers = await asyncio.gather(*answer_tasks, return_exceptions=True)

    for (idx, query), answer in zip(answer_meta, raw_answers):
        if isinstance(answer, Exception):
            results[idx] = {
                "query": query,
                "answer": None,
                "error": str(answer),
            }
        else:
            results[idx] = {
                "query": query,
                "answer": answer,
                "error": None,
            }

    return results
