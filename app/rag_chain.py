# app/rag_chain.py

import logging
from typing import List, Dict, Any, Optional

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from .config import OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE
from .retrieval import query_docs

logger = logging.getLogger(__name__)


# RAG prompt template
RAG_PROMPT_TEMPLATE = """You are a helpful AI assistant that answers questions based on the provided context from financial documents.

Use the following pieces of context to answer the question at the end. If you don't know the answer based on the context, just say that you don't know, don't try to make up an answer.

When answering:
1. Be precise and cite specific information from the context
2. If the context contains relevant page numbers or sections, mention them
3. If multiple sources provide relevant information, synthesize them coherently
4. Maintain a professional and informative tone

Context:
{context}

Question: {question}

Answer:"""


def format_docs(results: List[Dict[str, Any]]) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    Args:
        results: List of retrieval results from query_docs
    
    Returns:
        Formatted context string
    """
    if not results:
        return "No relevant context found."
    
    formatted_chunks = []
    for i, result in enumerate(results, 1):
        chunk_text = result.get("content", "")
        file_name = result.get("file_name", "Unknown")
        page_start = result.get("page_start", "N/A")
        page_end = result.get("page_end", "N/A")
        section_title = result.get("section_title", "")
        
        # Format each chunk with metadata
        chunk_header = f"[Source {i}: {file_name}"
        if page_start and page_end:
            if page_start == page_end:
                chunk_header += f", Page {page_start}"
            else:
                chunk_header += f", Pages {page_start}-{page_end}"
        if section_title:
            chunk_header += f", Section: {section_title}"
        chunk_header += "]"
        
        formatted_chunks.append(f"{chunk_header}\n{chunk_text}")
    
    return "\n\n---\n\n".join(formatted_chunks)


def create_rag_chain():
    """
    Create a LangChain RAG chain that retrieves documents and generates answers.
    
    Returns:
        A runnable RAG chain
    """
    # Initialize the LLM
    llm = ChatOpenAI(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=OPENAI_API_KEY,
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
    
    # Create the chain
    rag_chain = (
        {
            "context": lambda x: format_docs(x["retrieved_docs"]),
            "question": lambda x: x["question"]
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def answer_question(
    question: str,
    top_k: int = 5,
    file_name_filter: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Answer a question using RAG: retrieve relevant documents and generate an answer.
    
    Args:
        question: The user's question
        top_k: Number of documents to retrieve
        file_name_filter: Optional filter by file name
    
    Returns:
        Dictionary containing the answer, sources, and metadata
    """
    logger.info(f"Processing RAG question: '{question[:100]}...'")
    
    # Step 1: Retrieve relevant documents
    logger.debug(f"Retrieving top {top_k} documents...")
    retrieved_docs = query_docs(
        query=question,
        top_k=top_k,
        file_name_filter=file_name_filter,
    )
    
    if not retrieved_docs:
        logger.warning("No documents retrieved for the question")
        return {
            "answer": "I couldn't find any relevant information in the knowledge base to answer your question.",
            "sources": [],
            "question": question,
        }
    
    logger.debug(f"Retrieved {len(retrieved_docs)} documents")
    
    # Step 2: Generate answer using LLM
    logger.debug("Generating answer with LLM...")
    rag_chain = create_rag_chain()
    
    try:
        answer = rag_chain.invoke({
            "question": question,
            "retrieved_docs": retrieved_docs,
        })
        
        logger.info("Successfully generated answer")
        
        # Format sources for response
        sources = [
            {
                "file_name": doc.get("file_name"),
                "page_start": doc.get("page_start"),
                "page_end": doc.get("page_end"),
                "section_title": doc.get("section_title"),
                "score": doc.get("score"),
                "content_preview": doc.get("content", ""),  # Full content for UI display
            }
            for doc in retrieved_docs
        ]
        
        return {
            "answer": answer,
            "sources": sources,
            "question": question,
            "num_sources": len(sources),
        }
    
    except Exception as e:
        logger.error(f"Error generating answer: {str(e)}", exc_info=True)
        raise
