import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import logging
from typing import Iterator
from langchain.callbacks.base import BaseCallbackHandler
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


class StreamHandler(BaseCallbackHandler):
    """Custom callback handler for streaming responses"""
    def __init__(self):
        self.tokens = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)
        self.tokens.append(token)

def create_vector_db(pdf_path: str, persist_directory: str = "db"):
    """Create a vector database from the PDF document"""
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=100,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    vectordb = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    return vectordb

def setup_qa_chain(vectordb):
    """Set up the QA chain with custom prompt"""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.FileHandler('qa_system.log'), logging.StreamHandler()]
    )
    logger = logging.getLogger(__name__)

    # Initialize Llama model with streaming
    llm = Ollama(
        model="gemma2:2b",
        temperature=0.5,
        callbacks=[StreamHandler()]
    )
    # Create custom prompt
    prompt_template = """You are a helpful assistant.  working for a company , your job is to answer questions about the company . Use only the following context to answer the question. Do not use any external knowledge or make assumptions.
    If you cannot find the answer in the context, say "I cannot answer this question based on the provided document."

    Context: {context}

    Question: {question}

    Answer: """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # Create chain
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return chain

def main():
    try:
        logger = logging.getLogger(__name__)
        logger.info("Starting QA system")

        pdf_path = "data.pdf"
        persist_dir = "pdf_db"

        # Create or load vector database
        logger.info("Initializing components...")
        if not os.path.exists(persist_dir):
            logger.info("Creating new vector database...")
            vectordb = create_vector_db(pdf_path, persist_dir)
        else:
            logger.info("Loading existing vector database...")
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)

        # Pre-load the model and setup chain
        logger.info("Loading language model and setting up QA chain...")
        qa_chain = setup_qa_chain(vectordb)
        logger.info("System ready for questions!")

        # Interactive question answering loop
        print("\nWelcome to the PDF QA System! Type 'quit' to exit.")
        while True:
            try:
                question = input("\nEnter your question: ")
                if question.lower() == 'quit':
                    logger.info("Exiting QA system")
                    break

                logger.info(f"Processing question: {question}")
                result = qa_chain({"query": question})
                print("\nSources: ")
                for doc in result["source_documents"]:
                    logger.debug(f"Source document - Page {doc.metadata['page']}")
                    print(f"- Page {doc.metadata['page']}: {doc.page_content[:200]}...")

            except KeyboardInterrupt:
                logger.info("User interrupted the process")
                break
            except Exception as e:
                logger.error(f"Error processing question: {str(e)}", exc_info=True)
                print(f"An error occurred while processing your question. Please try again.")
                continue

    except Exception as e:
        logger.error(f"An error occurred during initialization: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()