import logging
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from langchain.schema import Document
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from cachetools import TTLCache
import json
import time

logger = logging.getLogger(__name__)

# Initialize the cache for storing chunks
chunks_cache = TTLCache(maxsize=100, ttl=300)

def load_chunks(search_client: SearchClient, local_path: str, pytesseract_available: bool):
    """Load chunks from cache or Azure Cognitive Search."""
    if 'chunks' in chunks_cache:
        logger.info("Loading chunks from cache...")
        return chunks_cache['chunks']

    chunks = []
    try:
        results = search_client.search(search_text="*", select=["id", "content", "embedding"])
        chunks = [Document(page_content=result["content"], metadata={"id": result["id"], "embedding": result["embedding"]}) for result in results]
        logger.info(f"Loaded {len(chunks)} chunks from Azure Cognitive Search")
    except Exception as e:
        logger.error(f"Error loading chunks from Azure Cognitive Search: {e}")
        raise

    chunks_cache['chunks'] = chunks
    return chunks

def load_chunks_from_pdf(local_path: str, search_client: SearchClient, pytesseract_available: bool):
    """Load chunks from a PDF file."""
    if not local_path:
        raise FileNotFoundError("PDF file not found.")

    strategy = "ocr_only" if pytesseract_available else "hi_res"
    loader = UnstructuredPDFLoader(file_path=local_path, strategy=strategy)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    for i, chunk in enumerate(chunks):
        embedding = chunk.metadata.get("embedding", [])
        doc = {
            "id": str(i),  # Ensure each document has a unique ID
            "content": chunk.page_content,
            "embedding": json.dumps(embedding)  # Ensure embedding is a string
        }
        search_client.upload_documents(documents=[doc])
    logger.info(f"Uploaded {len(chunks)} chunks from PDF to Azure Cognitive Search")
    return chunks

def initialize_system(openai_api_key, search_index_name, index_client, index_schema, search_client, local_path, pytesseract_available):
    """Initialize the system components."""
    retries = 3
    for attempt in range(retries):
        try:
            logger.info(f"Initialization attempt {attempt + 1}...")

            # Delete the existing index if it exists
            try:
                logger.info("Deleting existing index if it exists...")
                index_client.delete_index(search_index_name)
                logger.info(f"Deleted existing index: {search_index_name}")
            except Exception as e:
                logger.warning(f"No existing index to delete or error in deletion: {e}")

            # Create the index
            logger.info("Creating new index...")
            index_client.create_index(index_schema)
            logger.info("Index created successfully")

            logger.info("Initializing OpenAI embeddings...")
            embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
            logger.info("OpenAI embeddings initialized")

            logger.info("Loading chunks from Azure Cognitive Search...")
            chunks = load_chunks(search_client, local_path, pytesseract_available)

            if not chunks:
                logger.info("No chunks found in Azure Cognitive Search, loading from PDF...")
                chunks = load_chunks_from_pdf(local_path, search_client, pytesseract_available)

            logger.info("Loading LLM model...")
            llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
            logger.info("LLM model loaded")

            logger.info("Setting up query prompt...")
            QUERY_PROMPT = ChatPromptTemplate.from_template(
                """You are an AI language model assistant. Your task is to answer customer questions as best as you can with information that you can find in the added data in the vector database.
                You always remain polite and if you can't find it in the vector database, you indicate that.
                The original question: {question}
                Context: {context}"""
            )
            logger.info("Query prompt set up")

            sequence = QUERY_PROMPT | llm
            logger.info("Sequence initialized successfully")
            return sequence
        except Exception as e:
            logger.error(f"Initialization error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                logger.info("Retrying...")
                time.sleep(5)
            else:
                logger.error("Failed to initialize after multiple attempts")
                raise
