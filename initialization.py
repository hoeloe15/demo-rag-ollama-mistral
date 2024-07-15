import logging
import time
from azure.search.documents import SearchClient, SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from cachetools import TTLCache

logger = logging.getLogger(__name__)

# Initialize the cache for storing chunks
chunks_cache = TTLCache(maxsize=100, ttl=300)

def initialize_index(search_index_name, index_client, index_schema):
    """Initialize the Azure Cognitive Search index."""
    try:
        logger.info("Deleting existing index if it exists...")
        index_client.delete_index(search_index_name)
        logger.info(f"Deleted existing index: {search_index_name}")
    except Exception as e:
        logger.warning(f"No existing index to delete or error in deletion: {e}")

    logger.info("Creating new index...")
    index_client.create_index(index_schema)
    logger.info("Index created successfully")

def initialize_embeddings(openai_api_key):
    """Initialize the OpenAI embeddings."""
    logger.info("Initializing OpenAI embeddings...")
    embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
    logger.info("OpenAI embeddings initialized")
    return embedding_function

def load_llm_model(openai_api_key):
    """Load the LLM model."""
    logger.info("Loading LLM model...")
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
    logger.info("LLM model loaded")
    return llm

def setup_query_prompt():
    """Setup the query prompt template."""
    logger.info("Setting up query prompt...")
    QUERY_PROMPT = ChatPromptTemplate.from_template(
        """Je bent een AI language model assistent. Je taak is om zo goed mogelijk de vragen van klanten te beantwoorden met informatie die je uit de toegevoegde data kan vinden in de vectordatabase.
        Je blijft altijd netjes en als je het niet kan vinden in de vectordatabase, geef je dat aan.
        De originele vraag: {question}
        Context: {context}"""
    )
    logger.info("Query prompt set up")
    return QUERY_PROMPT

def load_chunks(search_client):
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

def load_chunks_from_pdf(local_path, pytesseract_available, search_client):
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

            # Initialize the index
            initialize_index(search_index_name, index_client, index_schema)

            # Initialize embeddings
            initialize_embeddings(openai_api_key)

            # Load chunks
            chunks = load_chunks(search_client)
            if not chunks:
                logger.info("No chunks found in Azure Cognitive Search, loading from PDF...")
                chunks = load_chunks_from_pdf(local_path, pytesseract_available, search_client)

            # Load LLM model
            llm = load_llm_model(openai_api_key)

            # Setup query prompt
            QUERY_PROMPT = setup_query_prompt()

            global sequence
            sequence = QUERY_PROMPT | llm
            logger.info("Sequence initialized successfully")
            return
        except Exception as e:
            logger.error(f"Initialization error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                logger.info("Retrying...")
                time.sleep(5)
            else:
                logger.error("Failed to initialize after multiple attempts")
                raise
