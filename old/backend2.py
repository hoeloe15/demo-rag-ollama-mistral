import os
import warnings
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document, AIMessage
from cachetools import TTLCache
from typing import List
import logging
import openai
import time
import json

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Flask setup
app = Flask(__name__)

# Configuration
local_path = "data/test.pdf"
openai_api_key = os.getenv('OPENAI_API_KEY')
search_service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')

def check_env_variables():
    """Check that all required environment variables are set."""
    required_vars = [openai_api_key, search_service_name, search_admin_key, search_index_name]
    if not all(required_vars):
        raise ValueError("One or more required environment variables are not set")

check_env_variables()

# Initialize Azure Cognitive Search client
search_endpoint = f"https://{search_service_name}.search.windows.net"
credential = AzureKeyCredential(search_admin_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=search_index_name, credential=credential)
index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)

# Define the index schema with searchable fields and vector field properties
index_schema = SearchIndex(
    name=search_index_name,
    fields=[
        SimpleField(name="id", type=SearchFieldDataType.String, key=True),
        SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="embedding", type=SearchFieldDataType.String)
    ]
)

# Initialize the cache for storing chunks
chunks_cache = TTLCache(maxsize=100, ttl=300)

# Check if pytesseract is available
try:
    import pytesseract
    pytesseract_available = True
except ImportError:
    pytesseract_available = False

class AzureSearchRetriever:
    """Custom Azure Search Retriever."""
    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    def get_relevant_documents(self, query: str, max_documents: int = 5) -> List[Document]:
        logger.info(f"Retrieving relevant documents for query: {query}")
        results = self.search_client.search(search_text=query, select=["id", "content", "embedding"], top=max_documents)
        documents = [Document(page_content=result["content"], metadata={"id": result["id"], "embedding": result["embedding"]}) for result in results]
        logger.info(f"Retrieved {len(documents)} documents")
        return documents

def load_chunks():
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

def load_chunks_from_pdf():
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

def truncate_context(context, max_tokens):
    """Truncate context to fit within the token limit."""
    tokens = context.split()
    if len(tokens) <= max_tokens:
        return context
    truncated_context = ' '.join(tokens[:max_tokens])
    logger.info(f"Context truncated to {max_tokens} tokens")
    return truncated_context

def response_to_dict(response):
    """Convert the AIMessage or other OpenAI response objects to a dictionary."""
    if isinstance(response, list):
        return [msg_to_dict(msg) for msg in response]
    return msg_to_dict(response)

def msg_to_dict(msg):
    """Convert AIMessage to a dictionary."""
    if isinstance(msg, AIMessage):
        return {
            "content": msg.content,
            "additional_kwargs": msg.additional_kwargs,
        }
    return msg

initialized = False

def initialize():
    """Initialize the system components."""
    global initialized
    if initialized:
        logger.info("Already initialized, skipping re-initialization.")
        return
    initialized = True

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
            chunks = load_chunks()

            if not chunks:
                logger.info("No chunks found in Azure Cognitive Search, loading from PDF...")
                chunks = load_chunks_from_pdf()

            logger.info("Loading LLM model...")
            llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
            logger.info("LLM model loaded")

            logger.info("Setting up query prompt...")
            QUERY_PROMPT = ChatPromptTemplate.from_template(
                """Je bent een AI language model assistent. Je taak is om zo goed mogelijk de vragen van klanten te beantwoorden met informatie die je uit de toegevoegde data kan vinden in de vectordatabase.
                Je blijft altijd netjes en als je het niet kan vinden in de vectordatabase, geef je dat aan.
                De originele vraag: {question}
                Context: {context}"""
            )
            logger.info("Query prompt set up")

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Server is running"}), 200

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    logger.info(f"Received question: {question}")
    try:
        # Retrieve relevant documents
        retriever = AzureSearchRetriever(search_client=search_client)
        documents = retriever.get_relevant_documents(question)
        # Concatenate context
        context = " ".join([doc.page_content for doc in documents])
        logger.info("Context: %s", context)
        # Truncate context to fit within the token limit
        max_tokens = 16000  # slightly less than model's limit to accommodate other tokens
        truncated_context = truncate_context(context, max_tokens)
        logger.info("Truncated context: %s", truncated_context)
        # Prepare the input
        input_data = {"context": truncated_context, "question": question}
        # Invoke the sequence
        response = sequence.invoke(input_data)
        
        # Convert response to a JSON serializable format
        response_content = response.content if hasattr(response, 'content') else None
        response_metadata = response.response_metadata if hasattr(response, 'response_metadata') else None
        response_id = response.id if hasattr(response, 'id') else None
        usage_metadata = response.usage_metadata if hasattr(response, 'usage_metadata') else None

        response_dict = {
            "content": response_content,
            "response_metadata": response_metadata,
            "id": response_id,
            "usage_metadata": usage_metadata,
        }
        
        logger.info("Response: %s", response_dict['content'])
        return jsonify({"response": response_dict["content"]})
    except openai.RateLimitError as e:
        logger.error(f"RateLimitError: {e}")
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    except openai.OpenAIError as e:
        logger.error(f"OpenAIError: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        logger.error(f"Error: {e}")
        return jsonify({"error": "An error occurred. Please try again later."}), 500


# Initialize the model when the script starts
if __name__ == '__main__':
    initialize()
    app.run(port=5000, debug=True)
