import os
import warnings
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document, AIMessage
from cachetools import TTLCache
from typing import List
import logging
import openai
import time
import json

# Custom modules
from azure_retriever import AzureSearchRetriever
from initialization import initialize_system, load_chunks_from_pdf

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

# Check if pytesseract is available
try:
    import pytesseract
    pytesseract_available = True
except ImportError:
    pytesseract_available = False

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

if __name__ == '__main__':
    if os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
        global sequence
        sequence = initialize_system(openai_api_key, search_index_name, index_client, index_schema, search_client, local_path, pytesseract_available)
    app.run(port=5000, debug=True)
