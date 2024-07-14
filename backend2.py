import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from cachetools import TTLCache
from typing import List
import openai

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)

# Configuration
openai_api_key = os.getenv('OPENAI_API_KEY')
search_service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')

# Validate environment variables
def validate_env_vars():
    if not all([openai_api_key, search_service_name, search_admin_key, search_index_name]):
        raise ValueError("One or more required environment variables are not set")

validate_env_vars()

# Initialize Azure Cognitive Search client
search_endpoint = f"https://{search_service_name}.search.windows.net"
credential = AzureKeyCredential(search_admin_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=search_index_name, credential=credential)

# Cache for storing chunks
chunks_cache = TTLCache(maxsize=100, ttl=300)

# Custom Azure Search Retriever
class AzureSearchRetriever:
    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    def get_relevant_documents(self, query: str) -> List[Document]:
        results = self.search_client.search(search_text=query)
        documents = [Document(page_content=result["content"]) for result in results]
        return documents

# Initialize OpenAI embeddings
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

def load_chunks():
    if 'chunks' in chunks_cache:
        return chunks_cache['chunks']

    chunks = []
    results = search_client.search(search_text="*")
    for result in results:
        chunks.append(Document(page_content=result["content"]))
    
    chunks_cache['chunks'] = chunks
    return chunks

def initialize_chain():
    global chain
    chunks = load_chunks()
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

    QUERY_PROMPT = PromptTemplate(
        input_variables=["question"],
        template="""You are an AI assistant. Answer the following question based on the provided context:
        Question: {question}"""
    )

    azure_retriever = AzureSearchRetriever(search_client=search_client)

    chain = lambda question: llm(QUERY_PROMPT.format(question=question))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        response = chain(question)
        return jsonify({"response": response})
    except openai.error.InvalidRequestError as e:
        return jsonify({"error": str(e)}), 400
    except openai.error.RateLimitError as e:
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    except Exception as e:
        return jsonify({"error": "An error occurred. Please try again later."}), 500

if __name__ == '__main__':
    initialize_chain()  # Initialize the chain
    app.run(port=5000, debug=True)
