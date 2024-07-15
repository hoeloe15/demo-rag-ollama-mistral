import os
import warnings
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import SearchIndex, SimpleField, SearchFieldDataType, SearchableField
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from cachetools import TTLCache
import time
from typing import List
import openai

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)

# Configuration
local_path = "data/test.pdf"
openai_api_key = os.getenv('OPENAI_API_KEY')
search_service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')

# Validate environment variables
if not all([openai_api_key, search_service_name, search_admin_key, search_index_name]):
    raise ValueError("One or more required environment variables are not set")

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
        SearchableField(name="content", type=SearchFieldDataType.String, searchable=True)
    ]
)

# Initialize the cache for storing chunks
chunks_cache = TTLCache(maxsize=100, ttl=300)

# Custom Azure Search Retriever
class AzureSearchRetriever:
    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    def get_relevant_documents(self, query: str, max_documents: int = 5) -> List[Document]:
        results = self.search_client.search(search_text=query, select=["id", "content"], top=max_documents)
        documents = [Document(page_content=result["content"], metadata={"id": result["id"]}) for result in results]
        return documents

azure_retriever = AzureSearchRetriever(search_client=search_client)

def load_chunks_from_pdf():
    if not local_path:
        raise FileNotFoundError("PDF file not found.")
    loader = UnstructuredPDFLoader(file_path=local_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    for i, chunk in enumerate(chunks):
        doc = {"id": str(i), "content": chunk.page_content}
        search_client.upload_documents(documents=[doc])
    return chunks

def initialize():
    try:
        index_client.get_index(search_index_name)
    except:
        index_client.create_index(index_schema)
    
    chunks = load_chunks_from_pdf()
    return chunks

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

    try:
        documents = azure_retriever.get_relevant_documents(question)
        context = " ".join([doc.page_content for doc in documents])
        prompt = f"Context: {context}\n\nQuestion: {question}"
        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=150
        )
        return jsonify({"response": response.choices[0].text.strip()})
    except openai.error.RateLimitError as e:
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    initialize()
    app.run(port=5000, debug=True)
