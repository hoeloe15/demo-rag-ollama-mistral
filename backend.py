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
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.schema import Document
from cachetools import TTLCache
import time
from typing import List
import openai
import json

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)

# Configuration
local_path = "data/ISO 27001.pdf"
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
        SearchableField(name="content", type=SearchFieldDataType.String, searchable=True),
        SimpleField(name="embedding", type=SearchFieldDataType.String)
    ]
)

# Initialize the cache for storing chunks
chunks_cache = TTLCache(maxsize=100, ttl=300)

initialized = False

def initialize():
    global initialized
    if initialized:
        print("Already initialized, skipping re-initialization.")
        return
    initialized = True

    retries = 3
    for attempt in range(retries):
        try:
            print(f"Initialization attempt {attempt + 1}...")
            
            # Delete the existing index if it exists
            try:
                print("Deleting existing index if it exists...")
                index_client.delete_index(search_index_name)
                print(f"Deleted existing index: {search_index_name}")
            except Exception as e:
                print(f"No existing index to delete or error in deletion: {e}")

            # Create the index
            print("Creating new index...")
            index_client.create_index(index_schema)
            print("Index created successfully")

            print("Initializing OpenAI embeddings...")
            embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)
            print("OpenAI embeddings initialized")

            print("Loading chunks from Azure Cognitive Search...")
            chunks = load_chunks()

            if not chunks:
                print("No chunks found in Azure Cognitive Search, loading from PDF...")
                chunks = load_chunks_from_pdf()

            print("Loading LLM model...")
            llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")
            print("LLM model loaded")

            print("Setting up query prompt...")
            QUERY_PROMPT = PromptTemplate(
                input_variables=["question"],
                template="""Je bent een AI language model assistent. Je taak is om zo goed mogelijk de vragen van klanten te beantwoorden met informatie die je uit de toegevoegde data kan vinden in de vectordatabase.
                Je blijft altijd netjes en als je het niet kan vinden in de vectordatabase, geef je dat aan. 
                De originele vraag: {question}""",
            )
            print("Query prompt set up")

            azure_retriever = AzureSearchRetriever(search_client=search_client)

            template = """Beantwoordt de vraag ALLEEN met de volgende context:
            {context}
            Vraag: {question}
            """

            prompt = ChatPromptTemplate.from_template(template)
            print("Prompt template created")

            global chain
            chain = (
                {"context": azure_retriever.get_relevant_documents, "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )
            print("Chain initialized successfully")
            return
        except Exception as e:
            print(f"Initialization error on attempt {attempt + 1}: {e}")
            if attempt < retries - 1:
                print("Retrying...")
                time.sleep(5)
            else:
                print("Failed to initialize after multiple attempts")
                raise

def load_chunks():
    if 'chunks' in chunks_cache:
        print("Loading chunks from cache...")
        return chunks_cache['chunks']
    
    chunks = []
    try:
        results = search_client.search(search_text="*", select=["id", "content", "embedding"])
        for result in results:
            chunks.append(Document(page_content=result["content"], metadata={"id": result["id"], "embedding": result["embedding"]}))
        print(f"Loaded {len(chunks)} chunks from Azure Cognitive Search")
    except Exception as e:
        print(f"Error loading chunks from Azure Cognitive Search: {e}")
        raise

    chunks_cache['chunks'] = chunks
    return chunks

def load_chunks_from_pdf():
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
    print(f"Uploaded {len(chunks)} chunks from PDF to Azure Cognitive Search")
    return chunks

# Check if pytesseract is available
try:
    import pytesseract
    pytesseract_available = True
except ImportError:
    pytesseract_available = False

# Initialize the model when the script starts
if __name__ == '__main__':
    initialize()

    app.run(port=5000, debug=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({"error": "No question provided"}), 400

    print(f"Received question: {question}")
    try:
        # Retrieve relevant documents
        documents = chain.get_input_steps()["context"].invoke(question)
        # Concatenate context
        context = " ".join([doc.page_content for doc in documents])
        # Truncate context to fit within the token limit
        max_tokens = 16000  # slightly less than model's limit to accommodate other tokens
        truncated_context = truncate_context(context, max_tokens)
        # Prepare the input
        input_data = {"context": truncated_context, "question": question}
        # Invoke the chain
        response = chain.invoke(input_data)
        response_dict = response_to_dict(response)
        print(f"Response: {response_dict}")
        return jsonify({"response": response_dict})
    except openai.error.RateLimitError as e:
        print(f"RateLimitError: {e}")
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    except openai.error.OpenAIError as e:
        print(f"OpenAIError: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred. Please try again later."}), 500

def truncate_context(context, max_tokens):
    """Truncate context to fit within the token limit."""
    tokens = context.split()
    if len(tokens) <= max_tokens:
        return context
    truncated_context = ' '.join(tokens[:max_tokens])
    print(f"Context truncated to {max_tokens} tokens")
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

