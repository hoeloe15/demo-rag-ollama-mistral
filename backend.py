import os
import warnings
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from pymongo import MongoClient
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.schema import Document
from cachetools import TTLCache

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)

# Configuration
local_path = "data/ISO 27001.pdf"
openai_api_key = os.getenv('OPENAI_API_KEY')
cosmos_connection_string = os.getenv('COSMOS_CONNECTION_STRING')
cosmos_db_name = os.getenv('COSMOS_DB_NAME')
cosmos_collection_name = os.getenv('COSMOS_COLLECTION_NAME')

# Validate environment variables
if not all([openai_api_key, cosmos_connection_string, cosmos_db_name, cosmos_collection_name]):
    raise ValueError("One or more required environment variables are not set")

# Initialize MongoDB client
client = MongoClient(cosmos_connection_string)
db = client[cosmos_db_name]
collection = db[cosmos_collection_name]

# Initialize OpenAI embeddings
embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

# Cache for storing chunks
chunks_cache = TTLCache(maxsize=100, ttl=300)

# Initialize global variables
chain = None

def load_chunks():
    if 'chunks' in chunks_cache:
        return chunks_cache['chunks']
    
    chunks = [Document(**chunk) for chunk in collection.find()]
    if not chunks:
        if local_path:
            loader = UnstructuredPDFLoader(file_path=local_path)
            data = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
            chunks = text_splitter.split_documents(data)
            collection.insert_many([chunk.dict() for chunk in chunks])
        else:
            raise FileNotFoundError("PDF file not found.")
    
    chunks_cache['chunks'] = chunks
    return chunks

def initialize():
    global chain
    try:
        print("Loading chunks...")
        chunks = load_chunks()
        
        print("Loading LLM model...")
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

        print("Setting up query prompt...")
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Je bent een AI language model assistent.. Je taak is om zo goed mogelijk de vragen van klanten te beantwoorden met informatie die je uit de toegevoegde data kan vinden in de vectordatabase.
            Je blijft altijd netjes en als je het niet kan vinden in de vectordatabase, geef je dat aan. 
            De originele vraag: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            collection, llm, prompt=QUERY_PROMPT
        )

        template = """Beantwoordt de vraag ALLEEN met de volgende context:
        {context}
        Vraag: {question}
        """

        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        print("Chain initialized successfully")
    except Exception as e:
        print(f"Initialization error: {e}")
        raise

# Initialize the model when the script starts
initialize()

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
        response = chain.invoke(question)
        print(f"Response: {response}")
        return jsonify({"response": response})
    except openai.error.RateLimitError as e:
        print(f"RateLimitError: {e}")
        return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred. Please try again later."}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
