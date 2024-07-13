import os
import pickle
import warnings
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)

local_path = "data/ISO 27001.pdf"
chunks_file = "data/chunks.pkl"
vector_db_path = "data/vector_db"
openai_api_key = os.getenv('OPENAI_API_KEY')

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")

embedding_function = OpenAIEmbeddings(model="text-embedding-ada-002", api_key=openai_api_key)

# Initialize global variables
chain = None

def load_chunks(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_chunks(chunks, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(chunks, f)

def initialize():
    global chain  # Ensure chain is global
    try:
        print("Loading chunks...")
        chunks = load_chunks(chunks_file)
        if chunks is None:
            if local_path:
                print("Loading PDF...")
                loader = UnstructuredPDFLoader(file_path=local_path)
                data = loader.load()
                print("Splitting text into chunks...")
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
                chunks = text_splitter.split_documents(data)
                save_chunks(chunks, chunks_file)
            else:
                raise FileNotFoundError("PDF file not found.")

        print("Loading LLM model...")
        llm = ChatOpenAI(api_key=openai_api_key, model="gpt-3.5-turbo")

        global vector_db
        if os.path.exists(vector_db_path):
            print("Loading existing vector database...")
            vector_db = Chroma(persist_directory=vector_db_path, embedding_function=embedding_function)
        else:
            print("Creating new vector database...")
            vector_db = Chroma.from_documents(
                documents=chunks,
                embedding=embedding_function,
                collection_name="local-ai",
                persist_directory=vector_db_path,
            )

        print("Setting up query prompt...")
        QUERY_PROMPT = PromptTemplate(
            input_variables=["question"],
            template="""Je bent een AI language model assistent.. Je taak is om zo goed mogelijk de vragen van klanten te beantwoorden met informatie die je uit de toegevoegde data kan vinden in de vectordatabase.
            Je blijft altijd netjes en als je het niet kan vinden in de vectordatabase, geef je dat aan. 
            De originele vraag: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
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
        print("Chain initialized:", chain is not None)
        print("Initialization successful")
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
    response = chain.invoke(question)
    print(f"Response: {response}")
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(port=5000, debug=True)
