import os
import pickle
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain.retrievers.multi_query import MultiQueryRetriever

local_path = "data/ISO 27001.pdf"
chunks_file = "data/chunks.pkl"
vector_db_path = "data/vector_db"
embedding_function = OllamaEmbeddings(model="nomic-embed-text", show_progress=True)

# Initialize global variables
chain = None

def load_chunks(file_path):
    print(f"Loading chunks from {file_path}...")
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return None

def save_chunks(chunks, file_path):
    print(f"Saving chunks to {file_path}...")
    with open(file_path, 'wb') as f:
        pickle.dump(chunks, f)

def initialize():
    global chain  # Ensure chain is global
    try:
        print("Starting initialization...")
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

        local_model = "mistral"
        print("Loading LLM model...")
        llm = ChatOllama(model=local_model)

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
            template="""You are an AI language model assistant. Your task is to generate three
            different versions of the given user question to retrieve relevant documents from
            a vector database. By generating multiple perspectives on the user question, your
            goal is to help the user overcome some of the limitations of the distance-based
            similarity search. Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        retriever = MultiQueryRetriever.from_llm(
            vector_db.as_retriever(), llm, prompt=QUERY_PROMPT
        )

        template = """Answer the question based ONLY on the following context:
        {context}
        Question: {question}
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
