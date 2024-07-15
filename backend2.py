import os
import warnings
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
import openai
from openai import OpenAI

import json

warnings.filterwarnings("ignore", category=FutureWarning)

# Load environment variables from .env file
load_dotenv()

# Flask setup
app = Flask(__name__)

# Configuration
openai_api_key = os.getenv('OPENAI_API_KEY')
search_service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
search_admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
search_index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')
client = OpenAI(api_key=openai_api_key)

# Validate environment variables
if not all([openai_api_key, search_service_name, search_admin_key, search_index_name]):
    raise ValueError("One or more required environment variables are not set")

# Initialize Azure Cognitive Search client
search_endpoint = f"https://{search_service_name}.search.windows.net"
credential = AzureKeyCredential(search_admin_key)
search_client = SearchClient(endpoint=search_endpoint, index_name=search_index_name, credential=credential)

# Initialize OpenAI client

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

    print(f"Received question: {question}")
    try:
        # Retrieve relevant documents from Azure Cognitive Search
        results = search_client.search(search_text=question, select=["id", "content"], top=5)
        context = " ".join([result["content"] for result in results])

        # Call OpenAI to get the answer
        response = client.completions.create(engine="davinci",
        prompt=f"Context: {context}\n\nQuestion: {question}\nAnswer:",
        max_tokens=150)
        answer = response.choices[0].text.strip()
        print(f"Response: {answer}")
        return jsonify({"response": answer})
    except openai.OpenAIError as e:
        print(f"OpenAIError: {e}")
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": "An error occurred. Please try again later."}), 500

if __name__ == '__main__':
    app.run(port=5000, debug=True)
