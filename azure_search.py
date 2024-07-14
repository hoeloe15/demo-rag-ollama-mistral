from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import ComplexField, SearchIndex, SimpleField, SearchFieldDataType
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

search_service_name = os.getenv('AZURE_SEARCH_SERVICE_NAME')
admin_key = os.getenv('AZURE_SEARCH_ADMIN_KEY')
index_name = os.getenv('AZURE_SEARCH_INDEX_NAME')

# Debugging output
print(f"Search Service Name: {search_service_name}")
print(f"Admin Key: {admin_key[:5]}...")  # Only print the first few characters for security
print(f"Index Name: {index_name}")

# Create the SearchIndexClient
endpoint = search_service_name
print(f"Endpoint: {endpoint}")
credential = AzureKeyCredential(admin_key)
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

# Define the index
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
    SimpleField(name="content", type=SearchFieldDataType.String, searchable=True),
    SimpleField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Double))
]

index = SearchIndex(name=index_name, fields=fields)

# Create the index
index_client.create_index(index)
print(f"Index '{index_name}' created successfully.")
