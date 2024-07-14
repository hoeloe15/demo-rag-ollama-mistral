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

# Create the SearchIndexClient
endpoint = f"https://{search_service_name}.search.windows.net"
credential = AzureKeyCredential(admin_key)
index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

# Define the index
fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True, filterable=True),
    SimpleField(name="content", type=SearchFieldDataType.String, searchable=True),
    ComplexField(name="embedding", type=SearchFieldDataType.Collection(SearchFieldDataType.Double), searchable=True)
]

index = SearchIndex(name=index_name, fields=fields)

# Create the index
index_client.create_index(index)
print(f"Index '{index_name}' created successfully.")
