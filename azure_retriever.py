import logging
from azure.search.documents import SearchClient
from langchain.schema import Document
from typing import List

logger = logging.getLogger(__name__)

class AzureSearchRetriever:
    """Custom Azure Search Retriever."""
    def __init__(self, search_client: SearchClient):
        self.search_client = search_client

    def get_relevant_documents(self, query: str, max_documents: int = 5) -> List[Document]:
        logger.info(f"Retrieving relevant documents for query: {query}")
        results = self.search_client.search(search_text=query, select=["id", "content", "embedding"], top=max_documents)
        documents = [Document(page_content=result["content"], metadata={"id": result["id"], "embedding": result["embedding"]}) for result in results]
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
