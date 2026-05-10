import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchField,
    SearchFieldDataType,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
key = os.environ["AZURE_SEARCH_KEY"]
index_name = os.environ.get("AZURE_SEARCH_INDEX", "rag-index")

# text-embedding-3-small produces 1536-d vectors (you deployed embedding-model)
embedding_dim = int(os.environ.get("EMBEDDING_DIM", "1536"))

# Names used by vector search
ALGO_NAME = "hnsw-algo"
PROFILE_NAME = "vector-profile"

client = SearchIndexClient(endpoint, AzureKeyCredential(key))

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),

    # Full-text searchable chunk content (keyword side of hybrid)
    SearchableField(name="content", type=SearchFieldDataType.String),

    # Source metadata
    SearchableField(name="doc", type=SearchFieldDataType.String, filterable=True, facetable=True),
    SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
    SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),

    # Vector field (semantic side of hybrid)
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=embedding_dim,
        vector_search_profile_name=PROFILE_NAME,
    ),
]

vector_search = VectorSearch(
    algorithms=[
        HnswAlgorithmConfiguration(
            name=ALGO_NAME,
            # parameters optional; defaults are fine for prototype
        )
    ],
    profiles=[
        VectorSearchProfile(
            name=PROFILE_NAME,
            algorithm_configuration_name=ALGO_NAME,
        )
    ],
)

index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

client.create_or_update_index(index)
print(f"✅ Index created/updated: {index_name} (dims={embedding_dim}, profile={PROFILE_NAME})")
