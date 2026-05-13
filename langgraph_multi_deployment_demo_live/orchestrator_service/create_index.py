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

embedding_dim = int(os.environ.get("EMBEDDING_DIM", "1536"))  # text-embedding-3-small → 1536
ALGO_NAME = "hnsw-algo"
PROFILE_NAME = "vector-profile"

client = SearchIndexClient(endpoint, AzureKeyCredential(key))

fields = [
    SimpleField(name="id", type=SearchFieldDataType.String, key=True),

    # What this chunk is
    SimpleField(name="entity_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
    SearchableField(name="title", type=SearchFieldDataType.String),
    SearchableField(name="content", type=SearchFieldDataType.String),

    # Traceability back to PostgreSQL
    SimpleField(name="source_table", type=SearchFieldDataType.String, filterable=True, facetable=True),
    SimpleField(name="source_id", type=SearchFieldDataType.String, filterable=True),
    SimpleField(name="chunk_index", type=SearchFieldDataType.Int32, filterable=True),
    SimpleField(name="chunk_id", type=SearchFieldDataType.String, filterable=True),

    # Filters (subset applies by entity_type)
    SimpleField(name="policy_type", type=SearchFieldDataType.String, filterable=True, facetable=True),
    SimpleField(name="severity", type=SearchFieldDataType.String, filterable=True, facetable=True),
    SimpleField(name="owner_team", type=SearchFieldDataType.String, filterable=True, facetable=True),
    SimpleField(name="effective_date", type=SearchFieldDataType.String, filterable=True),

    # Vector field
    SearchField(
        name="embedding",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=embedding_dim,
        vector_search_profile_name=PROFILE_NAME,
    ),
]

vector_search = VectorSearch(
    algorithms=[HnswAlgorithmConfiguration(name=ALGO_NAME)],
    profiles=[VectorSearchProfile(name=PROFILE_NAME, algorithm_configuration_name=ALGO_NAME)],
)

index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)
client.create_or_update_index(index)

print(f"✅ Index created/updated: {index_name} (dims={embedding_dim}, profile={PROFILE_NAME})")