import os
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    VectorSearch,
    VectorSearchAlgorithmConfiguration,
)

# ---------- Config ----------
endpoint = os.environ["AZURE_SEARCH_ENDPOINT"]
key = os.environ["AZURE_SEARCH_KEY"]
index_name = os.environ.get("AZURE_SEARCH_INDEX", "rag-index")
embedding_dim = int(os.environ.get("EMBEDDING_DIM", "1536"))

# ---------- Client ----------
client = SearchIndexClient(endpoint, AzureKeyCredential(key))

# ---------- Fields ----------
fields = [
    # Primary key for each chunk record
    SimpleField(name="id", type="Edm.String", key=True),

    # Chunk text used for keyword search + returned to the LLM for grounding
    SearchableField(name="content", type="Edm.String"),

    # Source doc name for citations/filtering
    SearchableField(name="doc", type="Edm.String", filterable=True, facetable=True),

    # Chunk order within doc (useful for debugging/reconstruction)
    SimpleField(name="chunk_index", type="Edm.Int32", filterable=True),

    # Stable citation id per chunk
    SimpleField(name="chunk_id", type="Edm.String", filterable=True),

    # Vector embedding field (used for vector similarity)
    SimpleField(
        name="embedding",
        type="Collection(Edm.Single)",
        searchable=True,
        vector_search_dimensions=embedding_dim,
        vector_search_configuration="vector-config"
    ),
]

# ---------- Vector search config ----------
# HNSW is the standard ANN algorithm used by Azure AI Search for vector search
vector_search = VectorSearch(
    algorithm_configurations=[
        VectorSearchAlgorithmConfiguration(
            name="vector-config",
            kind="hnsw",
        )
    ]
)

index = SearchIndex(
    name=index_name,
    fields=fields,
    vector_search=vector_search,
)

client.create_or_update_index(index)
print(f"✅ Index created/updated: {index_name} (embedding_dim={embedding_dim})")