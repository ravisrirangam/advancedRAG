import json
import umap
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

RANDOM_SEED = 224 
METRIC = "cosine"

def create_milvus_collection() :
    """ 
    Create the milvus database and collections.
    """

    # 1. Connect to Milvus Lite (local file-based DB)
    connections.connect(alias="default", uri="milvus.db")

    # 2. Define collection schema
    collection_name = "umap_embeddings"

    # Drop if already exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=10),  # Reduced to 10D
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=500)     # Store original text
    ]

    schema = CollectionSchema(fields, description="Embeddings reduced to 10D using UMAP")
    collection = Collection(name=collection_name, schema=schema, using="default", shards_num=1)
    
    return collection

def insert_into_milvus_collection_with_index(collection, docs, embeddings) :
    """ 
    Insert into milvus colllection and create index as well.
    """

    # Insert into Milvus Lite
    collection.insert([
        embeddings.tolist(),  # embeddings
        docs                      # store texts as metadata
    ])

    # 6. Create index for similarity search
    index_params = {"index_type": "IVF_FLAT", "metric_type": "L2", "params": {"nlist": 16}}
    collection.create_index(field_name="embedding", index_params=index_params)
    collection.load()

    return collection

def search_milvus(collection, umap_reducer, query_sentence) :
    """ 
    Search and print the answers for the given query from milvus database
    """

    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 7. Query / Search
    query_sentence = "Tell me about vector databases."
    query_embedding = model.encode([query_sentence])
    query_embedding_reduced = umap_reducer.transform(query_embedding)

    results = collection.search(
        data=query_embedding_reduced.tolist(),
        anns_field="embedding",
        param={"nprobe": 10},
        limit=2,
        output_fields=["text"]
    )

    print("\nSearch results:")
    for res in results[0]:
        print(f"Matched text: {res.entity.get('text')}, distance: {res.distance:.4f}")


def reduce_dimensions(embeddings: np.ndarray, dimensions: int, n_neighbors: Optional[int] = None) -> np.ndarray:
    """ 
    Reduce the Dimensions for the embeddings
    """
    
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)

    umap_reducer = umap.UMAP(n_neighbors=n_neighbors, n_components=dimensions, metric=METRIC, random_state=RANDOM_SEED)
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    return reduced_embeddings, umap_reducer

def load_json_as_docs () :
    """
    Loads the json file and return the summary field as list
    """   

    # Load JSON file
    with open("./src/data/2506.08276v1.json", "r") as f:
        data = json.load(f)

    # take "summary" field
    docs = [item["summary"] for item in data]

    return docs

def generate_embeddings (docs) :
    """
    Creates the embeddings for the given docs
    """    
    # Load embedding model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # Encode documents
    embeddings = model.encode(docs, convert_to_numpy=True)

    return embeddings

def ingest () :
    """
    Main method to ingest the data from json file into milvus database
    """

    ### Load json
    docs = load_json_as_docs()

    ### generate_embeddings
    embeddings = generate_embeddings(docs)

    ### reduce dimensions
    reduced_embeddings, umap_reducer = reduce_dimensions(embeddings, dimensions=10)

    ### create milvus collection
    collection = create_milvus_collection()

    ### insert into milvus
    collection = insert_into_milvus_collection_with_index(collection, docs, reduced_embeddings)

    ### Search in milvus
    query = "Explains dynamic batching"
    search_milvus (collection, umap_reducer, query)

ingest()
