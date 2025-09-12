import json
import os
import umap
import numpy as np
from typing import Optional
from sentence_transformers import SentenceTransformer
from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

RANDOM_SEED = 224 
METRIC = "cosine"
truncate_dim=128
TRUNCATE_DIM = 128
MODEL_ID_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"

def get_model() :
    model = SentenceTransformer(MODEL_ID_EMBEDDING, truncate_dim=TRUNCATE_DIM)
    return model

def create_milvus_collection() :
    """ 
    Create the milvus database and collections.
    """

    # Connect to Milvus Lite (local file-based DB)
    connections.connect(alias="default", uri="milvus.db")

    # 2. Define collection schema
    collection_name_summaries = "summaries"
    if utility.has_collection(collection_name_summaries):
        utility.drop_collection(collection_name_summaries)

    fields_summary = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="page_number", dtype=DataType.INT64),
        FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2000),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=TRUNCATE_DIM)
    ]
    schema_summary = CollectionSchema(fields_summary, description="Summaries with embeddings")
    collection_summary = Collection(name=collection_name_summaries, schema=schema_summary)

    return collection_summary

def insert_into_milvus_collection_with_index(collection_summary, meta, summaries, embeddings) :
    """ 
    Insert into milvus colllection and create index as well.
    """

    # get page_number list from meta data.
    page_numbers = [item["page_number"] for item in meta]

    # Insert summaries
    collection_summary.insert([
        page_numbers,
        summaries,
        embeddings.tolist()
    ])

    # Create index for similarity search
    collection_summary.create_index(field_name="embedding")
    collection_summary.load()

def search_milvus(meta, collection_summary, query_sentence) :
    """ 
    Search and print the answers for the given query from milvus database
    """
    print("--------------------------------------------------------------------------------")
    print(f"query_sentence : {query_sentence}")
    print("--------------------------------------------------------------------------------")

    # Query embeddings
    embeddings = generate_embeddings([query_sentence])

    # search
    results = collection_summary.search(
        data = embeddings.tolist(),
        anns_field = "embedding",
        param = {"nprobe": 10},
        limit = 2,
        output_fields=["page_number", "summary"]
    )

    print("\nSearch results:")
    for res in results[0]:
        print("--------------------------------------")
        summary = res.entity.get('summary')
        page_number = res.entity.get('page_number')

        print(f"Matched text: {summary}, distance: {res.distance:.4f}")
        print(f"Matched page_number: {page_number}")

        links = [item for item in meta if item.get("page_number") == page_number]

        print(f"Matched links: {links}")
        print("--------------------------------------")


def load_json (json_files) :
    """
    Loads the json file and return the summary field as list
    """   

    data = []
    for file_path in json_files:
        if os.path.exists(file_path):
            with open(file_path, "r", encoding="utf-8") as f:
                data.extend(json.load(f))   # Add entire JSON object

    # take "summary" field
    summaries = [item["summary"] for item in data]

    return data, summaries

def generate_embeddings (docs) :
    """
    Creates the embeddings for the given docs
    """    
    # Load embedding model
    model = SentenceTransformer(MODEL_ID_EMBEDDING, truncate_dim=TRUNCATE_DIM)

    # Encode documents
    embeddings = model.encode(docs, convert_to_numpy=True)

    return embeddings

def ingest () :
    """
    Main method to ingest the data from json file into milvus database
    """

    print("******************************************************************************************")

    json_files = ["./data/2501.14312v1.json", "./data/2506.08276v1.json"]

    ### Load json
    meta, summaries = load_json(json_files)

    ### generate_embeddings
    embeddings = generate_embeddings(summaries)

    ### create milvus collection
    collection_summary = create_milvus_collection()

    ### insert into milvus
    insert_into_milvus_collection_with_index(collection_summary, meta, summaries, embeddings)

    ### Search in milvus
    query = "Explain dynamic batching"
    search_milvus (meta, collection_summary, query)

    ### MEta data...
    print("******************************************************************************************")

ingest()
