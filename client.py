from pathway.xpacks.llm.vector_store import VectorStoreClient
from pymongo import MongoClient
from collections import defaultdict

PATHWAY_PORT = 8765
PATHWAY_HOST = "127.0.0.1"
# PATHWAY_HOST = "3.6.115.182"

client = VectorStoreClient(
    host=PATHWAY_HOST,
    port=PATHWAY_PORT,
)

dbclient = MongoClient("mongodb://localhost:27017")
db = dbclient["interiit"]
db = db["chunks"]

def graph_lookup(docs,n=3):

    chunks = defaultdict(lambda: defaultdict(str))
    for doc in docs:
        curr_id = (doc["metadata"]["id"]) 
        pipeline = [
            {
                "$match": {
                    "_id": curr_id  
                }
            },
            {
                "$graphLookup": {
                    "from": "chunks",  # The collection to search within
                    "startWith": "$next_id",  # Start from the next chunk of the current document
                    "connectFromField": "next_id",  # Field to follow for the next chunk
                    "connectToField": "_id",  # Field to connect to (we are linking the `next_id` to `_id`)
                    "as": "next_chunks",  # Store the resulting linked chunks in `next_chunks`
                    "maxDepth": (n-1)//2,  # Limit depth to avoid infinite loops (adjust as needed)
                    "depthField": "depth",  # Optionally, store the depth of each document in the result
                    "restrictSearchWithMatch": {"next_id": {"$ne": None}}  # Ensure the next_id exists
                }
            },
            {
                "$graphLookup": {
                    "from": "chunks",
                    "startWith": "$prev_id",  # Now trace the previous chunks
                    "connectFromField": "prev_id",  # Link back to the previous chunk
                    "connectToField": "_id",  # Field to connect to (link `prev_id` to `_id`)
                    "as": "prev_chunks",  # Store the linked chunks in `prev_chunks`
                    "maxDepth": (n-1)//2,  # Limit the depth for reverse lookup as well
                    "depthField": "depth",
                    "restrictSearchWithMatch": {"prev_id": {"$ne": None}}  # Ensure the prev_id exists
                }
            }
        ]
        result = list(db.aggregate(pipeline))
        doc_id = curr_id.split("_")[0]
        chunk_id = curr_id.split("_")[1]
        chunks[doc_id][chunk_id] = result[0]["chunk"][0]
        for chunk in result[0]["next_chunks"]:
            chunk_id = chunk['_id'].split("_")[1]
            chunks[doc_id][chunk_id] = chunk["chunk"][0]
        for chunk in result[0]["prev_chunks"]:
            chunk_id = chunk['_id'].split("_")[1]
            chunks[doc_id][chunk_id] = chunk["chunk"][0]
    result_list = []
    for doc_id in sorted(chunks.keys(),key=lambda x: int(x)):
        for chunk_id in sorted(chunks[doc_id].keys(), key=lambda x: int(x)):  
            result_list.append(chunks[doc_id][chunk_id])
    return result_list

def retrieve_documents(query: str):
    documents = client.query(query, k=3)
    documents = graph_lookup(documents)
    return documents


print(retrieve_documents("What does company licences include?"))