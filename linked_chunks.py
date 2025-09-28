from pymongo import MongoClient
import pathway as pw
from pathway.xpacks.llm.splitters import TokenCountSplitter
from typing import List,Dict

def linking(chunks : list[tuple[str,dict]],doc_num : int) -> list[tuple[str,dict]]:
    client = MongoClient("mongodb://localhost:27017")
    db = client["interiit"]
    db = db["chunks"]
    j = 0
    prev_id = None
    next_id = (str(str(doc_num) + "_" + str(j+1)))
    curr_id = (str(str(doc_num) + "_" + str(j)))
    for i in range(0, len(chunks)-1):
        doc = {
            "_id" : curr_id,
            "chunk" : chunks[i],
            "prev_id" : prev_id,
            "next_id" : next_id
        }
        chunks[i][1]['id'] = str(str(doc_num) + "_" + str(j))
        prev_id = curr_id
        curr_id = next_id
        j+=1
        next_id = (str(str(doc_num) + "_" + str(j+1)))
        db.insert_one(doc)

    k = len(chunks) -1
    doc = {
        "_id" : curr_id,
        "chunk" : chunks[k],
        "prev_id" : prev_id,
        "next_id" : None
    }
    chunks[k][1]['id'] = str(str(doc_num) + "_" + str(j))
    db.insert_one(doc)
    client.close()
    return chunks








