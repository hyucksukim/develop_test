from cmath import exp
from operator import getitem
from turtle import pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from pydantic import BaseModel
import pymongo # import
from pymongo import MongoClient
from bson.json_util import loads, dumps

import requests
import json
import urllib

#from inference import load_model
#from recommenders.datasets.sparse import AffinityMatrix
#from recommenders.utils.python_utils import binarize
import pandas as pd
import numpy as np
import random
import time

origins = ["*"]

app = FastAPI()

mongodb_client = "mongodb://118.67.143.144:30001/"

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = pymongo.MongoClient(mongodb_client)
db = client["amazon"]
collection = db["train"]
query = {}
cursor = collection.find(query, projection={'_id': 0, 
                                            "asin": 1})

df = pd.DataFrame.from_dict(cursor)
b = list(df['asin'].unique())
print("df : ", len(b))

@app.get("/rand/{idx}")
def read_item1(idx):
    time.sleep(0.1)
    return random.sample(b, int(idx))

def api_load_image(asin):
    input_text = asin
    client_id = "EZKJjFYCeCFKfj3qk2OM"
    client_secret = "dLMxOkCV6M"
    input_text_n = urllib.parse.quote(input_text)
    url = "https://openapi.naver.com/v1/search/book?query=" + input_text_n +"&display=3&sort=count"
    Send_request = urllib.request.Request(url)
    Send_request.add_header("X-Naver-Client-Id", client_id)
    Send_request.add_header("X-Naver-Client-Secret", client_secret)
    response = urllib.request.urlopen(Send_request)
    success = response.getcode()
    
    try:
        Response = response.read()
        tokens = json.loads(Response.decode('utf-8'))
        return tokens['items'][0]['image']
    except:
        return "https://bookthumb-phinf.pstatic.net/cover/061/154/06115486.jpg?type=m1&udate=20210317"

@app.get("/search/{asin}")
def load_image(asin):
    time.sleep(0.54)
    client = pymongo.MongoClient(mongodb_client)
    # db(=database): "recsys09"의 "amazon" 데이터베이스
    db = client["amazon"]
    # Collection(=table): "amazon" 데이터 베이스의 "books" 테이블
    Collection = db["url"]
    
    query = {'asin': asin}
    cursor = Collection.find_one(filter=query ,projection={'image_url' : True})
    try:
        return cursor['image_url'][0]
    except:
        return api_load_image(asin)
    
@app.get("/inference/{user}")
def getpredict(user):
    time.sleep(0.1)
    client = MongoClient(mongodb_client)
    db = client["amazon"]
    collection = db["train"]
    query = {'reviewerID': user}
    cursor = collection.find(query, projection={'_id': 0, 'asin':1, 'reviewText':1, 'overall':1})
    result = loads(dumps(cursor))
    
    items = [item['asin'] for item in result]
    rate = [item['overall'] for item in result]

    query_items = {"itemID": items, "rating": rate}

    model = load_model()

    df = pd.DataFrame.from_dict(query_items)
    df['userID'], df['timestamp'] = 1, 1
    df = df[['userID', 'itemID', 'rating', 'timestamp']]
    unique_train_items = np.load('C:\\Users\\dpqls\\SYB\\app\\data\\train_items.npy', allow_pickle=True)
    am_inference = AffinityMatrix(df=df, items_list=unique_train_items)
    
    inference_data, inference_map_users, inference_map_items = am_inference.gen_affinity_matrix()
    inference_data = binarize(a=inference_data, threshold=3.5)

    top_k =  model.recommend_k_items(x=inference_data,
                                                  k=10,
                                                  remove_seen=True
                                                  )
    
    top_k_df = am_inference.map_back_sparse(top_k, kind='prediction')

    return list(top_k_df['itemID'].values)
    
li = []
class Inter(BaseModel):
    id: str
    asin: str
    rate: int
    review: str
    
@app.post("/intersave")
def intersave(inter:Inter):
    
    client = pymongo.MongoClient(mongodb_client)
    db = client["amazon"]
    collection = db["train"]
    if inter.review:
        query = {'asin': inter.asin, 'reviewerID': inter.id, 'overall': inter.rate, 'reviewText': inter.review}
    else:
        query = {'asin': inter.asin, 'reviewerID': inter.id, 'overall': inter.rate}
    try:
        cursor = collection.insert_one(query)
        return "done"
    except:
        return "fail"

@app.get("/items/{asin}")
def get_item(asin):
    time.sleep(0.54)
    client = MongoClient(mongodb_client)
    db = client["amazon"]
    collection = db["books"]
    query = {'asin': asin}
    cursor = collection.find(query, projection={'_id': False})
    result = loads(dumps(cursor))
    return result

@app.get("/title/{asin}")
def get_title(asin):
    items = get_item(asin)
    return items[0]['title']
    
    