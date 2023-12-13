#Import libraries

print("Importing Libraries...")

import numpy as np
import pandas as pd

from beir import util, LoggingHandler
from beir.retrieval import models
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES

import logging
import pathlib, os

#Libraries for DPR model
from transformers import DPRContextEncoder, DPRContextEncoderTokenizerFast
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizerFast
from typing import Union, List, Dict, Tuple
from tqdm.autonotebook import trange
import torch


print("Loading Data...")
#Load Data
#documents
docs = pd.read_csv('./toy_data/docs.csv', dtype=str)

#queries
queries = pd.read_csv('./toy_data/queries.csv', dtype=str)

#qrels
qrels = pd.read_csv('./toy_data/qrels.csv', dtype=str)

print("Restructering data...")
#Refactoring data to fit DPR model.
new_docs = {}
for i in range(len(docs)):
    new_docs[docs['docno'][i]] = {'text' : docs['text'][i]}

new_queries = {}
for i in range(len(queries)):
    new_queries[queries['qid'][i]] = queries['query'][i]


new_qrels = {}
for i in range(len(qrels)):
    new_qrels[qrels['qid'][i]] = {qrels['docno'][i] : int(qrels['label'][i])}


#IMPLEMENTED MODEL FROM https://github.com/beir-cellar/beir
#https://github.com/beir-cellar/beir/blob/main/beir/retrieval/models/dpr.py

print("Creating DPR model...")

class DPR:
    def __init__(self, model_path: Union[str, Tuple] = None, **kwargs):
        # Query tokenizer and model
        self.q_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(model_path[0])
        self.q_model = DPRQuestionEncoder.from_pretrained(model_path[0])
        #self.q_model.cuda()
        self.q_model.eval()
        
        # Context tokenizer and model
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(model_path[1])
        self.ctx_model = DPRContextEncoder.from_pretrained(model_path[1])
        self.ctx_model.eval()
    
    def encode_queries(self, queries: List[str], batch_size: int = 16, **kwargs) -> torch.Tensor:
        query_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(queries), batch_size):
                encoded = self.q_tokenizer(queries[start_idx:start_idx+batch_size], truncation=True, padding=True, return_tensors='pt')
                model_out = self.q_model(encoded['input_ids'], attention_mask=encoded['attention_mask'])
                query_embeddings += model_out.pooler_output

        return torch.stack(query_embeddings)
        
    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int = 8, **kwargs) -> torch.Tensor:
        
        corpus_embeddings = []
        with torch.no_grad():
            for start_idx in trange(0, len(corpus), batch_size):
                #titles = [row['title'] for row in corpus[start_idx:start_idx+batch_size]]
                texts = [row['text']  for row in corpus[start_idx:start_idx+batch_size]]
                #encoded = self.ctx_tokenizer(titles, texts, truncation='longest_first', padding=True, return_tensors='pt')
                encoded = self.ctx_tokenizer(texts, truncation='longest_first', padding=True, return_tensors='pt')
                model_out = self.ctx_model(encoded['input_ids'], attention_mask=encoded['attention_mask'])
                corpus_embeddings += model_out.pooler_output.detach()
        
        return torch.stack(corpus_embeddings)
    

print("Creating model and retrieve results...")
model_dpr = DRES(DPR((
     "facebook/dpr-question_encoder-multiset-base",
     "facebook/dpr-ctx_encoder-multiset-base"), batch_size=16))
retriever_dpr = EvaluateRetrieval(model_dpr, score_function="dot") # or "dot" for dot-product
results_dpr = retriever_dpr.retrieve(new_docs, new_queries)

print('Computing evaluation results...')
#### Evaluate your model with NDCG@k, MAP@K, Recall@K and Precision@K  where k = [1,3,5,10,100,1000] 
ndcg, _map, recall, precision = retriever_dpr.evaluate(new_qrels, results_dpr, retriever_dpr.k_values)

print("NDCG: ", ndcg)
print("MAP: ", _map)
print("Recall: ", recall)
print("Precision: ", precision)