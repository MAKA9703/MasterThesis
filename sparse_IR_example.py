#Import Libraries
import numpy as np
import pandas as pd

print("Implementing PyTerrier.....")
#Implement PyTerrier
import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

print("Loading Data....")
# Load Data
#documents
docs = pd.read_csv('./toy_data/docs.csv', dtype=str)

#queries
queries = pd.read_csv('./toy_data/queries.csv', dtype=str)

#qrels
qrels = pd.read_csv('./toy_data/qrels.csv', dtype=str)

# Build DEFAULT index
#indexer = pt.DFIndexer("./indexes/default", overwrite=True, blocks=True)
#index_ref = indexer.index(docs["text"], docs["docno"])
#index = pt.IndexFactory.of(index_ref)

print("Loading already build index for documents.....")
# Loading already build index
index_ref = pt.IndexRef.of("./indexes/default/data.properties")
index = pt.IndexFactory.of(index_ref)


print("Building Sparse IR Systems....")
#Build Sparse IR Systems
tf = pt.BatchRetrieve(index, wmodel="Tf")
tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")


print("Evaluating Sparse IR models.....")
# Evaluate models on queries using PyTerrier Experiment Interface
qrels = qrels.astype({'label': 'int32'})
final_res = pt.Experiment(
                        retr_systems = [tf, tf_idf, bm25],
                        names =  ["TF", "TF-IDF", "BM25"],
                        topics = queries, 
                        qrels = qrels,
                        eval_metrics = ["map", "ndcg", "ndcg_cut_10", "P_10"])

print(final_res)