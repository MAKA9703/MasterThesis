#Import Libraries
import pandas as pd

print("Implementing PyTerrier.....")
#Implement PyTerrier
import pyterrier as pt
if not pt.started():
    pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])

print("Loading Data....")
# Load Data
#documents
docs = pd.read_json('./scifact/corpus.jsonl', lines=True, dtype=str)
docs = docs.rename(columns={"_id": "docno"})

#train_data
df_train = pd.read_csv('./scifact/train.csv', sep='\t', dtype=str)
train_query = df_train[['qid', 'query']]

#test_data 
df_test = pd.read_csv('./scifact/test.csv', sep='\t', dtype=str)
df_test2 = df_test[['qid', 'query']]
df_test2.to_csv('./scifact/my_test_queries.csv', sep = '\t', index=False, header=False)
test_query = pt.io.read_topics('./scifact/my_test_queries.csv', format='singleline')


#qrels
train_qrels = pd.read_csv('./scifact/qrels/train.tsv', sep='\t', dtype=str)
test_qrels = pd.read_csv('./scifact/qrels/test.tsv', sep='\t', dtype=str)
test_qrels = test_qrels.rename(columns={"query-id": "qid", "corpus-id" : "docno", "score": "label"})
test_qrels['iteration'] = 0



print("Loading already build index for documents.....")
# Loading already build index
index_ref = pt.IndexRef.of("./indexes_scifact/both/data.properties")
index = pt.IndexFactory.of(index_ref)


print("Building Sparse IR Systems....")
#Build Sparse IR Systems
tf = pt.BatchRetrieve(index, wmodel="Tf")
tf_idf = pt.BatchRetrieve(index, wmodel="TF_IDF")
bm25 = pt.BatchRetrieve(index, wmodel="BM25")


print("Evaluating Sparse IR models.....")
# Evaluate models on queries using PyTerrier Experiment Interface
test_qrels = test_qrels.astype({'label': 'int32'})
#final_res = pt.Experiment(
#                        retr_systems = [tf, tf_idf, bm25],
#                        names =  ["TF", "TF-IDF", "BM25"],
#                        topics = test_query, 
#                        qrels = test_qrels,
#                        #eval_metrics = ["map", "ndcg", "ndcg_cut_10", "P_10", "recall_10"]
#                        eval_metrics = ["ndcg_cut_10", "ndcg_cut_100", "ndcg_cut_1000",
#                                        "map_cut_10", "map_cut_100", "map_cut_1000", 
#                                        "P_10", "P_100", "P_1000", 
#                                        "recall_10", "recall_100", "recall_1000"])

final_res_ndcg = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_query, 
                        qrels = test_qrels,
                        eval_metrics = ["ndcg_cut_10", "ndcg_cut_100", "ndcg_cut_1000"])

final_res_map = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_query, 
                        qrels = test_qrels,
                        eval_metrics = ["map_cut_10", "map_cut_100", "map_cut_1000"])

final_res_precision = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_query, 
                        qrels = test_qrels,
                        eval_metrics = ["P_10", "P_100", "P_1000"])

final_res_recall = pt.Experiment(
                        retr_systems = [bm25],
                        names =  ["BM25"],
                        topics = test_query, 
                        qrels = test_qrels,
                        #eval_metrics = ["map", "ndcg", "ndcg_cut_10", "P_10", "recall_10"]
                        eval_metrics = ["recall_10", "recall_100", "recall_1000"])

print(final_res_ndcg)
print(final_res_map)
print(final_res_precision)
print(final_res_recall)