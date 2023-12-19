import pandas as pd


df_train = pd.read_csv("datasets/scifact/train.csv")
df_val = pd.read_csv("datasets/scifact/test.csv")
df_train[['qid','query']].to_csv("datasets/scifact/train.source_rl",index = None,header=None)
df_train[['qid','text']].to_csv("datasets/scifact/train.target_rl",index = None, header = None)

df_val[['qid','query']].to_csv("datasets/scifact/val.source_rl",index = None,header=None)
df_val[['qid','text']].to_csv("datasets/scifact/val.target_rl",index = None,header=None)

df_val[['qid','query']].drop_duplicates().to_csv("datasets/scifact/test.source",index = None,header = None)
df_val[['qid','text']].drop_duplicates().to_csv("datasets/scifact/test.target",index = None,header = None)