import numpy as np
import pandas as pd
import datetime
import time

from surprise import KNNBasic
from surprise import GridSearch
from surprise import Dataset
from surprise import Reader
from surprise import BaselineOnly
from surprise import accuracy
from surprise import evaluate, print_perf

from sklearn.model_selection import train_test_split

rating=pd.read_csv('D:/01_Download/data/movielens_20171111/ratings.csv')
rating=rating.rename(columns={'userId':'userID','movieId':'itemID'})
rating=rating[['userID','itemID','rating']]

# train & test split
rating_train, rating_test=train_test_split(rating, train_size=0.001, test_size=0.001, random_state=12345)
print("================================================")
print("Training sample:")
print(rating_train.describe())
print("================================================")
print("Validation sample:")
print(rating_test.describe())

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0.5, 5))

# The columns must correspond to user id, item id and ratings (in that order).
rating_train2 = Dataset.load_from_df(rating_train[['userID','itemID','rating']], reader)
rating_test2 = Dataset.load_from_df(rating_test[['userID','itemID','rating']], reader)

trainset = rating_train2.build_full_trainset()
testset = rating_test2.build_full_trainset().build_testset()

# K Nearest Neighbors (basic)
name=['pearson_baseline'] # where default = 'msd'
user_based=[True, False] # user or item based
shrinkage=[100, 200] # where shrinkage>=0 and default = 100
k=[20, 40] # maximum neighbors where default = 40
min_k=[1, 5] # minimum neighbors where default = 1

count=1

for o in name:
    for p in user_based:
        for q in shrinkage:
            for n1 in k:
                for n2 in min_k:
                    print("================================================")
                    sim_options = {'name': o,
                                   'user_based': p,
                                   'shrinkage': q
                                   }

                    algo = KNNBasic(k=n1, min_k=n2, sim_options=sim_options)

                    algo.train(trainset)

                    print("This is the #" + str(count) + " parameter combination")

                    predictions=algo.test(testset)

                    print("name=" + str(o) + ", user_based=" + str(p) + ", shrinkage=" + str(q) + ", k=" + str(n1) + ", min_k=" + str(n2))

                    accuracy.rmse(predictions, verbose=True)
                    accuracy.fcp(predictions, verbose=True)
                    accuracy.mae(predictions, verbose=True)
                    count=count+1


name=['cosine', 'pearson', 'msd'] # where default = 'msd'
user_based=[False] # user or item based
k=[20, 40] # maximum neighbors where default = 40
min_k=[1, 5] # minimum neighbors where default = 1

count=1

for o in name:
    for p in user_based:
            for n1 in k:
                for n2 in min_k:
                    print("================================================")
                    sim_options = {'name': o,
                                   'user_based': p
                                   }

                    algo = KNNBasic(k=n1, min_k=n2, sim_options=sim_options)

                    algo.train(trainset)

                    print("This is the #" + str(count) + " parameter combination")

                    predictions=algo.test(testset)

                    print("name=" + str(o) + ", user_based=" + str(p) + ", k=" + str(n1) + ", min_k=" + str(n2))

                    accuracy.rmse(predictions, verbose=True)
                    accuracy.fcp(predictions, verbose=True)
                    accuracy.mae(predictions, verbose=True)
                    count=count+1