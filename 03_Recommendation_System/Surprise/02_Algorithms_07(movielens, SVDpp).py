import numpy as np
import pandas as pd
import datetime as dt
import time

from surprise import SVDpp
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
rating_train, rating_test=train_test_split(rating, train_size=0.1, test_size=0.01, random_state=12345)
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

#SVD Model

n_factors=[20] # where default = 20
n_epochs=[5] # where default = 20
lr_all=[0.007] # where default = 0.007
reg_all=[0.02] # where default = 0.02

count=1

for i in n_factors:
    for j in n_epochs:
        for k in lr_all:
            for m in reg_all:
                start = dt.datetime.today()
                print("================================================")
                algo = SVDpp(n_factors=i, n_epochs=j, lr_all=k, reg_all=m)

                algo.train(trainset)
                print("This is the #" + str(count) + " parameter combination")
                predictions=algo.test(testset)

                print("n_factors="+str(i)+", n_epochs="+str(j)+", lr_all="+str(k)+", reg_all="+str(m))
                accuracy.rmse(predictions, verbose=True)
                accuracy.fcp(predictions, verbose=True)
                accuracy.mae(predictions, verbose=True)
                count=count+1
                end = dt.datetime.today()
                print("Runtime: "+str(end - start))