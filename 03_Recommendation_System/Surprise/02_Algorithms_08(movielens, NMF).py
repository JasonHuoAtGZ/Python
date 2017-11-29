import numpy as np
import pandas as pd
import datetime as dt
import time

from surprise import NMF
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
rating_train, rating_test=train_test_split(rating, train_size=0.01, test_size=0.01, random_state=12345)
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

#NMF Model

n_factors=[15] # where default = 15
n_epochs=[50] # where default = 50
reg_pu=[0.06] # where default = 0.06
reg_qi=[0.06] # where default = 0.06

count=1

for i in n_factors:
    for j in n_epochs:
        for k in reg_pu:
            for m in reg_qi:
                start = dt.datetime.today()
                print("================================================")
                algo = NMF(n_factors=i, n_epochs=j, reg_pu=k, reg_qi=m, biased=True)

                algo.train(trainset)
                print("This is the #" + str(count) + " parameter combination")
                predictions=algo.test(testset)

                print("n_factors="+str(i)+", n_epochs="+str(j)+", reg_pu="+str(k)+", reg_qi="+str(m))
                accuracy.rmse(predictions, verbose=True)
                accuracy.fcp(predictions, verbose=True)
                accuracy.mae(predictions, verbose=True)
                count=count+1
                end = dt.datetime.today()
                print("Runtime: "+str(end - start))