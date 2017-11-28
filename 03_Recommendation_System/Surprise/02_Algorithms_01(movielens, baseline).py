import numpy as np
import pandas as pd
import datetime
import time

from surprise import SVD
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

## Baseline model using ALS
n_epochs=[5, 10]
reg_u=[12, 24] # where reg_u>0, and default = 15
reg_i=[5, 10] # where reg_i>0, and default = 10

count=1

for i in n_epochs:
    for j in reg_u:
        for k in reg_i:
            print("================================================")
            bsl_options = {'method': 'als',
                           'n_epochs': i,
                           'reg_u': j,
                           'reg_i': k
                           }

            algo = BaselineOnly(bsl_options=bsl_options)

            algo.train(trainset)
            print("This is the #" + str(count) + " parameter combination")
            predictions=algo.test(testset)

            print("n_epochs="+str(i)+", "+"reg_u="+str(j)+", "+"reg_i="+str(k))
            accuracy.rmse(predictions, verbose=True)
            accuracy.fcp(predictions, verbose=True)
            accuracy.mae(predictions, verbose=True)
            count=count+1


## baseline model using SGD
n_epochs=[5, 10]
reg=[0.2, 0.02] # where reg_u>0, and default = 0.02
learning_rate=[0.05, 0.005] # where between 0 and 1, and default = 0.005

count=1

for i in n_epochs:
    for j in reg:
        for k in learning_rate:
            print("================================================")

            bsl_options = {'method': 'sgd',
                           'reg': j,
                           'learning_rate': k
                           }

            algo = BaselineOnly(bsl_options=bsl_options)

            algo.train(trainset)
            print("This is the #" + str(count) + " parameter combination")
            predictions=algo.test(testset)

            print("n_epochs="+str(i)+", "+"reg="+str(j)+", "+"learning_rate="+str(k))
            accuracy.rmse(predictions, verbose=True)
            accuracy.fcp(predictions, verbose=True)
            accuracy.mae(predictions, verbose=True)
            count=count+1

