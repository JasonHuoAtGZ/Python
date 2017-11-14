import numpy as np
import pandas as pd
import datetime
import time

from surprise import SVD
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
print(rating_train.describe())
print(rating_test.describe())

# A reader is still needed but only the rating_scale param is requiered.
reader = Reader(rating_scale=(0.5, 5))

# The columns must correspond to user id, item id and ratings (in that order).
rating_train2 = Dataset.load_from_df(rating_train[['userID','itemID','rating']], reader)
rating_test2 = Dataset.load_from_df(rating_test[['userID','itemID','rating']], reader)

# model
#rating_train2.split(n_folds=2)
#rating_test2=rating_test2.build_full_trainset()
algo = SVD()

trainset = rating_train2.build_full_trainset()
algo.train(trainset)

testset = rating_test2.build_full_trainset().build_testset()
predictions = algo.test(testset)

accuracy.rmse(predictions, verbose=True)

predictions_df=pd.DataFrame(predictions)

print(predictions_df.head(10))
print(predictions_df.describe())



'''
print(datetime.datetime.now())
for trainset, testset in rating_train2.folds():
    print(datetime.datetime.now())
    # train and test algorithm.
    algo.train(trainset)
    predictions = algo.test(testset)

    # Compute and print Root Mean Squared Error
    rmse = accuracy.rmse(predictions, verbose=True)

validation=algo.test(rating_test2)
rmse2 = accuracy.rmse(validation, verbose=True)
print(rmse2)


data = Dataset.load_builtin('ml-100k')










# We can also do this during a cross-validation procedure!

print('CV procedure:')
data.split(3)
for i, (trainset_cv, testset_cv) in enumerate(data.folds()):

    print('fold number', i + 1)

    algo.train(trainset_cv)



    print('On testset,', end='  ')

    predictions = algo.test(testset_cv)

    accuracy.rmse(predictions, verbose=True)



    print('On trainset,', end=' ')

    predictions = algo.test(trainset_cv.build_testset())

    accuracy.rmse(predictions, verbose=True)
'''