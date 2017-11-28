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
rating_train, rating_test=train_test_split(rating, train_size=0.001, test_size=0.001, random_state=12345)
print("Training sample:")
print(rating_train.describe())
print("Validation sample:")
print(rating_test.describe())

# A reader to define rating_scale param.
reader = Reader(rating_scale=(0.5, 5))

# The columns must correspond to user id, item id and ratings (in that order).
rating_train = Dataset.load_from_df(rating_train[['userID','itemID','rating']], reader)
rating_test = Dataset.load_from_df(rating_test[['userID','itemID','rating']], reader)

# load data from df to what surprise intakes
print("After load_from_df: training")

temp_df=pd.DataFrame(rating_train)
print(temp_df.info())

#print("After load_from_df: validation")
#print(pd.DataFrame(rating_test).info())

