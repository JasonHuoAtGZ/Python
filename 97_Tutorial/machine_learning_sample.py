print("""\
    ==================================================================
    =============================Indexing=============================
    =============================20210325=============================
    ==================================================================
    """
      )

import numpy as np
import pandas as pd
from MyPackage.DataExplore import data_explore_char
from MyPackage.DataExplore import data_explore_num
from MyPackage.DataPrepare import my_onehot_encoder
from MyPackage.DataPrepare import data_preparation
from MyPackage.DataPrepare import my_dataframe_split


"""
randseed = 1

frac_train = 0.3333
frac_valid = 0.3333
frac_valid_adj = frac_valid/(1-frac_train)

df_base_train = df_base.sample(frac=frac_train, random_state=randseed)

df_base_valid = df_base.drop(df_base_train.index).sample(frac=frac_valid_adj, random_state=randseed)

df_base_holdout = df_base.drop(df_base_train.index).drop(df_base_valid.index)

print(df_base.shape)
print(df_base_train.shape)
print(df_base_valid.shape)
print(df_base_holdout.shape)
"""



print("""\
    ==================================================================
    ========================GradientBoosting==========================
    ==================================================================
    """
      )

df_base = pd.read_csv("C:/Users/jason/PycharmProjects/Python/MyData/bank-full.csv", sep=";", header=0)
df_to_score = pd.read_csv("C:/Users/jason/PycharmProjects/Python/MyData/bank_to_score.csv", sep=";", header=0)

df_base['response'] = np.where(df_base['y'] == "yes", 1, 0)
df_base = df_base.drop(['y'], axis=1)
df_to_score = df_to_score.drop(['y'], axis=1)

df_base2 = data_preparation(df_base)
df_to_score = data_preparation(df_to_score, df_base)

df_train, df_valid, df_holdout = my_dataframe_split(df_base2, frac_train=0.3, frac_valid=0.3, random_state=1)

print(df_train.head(5))
print(df_valid.head(5))
print(df_holdout.head(5))
print(df_to_score.head(5))

from MyPackage.MySkLearn.GBLearner import GBLearner

new_model = GBLearner(mode='default', df_train=df_train, df_valid=df_valid, str_resp='response')
new_model.training()

print(new_model.best_param)
print(new_model.top_variable)
print(new_model.best_model)

print(new_model.validating_param(df_holdout))
print(new_model.validating_out(df_holdout).head(10))


