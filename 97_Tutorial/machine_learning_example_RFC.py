import numpy as np
import pandas as pd
from MyPackage.DataExplore import data_explore_char
from MyPackage.DataExplore import data_explore_num
from MyPackage.DataPrepare import my_onehot_encoder
from MyPackage.DataPrepare import missing_imputation
from MyPackage.DataPrepare import data_preparation
from MyPackage.DataPrepare import my_dataframe_split
from MyPackage.MySkLearn.RFLearner import RFLearner


print("""\
    ==================================================================
    =========================Random Forest============================
    ==================================================================
    """
      )

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 1: data preparation

# load training data
df_base = pd.read_csv("C:/Users/jason/PycharmProjects/Python/MyData/bank-full.csv", sep=";", header=0)

# load scoring data
df_to_score = pd.read_csv("C:/Users/jason/PycharmProjects/Python/MyData/bank_to_score.csv", sep=";", header=0)

# create response variable
df_base['response'] = np.where(df_base['y'] == "yes", 1, 0)
df_base = df_base.drop(['y'], axis=1)
df_to_score = df_to_score.drop(['y'], axis=1)

# data preparation: missing value imputation and one-hot encoding
df_base2 = data_preparation(df_base)
df_to_score = data_preparation(df_to_score, df_base)

# create in-time training and validation sample
df_train, df_valid, df_holdout = my_dataframe_split(df_base2, frac_train=0.3, frac_valid=0.3, random_state=1)

print("scoring sample dimension: ", df_to_score.shape)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 2: model development and validation

# train model
new_model = RFLearner(mode='superfast', df_train=df_train, df_valid=df_valid, str_resp='response')

print(new_model.hyper_param)
