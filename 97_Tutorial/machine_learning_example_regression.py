import numpy as np
import pandas as pd
from MyPackage.MyToolKit.DataPrepare import data_preparation
from MyPackage.MyToolKit.DataPrepare import my_dataframe_split
from MyPackage.MySkLearn.MachineLearningRegressor import MLRegressor
from MyPackage.MyToolKit.DataExplore import data_explore_num


print("""\
    ==================================================================
    ==================Machine Learning Regressor======================
    ==================================================================
    """
      )

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 1: data preparation

# load training data
df_base = pd.read_csv("C:/Users/jason/PycharmProjects/Python/MyData/beijing_PM25.csv", header=0)
df_to_score = pd.read_csv("C:/Users/jason/PycharmProjects/Python/MyData/beijing_PM25_to_score.csv", header=0)
df_base = df_base.drop(['No'], axis=1)
df_to_score = df_to_score.drop(['No'], axis=1)

print(df_base.shape)
print(df_to_score.shape)

df_base = df_base.dropna()
df_to_score = df_to_score.dropna()

df_base.rename(columns={'pm2.5': 'target'}, inplace=True)

print(df_base.shape)

# data preparation: missing value imputation and one-hot encoding
df_base2 = data_preparation(df_base)
df_to_score = data_preparation(df_to_score, df_base)

data_explore_num(df_base2) # no missing value, no character variables
data_explore_num(df_to_score)

# create in-time training and validation sample
df_train, df_valid, df_holdout = my_dataframe_split(df_base2, frac_train=0.3, frac_valid=0.3, random_state=1)


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 2: model development and validation

# params = {'n_estimators': [2000]}

# train model
new_model = MLRegressor(estimator='XGBRegressor', mode='default', df_train=df_train, df_valid=df_valid, str_resp='target')
new_model.training()

# best model
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("best model - hyper parameter")
print(new_model.best_param)
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("best model - feature importance")
print(new_model.top_variable)
# export all parameters
new_model.param.to_csv("C:/Users/jason/PycharmProjects/Python/MyData/param.csv")

# hold-out sample validation
print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("hold-out sample validation - hyper parameter")
print(new_model.validating_param(df_holdout))


print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("hold-out sample validation - top rows")
print(new_model.validating_out(df_holdout).head(10))


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# Step 3: model scoring

print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
print("model scoring - top rows")
print(new_model.scoring_out(df_to_score).head(10))

