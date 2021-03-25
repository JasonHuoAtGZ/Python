import pandas as pd
import numpy as np

# from sklearn.model_selection import train_test_split

df_raw=pd.read_csv("C:/Users/jason/Python/data/default_of_credit_card_clients.csv")

print("================================")
print(df_raw.info())
print(df_raw.groupby(['Y']).agg({'Y':[np.size]}))

# sample preparation
# arr_raw=df_raw.values
# train, valid = train_test_split(arr_raw, test_size=0.5, random_state=12345)
df_train=df_raw.sample(frac=0.5, replace=False, random_state=12345).rename(columns={'Y':'response'})
df_valid=df_raw.sample(frac=0.5, replace=False, random_state=54321).rename(columns={'Y':'response'})

print("================================")
print(df_train.info())
print(df_train.groupby(['response']).agg({'response':[np.size]}))

"""
print("================================")
print(df_valid.info())
print(df_valid.groupby(['response']).agg({'response':[np.size]}))

# model processing
start=dt.datetime.today()
clf=GBL(mode='default', df_train=df_train, df_valid=df_valid, str_resp='response')
clf._training()
end=dt.datetime.today()
#print(clf.param)
print(end-start)
"""

