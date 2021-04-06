print("""\
    ==================================================================
    =============================Indexing=============================
    =============================20210322=============================
    ==================================================================
    """
      )

import numpy as np
import pandas as pd

narray_1 = np.array([[1.1, 2.2, 3.3],
                     [4.5, 5.6, 6.7]])

print(narray_1)
print(narray_1 * 10)
print(narray_1 + narray_1)

print(narray_1.shape)
print(narray_1.ndim)
print(narray_1.dtype)

narray_2 = np.zeros((3, 10))
print(narray_2)

print("""\
    ==================================================================
    =============================Indexing=============================
    =============================20210323=============================
    ==================================================================
    """
      )

print("Series")
s1 = pd.Series([3, 5, 7, 9, 10])

print(s1)
print(np.dtype(s1.index))
print(s1[0])
print(s1.values)

for item in s1.index:
    print(s1[item])

s2 = pd.Series([1, 2, 3, 4], index=['001', '002', '003', '004'])
print(s2)

print("""\
    ==================================================================
    =============================Indexing=============================
    =============================20210324=============================
    ==================================================================
    """
      )

df_bank_base = pd.read_csv("C:/Users/jason/Working_Folder/03_Python/bank-full.csv", sep=";", header=0)

print(df_bank_base)
print(df_bank_base.columns)

df_temp = df_bank_base[['age', 'job']]
print(df_temp.head(10))

print(df_bank_base.info)

df_temp2 = df_bank_base['job']

print(df_temp2.value_counts())

"""
for col in df_bank_base.columns:
    print(df_bank_base[col][:10])
"""

"""
numeric_type = ['int16', 'int32', 'int64', 'float32', 'float64']
df_num = df_bank_base.select_dtypes(include=numeric_type)

df_smy = df_bank_base.describe(percentiles=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 /
                                            0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

df_smy_cnt = df_smy.loc['count']
record_cnt = df_bank_base.shape[0]
df_smy_cnt = pd.DataFrame(df_smy_cnt / record_cnt)
df_smy_cnt = df_smy_cnt.rename(columns={'count':'C_rate'})
df_smy_cnt = df_smy_cnt.transpose()

# df_temp3 = pd.DataFrame(df_smy_cnt, )

df_smy2 = df_smy_cnt.append(df_smy)
print(df_smy2)


numeric_type = ['int16', 'int32', 'int64', 'float32', 'float64']
df_char = df_bank_base.select_dtypes(exclude=numeric_type)

print(df_char.head(10))

record_cnt = df_char.shape[0]

df_sum = pd.DataFrame(columns=['id_temp'])

for col in df_char.columns:
    df_temp = pd.DataFrame(df_char[col].value_counts())
    df_temp[col+'_val'] = df_temp.index
    df_temp[col+'_p'] = df_temp[col] / record_cnt

    df_temp = df_temp.rename(columns={col:col+'_cnt'})
    df_temp = df_temp.reset_index(drop=True)

    df_temp['id_temp'] = df_temp.index

    df_sum = pd.merge(df_sum, df_temp, how='outer', on='id_temp')

    print(df_temp)

print(df_sum)
"""

from MyPackage.MyToolKit.DataExplore import data_explore_num
from MyPackage.MyToolKit.DataExplore import data_explore_char

data_explore_num(df_bank_base)
data_explore_char(df_bank_base)

"""
# Label Encoder
# 1. INSTANTIATE
# encode labels with value between 0 and n_classes-1.
le = preprocessing.LabelEncoder()

# 2. FIT AND TRANSFORM
# use df.apply() to apply le.fit_transform to all columns
df_char2 = df_char.apply(le.fit_transform)
print("==========================================")
print(df_char.head(10))
print("==========================================")
print(df_char2.head(10))
print("==========================================")

# One-Hot encoder
# 1. INSTANTIATE
enc = preprocessing.OneHotEncoder()

# 2. FIT
enc.fit(df_char)

# 3. Transform
onehotlabels = pd.DataFrame(enc.transform(df_char).toarray())
print(onehotlabels.shape)
print(onehotlabels.head(10))


df_temp = pd.DataFrame(df_char['job'].value_counts())
col_name = df_temp.columns[0]
dist_char_val = df_temp.index
dist_char_val_cnt = df_temp.shape[0]

print(col_name)
print(dist_char_val)
print(dist_char_val_cnt)

df_char_new = pd.DataFrame()

for i in range(dist_char_val_cnt-1):
    df_char_new[col_name+'_'+str(i)] = np.where(df_char[col_name] == dist_char_val[i], 1, 0)

print(df_char.head(10))
print(df_char_new.head(10))


"""

from MyPackage.MyToolKit.DataExplore import data_preparation

"""
df_char = df_bank_base.select_dtypes(include=[object])

df1 = pd.DataFrame(df_bank_base[['job', 'marital']])
df1_new = my_onehot_encoder(df1)

print(df1.head(10))
print(df1_new.head(10))

df2 = pd.DataFrame(df_bank_base[['age', 'duration']])
print(df2.head(10))

df3 = missing_imputation(df2)

print(df3.head(10))
"""

df_bank_base2 = data_preparation(df_bank_base)

print(df_bank_base.head(10))
print(df_bank_base.shape)
print(df_bank_base2.head(10))
print(df_bank_base2.shape)
