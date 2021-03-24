"""
This module is a collection of data exploration for dataframe only
"""
import pandas as pd
import numpy as np
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def data_explore_num(df_in):

    df_smy = df_in.describe(percentiles=[0.01, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50 /
                                                0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99])

    df_smy_cnt = df_smy.loc['count']
    record_cnt = df_in.shape[0]
    df_smy_cnt = pd.DataFrame(df_smy_cnt / record_cnt)
    df_smy_cnt = df_smy_cnt.rename(columns={'count':'C_rate'})
    df_smy_cnt = df_smy_cnt.transpose()

    df_smy_cnt = df_smy_cnt.append(df_smy)

    print("======================================================================================>>")
    print("==================================Percentiles=========================================>>")
    print(df_smy_cnt)
    print("======================================================================================>>")
    print("\n")
    print("\n")

    return df_smy_cnt


def data_explore_char(df_in):
    """
    numeric_type = ['int16', 'int32', 'int64', 'float32', 'float64']
    df_in = df_in.select_dtypes(exclude=numeric_type)
    """
    df_in = df_in.select_dtypes(include=[object])

    record_cnt = df_in.shape[0]

    df_smy_cnt = pd.DataFrame(columns=['id_temp'])

    for col in df_in.columns:
        df_temp = pd.DataFrame(df_in[col].value_counts())
        df_temp[col+'_val'] = df_temp.index
        df_temp[col+'_p'] = df_temp[col] / record_cnt

        df_temp = df_temp.rename(columns={col:col+'_cnt'})
        df_temp = df_temp.reset_index(drop=True)

        df_temp['id_temp'] = df_temp.index

        df_smy_cnt = pd.merge(df_smy_cnt, df_temp, how='outer', on='id_temp')

    df_smy_cnt = df_smy_cnt.drop(['id_temp'], axis=1)

    print("======================================================================================>>")
    print("==================================Distribution========================================>>")
    print(df_smy_cnt)
    print("======================================================================================>>")
    print("\n")
    print("\n")

    return df_smy_cnt


def my_onehot_encoder(df_in):

    df_char = pd.DataFrame(df_in.select_dtypes(include=[object]))
    df_out = pd.DataFrame()

    for col in df_char.columns:
        df_temp = pd.DataFrame(df_char[col].value_counts())
        dist_char_val = df_temp.index
        dist_char_val_cnt = df_temp.shape[0]

        """
        print(col)
        print(dist_char_val)
        print(dist_char_val_cnt)
        """

        for i in range(dist_char_val_cnt-1):
            df_out[col+'_'+str(i)] = np.where(df_char[col] == dist_char_val[i], 1, 0)

    return df_out


def missing_imputation(df_in, imput_val=0):
    # only keep numeric variables
    df_in = pd.DataFrame(df_in.select_dtypes(exclude=[object]))

    for col in df_in.columns:
        df_in[col] = np.where(df_in[col].isnull(), imput_val, df_in[col])

    return df_in


def data_preparation(df_in):
    df_num_imputed = missing_imputation(df_in)
    df_char_dummy = my_onehot_encoder(df_in)

    df_num_imputed['id_temp'] = df_num_imputed.index
    df_char_dummy['id_temp'] = df_char_dummy.index

    df_out = pd.merge(df_num_imputed, df_char_dummy, how='outer', on='id_temp')
    df_out = df_out.drop(['id_temp'], axis=1)

    return df_out