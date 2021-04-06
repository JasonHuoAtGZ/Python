"""
This module is a collection of data preparation for propensity modeling
"""
import pandas as pd
import numpy as np


def my_onehot_encoder(df_in, df_train=None):

    if df_train is None:
        df_char = pd.DataFrame(df_in.select_dtypes(include=object))
    else:
        df_char = pd.DataFrame(df_train.select_dtypes(include=object))

    df_out = pd.DataFrame()

    for col in df_char.columns:
        df_temp = pd.DataFrame(df_char[col].value_counts())
        dist_char_val = df_temp.index
        dist_char_val_cnt = df_temp.shape[0]

        for i in range(dist_char_val_cnt-1):
            df_out[col+'_'+str(i)] = np.where(df_in[col] == dist_char_val[i], 1, 0)

    return df_out


def missing_imputation(df_in, imput_val=0):
    # only keep numeric variables
    df_in = pd.DataFrame(df_in.select_dtypes(exclude=[object]))

    for col in df_in.columns:
        df_in[col] = np.where(df_in[col].isnull(), imput_val, df_in[col])

    return df_in


def data_preparation(df_in, df_train=None):
    df_num_imputed = missing_imputation(df_in)
    df_char_dummy = my_onehot_encoder(df_in, df_train)

    df_num_imputed['id_temp'] = df_num_imputed.index
    df_char_dummy['id_temp'] = df_char_dummy.index

    df_out = pd.merge(df_num_imputed, df_char_dummy, how='outer', on='id_temp')
    df_out = df_out.drop(['id_temp'], axis=1)

    return df_out


def my_dataframe_split(df_in, frac_train=0.5, frac_valid=0.5, random_state=1):

    if frac_train + frac_valid > 1 or frac_train > 1 or frac_valid > 1 or frac_train <= 0 or frac_valid <= 0:
        print("ErrorMessage: invalid split !!!")
        return

    elif frac_train + frac_valid == 1:
        frac_valid_adj = frac_valid/(1-frac_train)
        df_base_train = df_in.sample(frac=frac_train, random_state=random_state)
        df_base_valid = df_in.drop(df_base_train.index).sample(frac=frac_valid_adj, random_state=random_state)

        print("Input dimension: ", df_in.shape)
        print("Output - training sample dimension: ", df_base_train.shape)
        print("Output - validation sample dimension: ", df_base_valid.shape)

        df_base_train = df_base_train.reset_index(drop=True)
        df_base_valid = df_base_valid.reset_index(drop=True)

        return df_base_train, df_base_valid

    else:
        frac_valid_adj = frac_valid/(1-frac_train)
        df_base_train = df_in.sample(frac=frac_train, random_state=random_state)
        df_base_valid = df_in.drop(df_base_train.index).sample(frac=frac_valid_adj, random_state=random_state)
        df_base_holdout = df_in.drop(df_base_train.index)
        df_base_holdout = df_base_holdout.drop(df_base_valid.index)

        print("Input dimension: ", df_in.shape)
        print("Output - training sample dimension: ", df_base_train.shape)
        print("Output - validation sample dimension: ", df_base_valid.shape)
        print("Output - holdout sample dimension: ", df_base_holdout.shape)

        df_base_train = df_base_train.reset_index(drop=True)
        df_base_valid = df_base_valid.reset_index(drop=True)
        df_base_holdout = df_base_holdout.reset_index(drop=True)

        return df_base_train, df_base_valid, df_base_holdout


