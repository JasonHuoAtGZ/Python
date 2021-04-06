import pandas as pd
import numpy as np
import time

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)


def decile_lift(df_scored, str_group, str_resp):

    # df_in: a dataframe that contains both group & response columns
    # str_group: a string that specifies grouping name
    # str_resp: a string that specifies response name

    if df_scored is None:
        print("Error: no scored file for decile_lift() !!!!")
        return
    else:

        # group by decile and calculate response rate and lift
        deciles = df_scored.groupby([str_group]).agg({str_resp: [np.size, np.mean]})
        deciles.columns = deciles.columns.droplevel(level=0)
        deciles['lift'] = deciles['mean']/df_scored[str_resp].mean()
        pd_group = pd.DataFrame(deciles.index)
        deciles = deciles.reset_index(drop=True)
        deciles[str_group] = pd_group
        deciles['temp_lift'] = 'lift_'
        deciles['temp_count'] = 'decile_'
        deciles['temp_resp'] = 'resp_'
        deciles['lift2'] = deciles.temp_lift.str.cat(deciles[str_group].astype(str))
        deciles['group'] = deciles.temp_count.str.cat(deciles[str_group].astype(str))
        deciles['resp'] = deciles.temp_resp.str.cat(deciles[str_group].astype(str))

        deciles = deciles.T

        # get decile count
        count_part = deciles[deciles.index == 'size']
        count_part.columns = deciles.loc['group']
        count_part_1 = count_part.reset_index(drop=True)

        # get decile response rate
        resp_part = deciles[deciles.index == 'mean']
        resp_part.columns = deciles.loc['resp']
        resp_part_1 = resp_part.reset_index(drop=True)

        # get decile lift
        lift_part = deciles[deciles.index == 'lift']
        lift_part.columns = deciles.loc['lift2']
        lift_part_1 = lift_part.reset_index(drop=True)

        deciles = pd.concat([count_part_1, resp_part_1, lift_part_1], axis=1)

        return deciles


def maximum_ks(df_scored, str_resp, str_score):

    # df_in: a dataframe that contains both group & response columns
    # str_group: a string that specifies grouping name
    # str_resp: a string that specifies response name

    if df_scored is None:
        print("Error: no scored file for maximum_ks() !!!!")
    else:
        # calculate Maximum KS
        max_ks_sort = df_scored.sort_values([str_score], ascending=1)
        max_ks_sort['good'] = max_ks_sort[str_resp]
        max_ks_sort['bad'] = 1-max_ks_sort[str_resp]
        max_ks_sort['t_resp1'] = max_ks_sort.good.cumsum()
        max_ks_sort['t_resp0'] = max_ks_sort.bad.cumsum()
        max_ks_sort['c_resp1'] = max_ks_sort.t_resp1/max_ks_sort.good.sum()
        max_ks_sort['c_resp0'] = max_ks_sort.t_resp0/max_ks_sort.bad.sum()
        max_ks_sort['max_ks'] = abs(max_ks_sort.c_resp1-max_ks_sort.c_resp0)

        max_ks_score = max_ks_sort[(max_ks_sort.max_ks==max_ks_sort.max_ks.max())]

        max_ks_score = max_ks_score.rename(columns={str_score:'max_ks_score'})
        max_ks_score = max_ks_score[['max_ks_score', 'max_ks']]
        max_ks_score = max_ks_score.reset_index(drop=True)

        return max_ks_score


def c_stat(df_scored, str_resp, str_score):
    # df_in: a dataframe that contains both group & response columns
    # str_group: a string that specifies grouping name
    # str_resp: a string that specifies response name
    if df_scored is None:
        print("Error: no scored file for c_stat() !!!!")
        return
    else:
        # C-stat / concordant %
        c_stat_sort = df_scored.sort_values([str_score], ascending=0)
        c_stat_sort = c_stat_sort.reset_index(drop=True)
        c_stat_sort['rp'] = c_stat_sort.index
        num_resp = c_stat_sort.response.sum()
        rp_sum = sum(c_stat_sort.response*c_stat_sort.rp)
        row_count = c_stat_sort[str_resp].count()

        c_stat = 1-((rp_sum-0.5*num_resp*(num_resp+1))/(num_resp*(row_count-num_resp)))
        c_stat_score = pd.DataFrame([c_stat], columns=['c_stat'])

        return c_stat_score

"""
def get_result(df_scored):
    deciles = decile_lift(df_scored)
    maxks = maximum_ks(df_scored)
    cstat = c_stat(df_scored)

    param=pd.concat([
        pd.DataFrame([n_estimators], columns=['n_estimators']),
        pd.DataFrame([learning_rate], columns=['learning_rate']),
        pd.DataFrame([min_samples_split], columns=['min_samples_split']),
        pd.DataFrame([min_samples_leaf], columns=['min_samples_leaf']),
        pd.DataFrame([max_depth], columns=['max_depth']),
        pd.DataFrame([max_features], columns=['max_features']),
        pd.DataFrame([subsample], columns=['subsample']),
        pd.DataFrame([random_state], columns=['random_state']),
        pd.DataFrame([criterion], columns=['criterion']),
        cstat,
        maxks,
        deciles], axis=1)

    return param
"""