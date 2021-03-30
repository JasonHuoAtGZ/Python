
"""
hyper_param = {
    'n_estimators': [50, 60, 70, 80, 90, 100],
    'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
    'min_samples_split': [50, 100, 200, 500, 1000],
    'min_samples_leaf': [25, 50, 100, 250, 500],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'max_features': ['sqrt'],
    'subsample': [0.8, 0.9],
    'random_state': [10],
    'criterion': ['friedman_mse']
}

print(hyper_param)
print(hyper_param.keys())
print(len(list(hyper_param.keys())))
print(hyper_param['min_samples_leaf'])




mode = 'superslow'

if mode is None or mode == 'default':
    hyper_param = {
        'n_estimators': [70],
        'learning_rate': [0.1],
        'min_samples_split': [50],
        'min_samples_leaf': [25],
        'max_depth': [3],
        'max_features': ['sqrt'],
        'subsample': [0.9],
        'random_state': [10],
        'criterion': ['friedman_mse']
    }
elif mode == 'superfast':
    hyper_param = {
        'n_estimators': [70],
        'learning_rate': [0.1],
        'min_samples_split': [50, 100],
        'min_samples_leaf': [25, 50],
        'max_depth': [3, 4, 5, 6],
        'max_features': ['sqrt'],
        'subsample': [0.9],
        'random_state': [10],
        'criterion': ['friedman_mse']
    }
elif mode == 'fast':
    hyper_param = {
        'n_estimators': [70, 80, 90],
        'learning_rate': [0.1, 0.2],
        'min_samples_split': [50, 100, 200],
        'min_samples_leaf': [25, 50, 100],
        'max_depth': [3, 4, 5, 6],
        'max_features': ['sqrt'],
        'subsample': [0.9],
        'random_state': [10],
        'criterion': ['friedman_mse']
    }
elif mode == 'medium':
    hyper_param = {
        'n_estimators': [70, 80, 90, 100],
        'learning_rate': [0.1, 0.2, 0.3],
        'min_samples_split': [50, 100, 200, 500],
        'min_samples_leaf': [25, 50, 100, 250],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'max_features': ['sqrt'],
        'subsample': [0.9],
        'random_state': [10],
        'criterion': ['friedman_mse']
    }
elif mode == 'slow':
    hyper_param = {
        'n_estimators': [50, 60, 70, 80, 90, 100],
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
        'min_samples_split': [50, 100, 200, 500, 1000],
        'min_samples_leaf': [25, 50, 100, 250, 500],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'max_features': ['sqrt'],
        'subsample': [0.8, 0.9],
        'random_state': [10],
        'criterion': ['friedman_mse']
    }
elif mode == 'superslow':
    hyper_param = {
        'n_estimators': [50, 60, 70, 80, 90, 100],
        'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
        'min_samples_split': [50, 100, 200, 500, 1000],
        'min_samples_leaf': [25, 50, 100, 250, 500],
        'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
        'max_features': ['sqrt'],
        'subsample': [0.7, 0.8, 0.9],
        'random_state': [10],
        'criterion': ['friedman_mse']
    }




import itertools
from MyPackage.DataExplore import display_string_with_quote


# initiate the list to store parameter combination
list = []

for item in hyper_param.keys():
    list.append(hyper_param[item])


# initiate the parameter list with proper format
str_hyper_param = ''

# create combination of all parameters
hyper_param_combination = itertools.product(*list)

# create the parameters list for direct model fit
combination_count = 0
for p in hyper_param_combination:
    str_hyper_param = ''
    for i, item in enumerate(hyper_param.keys()):
        str_hyper_param = str_hyper_param + str(item) + ' = ' + str(display_string_with_quote(p[i])) + ' ,'

    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    str_hyper_param = str_hyper_param.rstrip(str_hyper_param[-1])
    print(str_hyper_param)

    combination_count = combination_count + 1

print("# of combination: ", combination_count)

"""


import pandas as pd
from MyPackage.MLMeasurement import decile_lift
from MyPackage.MLMeasurement import maximum_ks
from MyPackage.MLMeasurement import c_stat

hyper_param = {
    'n_estimators': [50, 60],
    'learning_rate': [0.1, 0.2],
    'min_samples_split': [50, 100],
    'min_samples_leaf': [25, 50],
    'max_depth': [3, 4],
    'max_features': ['sqrt'],
    'subsample': [0.8, 0.9],
    'random_state': [10],
    'criterion': ['friedman_mse']
}

df_scored = pd.read_csv("C:/Users/jason/Working_Folder/03_Python/scored_file.csv")

import itertools
from MyPackage.DataExplore import display_string_with_quote


# initiate the list to store parameter combination
list = []

for item in hyper_param.keys():
    list.append(hyper_param[item])


# initiate the parameter list with proper format
str_hyper_param = ''

# create combination of all parameters
hyper_param_combination = itertools.product(*list)

# create the parameters list for direct model fit
combination_count = 0
for p in hyper_param_combination:
    str_hyper_param = ''
    df_temp = pd.DataFrame()
    for i, item in enumerate(hyper_param.keys()):
        str_hyper_param = str_hyper_param + str(item) + ' = ' + str(display_string_with_quote(p[i])) + ' ,'

        df_temp = pd.concat([df_temp, pd.DataFrame([p[i]], columns=[item])], axis=1)

    deciles = decile_lift(df_scored, 'response', 'score')
    maxks = maximum_ks(df_scored, 'response', 'score')
    cstat = c_stat(df_scored, 'response', 'score')

    df_temp = pd.concat([df_temp, deciles, maxks, cstat], axis=1)
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
    str_hyper_param = str_hyper_param.rstrip(str_hyper_param[-1])
    print(str_hyper_param)
    print(df_temp)


    # put all parameters and KPIs in a row in dataframe
    """


    df_param = pd.DataFrame()
    for item in hyper_param.keys():
        # df_param = pd.concat([df_param, ], axis=1)
        df_temp = pd.DataFrame([], columns=)
    """

"""
def get_result(self, df_scored):

    deciles = decile_lift(df_scored)
    maxks = maximum_ks(df_scored)
    cstat = c_stat(df_scored)

    df_param = pd.DataFrame()

    for item in ls_param:
        param = pd.concat([param,


                           ], axis=1)

    param=pd.concat([
        pd.DataFrame([self.n_estimators], columns=['n_estimators']),
        pd.DataFrame([self.learning_rate], columns=['learning_rate']),
        pd.DataFrame([self.min_samples_split], columns=['min_samples_split']),
        pd.DataFrame([self.min_samples_leaf], columns=['min_samples_leaf']),
        pd.DataFrame([self.max_depth], columns=['max_depth']),
        pd.DataFrame([self.max_features], columns=['max_features']),
        pd.DataFrame([self.subsample], columns=['subsample']),
        pd.DataFrame([self.random_state], columns=['random_state']),
        pd.DataFrame([self.criterion], columns=['criterion']),
        cstat,
        maxks,
        deciles], axis=1)

    return param

"""
