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

df_temp = hyper_param.keys()

print(df_temp)