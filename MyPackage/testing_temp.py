if hyper_param_in is not None:
    hyper_param = hyper_param_in
else:







# Extreme Gradient Boosting hyper parameters
    if self.mode is None or self.mode == 'default':
        self.hyper_param = {
            'n_estimators': [70],
            'max_depth': [3],
            'subsample': [0.9],
            'colsample_bytree': [0.5], # 0.5 - 1
            'learning_rate': [0.1],
            'random_state': [10],
            'booster': ['gbtree']
        }
    elif self.mode == 'superfast':
        self.hyper_param = {
            'n_estimators': [70],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.9],
            'colsample_bytree': [0.5],  # 0.5 - 1
            'learning_rate': [0.1],
            'random_state': [10],
            'booster': ['gbtree']
        }
    elif self.mode == 'fast':
        self.hyper_param = {
            'n_estimators': [70, 80, 90],
            'max_depth': [3, 4, 5, 6],
            'subsample': [0.9],
            'colsample_bytree': [0.5],  # 0.5 - 1
            'learning_rate': [0.1, 0.2],
            'random_state': [10],
            'booster': ['gbtree']
        }
    elif self.mode == 'medium':
        self.hyper_param = {
            'n_estimators': [70, 80, 90, 100],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'subsample': [0.9],
            'colsample_bytree': [0.5],  # 0.5 - 1
            'learning_rate': [0.1, 0.2, 0.3],
            'random_state': [10],
            'booster': ['gbtree']
        }
    elif self.mode == 'slow':
        self.hyper_param = {
            'n_estimators': [50, 60, 70, 80, 90, 100],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'subsample': [0.8, 0.9],
            'colsample_bytree': [0.5, 0.6, 0.7],  # 0.5 - 1
            'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5],
            'random_state': [10],
            'booster': ['gbtree']
        }
    elif self.mode == 'superslow':
        self.hyper_param = {
            'n_estimators': [50, 60, 70, 80, 90, 100],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'subsample': [0.7, 0.8, 0.9],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8, 0.9, 1],  # 0.5 - 1
            'learning_rate': [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
            'random_state': [10],
            'booster': ['gbtree']
        }