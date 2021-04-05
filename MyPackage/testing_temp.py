if hyper_param_in is not None:
    hyper_param = hyper_param_in
else:



# Gradient Boosting hyper parameters

if self.mode is None or self.mode == 'default':
    self.hyper_param = {
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
elif self.mode == 'superfast':
    self.hyper_param = {
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
elif self.mode == 'fast':
    self.hyper_param = {
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
elif self.mode == 'medium':
    self.hyper_param = {
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
elif self.mode == 'slow':
    self.hyper_param = {
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
elif self.mode == 'superslow':
    self.hyper_param = {
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

# Random Forest hyper parameters
    if self.mode is None or self.mode == 'default':
        self.hyper_param = {
            'n_estimators': [70],
            'max_depth': [3],
            'min_samples_split': [50],
            'min_samples_leaf': [25],
            'max_features': ['auto'],
            'bootstrap': [True],
            'criterion': ['gini'],
            'random_state': [10]
        }
    elif self.mode == 'superfast':
        self.hyper_param = {
            'n_estimators': [70],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [50, 100],
            'min_samples_leaf': [25, 50],
            'max_features': ['auto'],
            'bootstrap': [True],
            'criterion': ['gini'],
            'random_state': [10]
        }
    elif self.mode == 'fast':
        self.hyper_param = {
            'n_estimators': [70, 80, 90],
            'max_depth': [3, 4, 5, 6],
            'min_samples_split': [50, 100, 200],
            'min_samples_leaf': [25, 50, 100],
            'max_features': ['auto'],
            'bootstrap': [True],
            'criterion': ['gini'],
            'random_state': [10]
        }
    elif self.mode == 'medium':
        self.hyper_param = {
            'n_estimators': [70, 80, 90, 100],
            'max_depth': [3, 4, 5, 6, 7, 8],
            'min_samples_split': [50, 100, 200, 500],
            'min_samples_leaf': [25, 50, 100, 250],
            'max_features': ['auto'],
            'bootstrap': [True],
            'criterion': ['gini'],
            'random_state': [10]
        }
    elif self.mode == 'slow':
        self.hyper_param = {
            'n_estimators': [50, 60, 70, 80, 90, 100],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [50, 100, 200, 500, 1000],
            'min_samples_leaf': [25, 50, 100, 250, 500],
            'max_features': ['auto'],
            'bootstrap': [True],
            'criterion': ['gini'],
            'random_state': [10]
        }
    elif self.mode == 'superslow':
        self.hyper_param = {
            'n_estimators': [50, 60, 70, 80, 90, 100],
            'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'min_samples_split': [50, 100, 200, 500, 1000],
            'min_samples_leaf': [25, 50, 100, 250, 500],
            'max_features': ['auto'],
            'bootstrap': [True],
            'criterion': ['gini'],
            'random_state': [10]
        }

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