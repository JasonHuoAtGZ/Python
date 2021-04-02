"""Gradient Boosting Learner
this module inherits sklearn.ensemble.GradientBoostingClassifier by incorporating C-stat, Maximum KS & decile lift
which are the most important statistics to validate model prediction for campaign targeting / order ranking
Developed by Jason Huo
Email: jason_huo1983@hotmail.com
"""

import pandas as pd
import numpy as np
import itertools
from sklearn.ensemble import GradientBoostingClassifier
from MyPackage.MLMeasurement import decile_lift
from MyPackage.MLMeasurement import maximum_ks
from MyPackage.MLMeasurement import c_stat
from MyPackage.DataExplore import display_string_with_quote


class GBLearner(GradientBoostingClassifier):
    def __init__(self, *, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False,
                 validation_fraction=0.1, n_iter_no_change=None, tol=1e-4,
                 ccp_alpha=0.0, mode=None, df_train=None, df_valid=None, str_resp=None, str_score=None):
        # str_output='score_1', str_resp, str_group)

        super().__init__(
            loss=loss, learning_rate=learning_rate, n_estimators=n_estimators,
            criterion=criterion, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_depth=max_depth, init=init, subsample=subsample,
            max_features=max_features,
            random_state=random_state, verbose=verbose,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            min_impurity_split=min_impurity_split,
            warm_start=warm_start, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change, tol=tol, ccp_alpha=ccp_alpha)

        self.df_train = df_train
        self.mode = mode
        self.df_valid = df_valid
        self.str_resp = str_resp  # response variable
        self.str_score = 'score_1'

        self.train_base = self.df_train  # save the original training dataframe
        self.param = pd.DataFrame()  # full set of parameters
        self.best_param = pd.DataFrame()  # best parameters
        self.best_model = None  # trained model with the best parameter
        self.variable_importances = None  # variable of importance

        # to grid search all parameters and return the full set of parameters and KPIs
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

        if self.df_train is None:
            print("Training Sample is missing !")

        if self.df_valid is None:
            print("Validation sample is missing !")

        if self.str_resp is None:
            print("Response variable is not specified !")


    def _training(self):
        # data preparation
        # training sample
        y_train=self.df_train[self.str_resp].values
        x_train=self.df_train.drop(self.str_resp, axis=1).values

        # validation sample
        y_valid = self.df_valid[self.str_resp].values
        x_valid = self.df_valid.drop(self.str_resp, axis=1).values

        # grid search on given parameter mode

        # initiate the list to store parameter combination
        list_param = []

        for item in self.hyper_param.keys():
            list_param.append(self.hyper_param[item])

        # create combination of all parameters
        hyper_param_combination = itertools.product(*list_param)

        # create the parameters list for direct model fit
        for p in hyper_param_combination:
            str_hyper_param = ''
            df_param_temp = pd.DataFrame()
            for i, item in enumerate(self.hyper_param.keys()):
                str_hyper_param = str_hyper_param + str(item) + ' = ' + str(display_string_with_quote(p[i])) + ' ,'
                df_param_temp = pd.concat([df_param_temp, pd.DataFrame([p[i]], columns=[item])], axis=1)

            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            str_hyper_param = str_hyper_param.rstrip(str_hyper_param[-1])
            print(str_hyper_param)
            # model training
            clf = GBLearner(str_hyper_param)

            # model training
            clf.fit(x_train, y_train)

            # model validation
            p_predict = pd.DataFrame(clf.predict_proba(x_valid))
            p_actual = self.df_valid[self.str_resp]
            p_valid = pd.concat([p_predict, p_actual], axis=1)
            p_valid.rename(columns={0: 'score_0', 1: 'score_1'}, inplace=True)

            p_rank = pd.DataFrame(pd.qcut(p_valid['score_1'], 10, labels=False, duplicates='drop'))
            p_rank.rename(columns={'score_1': 'group'}, inplace=True)
            p_valid2 = pd.concat([p_valid, p_rank], axis=1)

            # get parameters and KPIs
            deciles = decile_lift(p_valid2, self.str_resp, self.str_score)
            maxks = maximum_ks(p_valid2, self.str_resp, self.str_score)
            cstat = c_stat(p_valid2, self.str_resp, self.str_score)

            df_param_temp = pd.concat([df_param_temp, deciles, maxks, cstat], axis=1)

            self.param = self.param.append(df_param_temp)

        # find best parameter, will enhance selection mode to define best parameter
        self.best_param = self.param[(self.param['c_stat'] == self.param['c_stat'].max())]
        self.best_param = pd.DataFrame(self.best_param.iloc[0])
        self.best_param = self.best_param.T

        # save the best model
        # need to revise and remove hard-coding later
        self.best_model = GBLearner(
            n_estimators=self.best_param['n_estimators'].values[0],
            learning_rate=self.best_param['learning_rate'].values[0],
            min_samples_split=self.best_param['min_samples_split'].values[0],
            min_samples_leaf=self.best_param['min_samples_leaf'].values[0],
            max_depth=self.best_param['max_depth'].values[0],
            max_features=self.best_param['max_features'].values[0],
            subsample=self.best_param['subsample'].values[0],
            random_state=self.best_param['random_state'].values[0],
            criterion=self.best_param['criterion'].values[0]
            # str_resp=self.str_resp
        )

        # save the variable of importance for the best model
        importance = pd.DataFrame(self.best_model.feature_importances_, columns=['importance'])
        variable = pd.DataFrame(self.df_train.drop(self.str_resp, axis=1).columns, columns=['variable'])
        features = pd.concat([importance, variable], axis=1)
        self.variable_importances = features[(features['importance'] != 0)].sort_values(by=['importance'], ascending=0) \
            .reset_index(drop=True)

        return

    def _validating(self, df_to_valid):
        # output validation results
        x_valid=df_to_valid.drop(self.str_resp, axis=1).values

        p_predict=pd.DataFrame(self.best_model.predict_prob(x_valid))
        p_actual=pd.DataFrame(df_to_valid[self.str_resp])
        p_valid=pd.concat([p_predict, p_actual], axis=1)
        p_valid.rename(columns={0:'score_0', 1:'score_1'}, inplace=True)
        p_rank=pd.DataFrame(pd.qcut(p_valid['score_1'], 10, labels=False))
        p_rank.rename(columns={'score_1':'group'}, inplace=True)
        p_valid2=pd.concat([p_valid, p_rank], axis=1)

        return p_valid2

    def _validating_out(self, df_to_valid):
        # output validation sample
        x_valid=df_to_valid.drop(self.str_resp, axis=1).values

        p_predict=pd.DataFrame(self.best_model.predict_prob(x_valid))
        p_actual=pd.DataFrame(df_to_valid[self.str_resp])
        p_valid=pd.concat([p_predict, p_actual], axis=1)
        p_valid.rename(columns={0:'score_0', 1:'score_1'}, inplace=True)
        p_rank=pd.DataFrame(pd.qcut(p_valid['score_1'], 10, labels=False))
        p_rank.rename(columns={'score_1':'group'}, inplace=True)
        p_valid2=pd.concat([p_valid, p_rank], axis=1)

        return p_valid2

