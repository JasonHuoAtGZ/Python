"""Gradient Boosting Learner
this module inherits sklearn.ensemble.GradientBoostingClassifier by incorporating C-stat, Maximum KS & decile lift
which are the most important statistics to validate model prediction for campaign targeting / order ranking
Developed by Jason Huo
Email: jason_huo1983@hotmail.com
"""

import pandas as pd
import numpy as np
import time
import itertools
from sklearn.ensemble import GradientBoostingClassifier
from MyPackage.MLMeasurement import decile_lift
from MyPackage.MLMeasurement import maximum_ks
from MyPackage.MLMeasurement import c_stat


class GBLearner:

    def __init__(self, mode=None, df_train=None, df_valid=None, str_resp=None):
        # str_output='score_1', str_score=None

        self.mode = mode
        self.df_train = df_train.reset_index(drop=True)
        self.df_valid = df_valid.reset_index(drop=True)
        self.str_resp = str_resp
        self.str_score = 'score_1'

        self.train_base = self.df_train  # save the original training dataframe
        self.param = pd.DataFrame()  # full set of parameters
        self.best_param = pd.DataFrame()  # best parameters
        self.best_model = None  # trained model with the best parameter
        self.top_variable = None  # variable of importance

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
        # Program start time
        overall_start_time = time.time()

        # data preparation
        # training sample
        y_train = self.df_train[self.str_resp].values
        x_train = self.df_train.drop(self.str_resp, axis=1).values

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
        iteration_count = 0

        for p in hyper_param_combination:
            interation_start_time = time.time() # interation start time

            hyper_param_single = dict(zip(self.hyper_param.keys(), p))
            df_param_temp = pd.DataFrame({'estimator': ['GradientBoostingClassifier'], 'mode': [self.mode]})

            iteration_count = iteration_count + 1

            for i, item in enumerate(self.hyper_param.keys()):
                df_param_temp = pd.concat([df_param_temp, pd.DataFrame([p[i]], columns=[item])], axis=1)

            # model training
            clf = GradientBoostingClassifier(**hyper_param_single)

            # model fit
            clf.fit(x_train, y_train)

            # model validation
            p_predict = pd.DataFrame(clf.predict_proba(x_valid))
            p_actual = pd.DataFrame(self.df_valid[self.str_resp])
            p_valid = pd.concat([p_predict, p_actual], axis=1)
            p_valid.rename(columns={0: 'score_0', 1: 'score_1'}, inplace=True)

            p_rank = pd.DataFrame(pd.qcut(p_valid['score_1'], 10, labels=False, duplicates='drop'))
            p_rank.rename(columns={'score_1': 'decile'}, inplace=True)
            p_valid2 = pd.concat([p_valid, p_rank], axis=1)

            # get parameters and KPIs
            deciles = decile_lift(p_valid2, 'decile', self.str_score)
            maxks = maximum_ks(p_valid2, self.str_resp, self.str_score)
            cstat = c_stat(p_valid2, self.str_resp, self.str_score)

            df_param_temp = pd.concat([df_param_temp, deciles, maxks, cstat], axis=1)

            self.param = self.param.append(df_param_temp)

            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("# ", iteration_count, "interation")
            print(df_param_temp)
            print('Execution time in seconds: ' + str(time.time() - interation_start_time))

        # find best parameter, will enhance selection mode to define best parameter
        self.best_param = self.param[(self.param['c_stat'] == self.param['c_stat'].max())]
        self.best_param = pd.DataFrame(self.best_param.iloc[0])
        self.best_param = self.best_param.T


        # save the best model
        # need to revise and remove hard-coding later

        best_param_temp = self.best_param[self.hyper_param.keys()]
        best_param_temp2 = best_param_temp.to_dict(orient="index")

        # best_param_temp2 = {'n_estimators': 70, 'learning_rate': 0.1, 'min_samples_split': 50, 'min_samples_leaf': 25, 'max_depth': 3, 'max_features': 'sqrt', 'subsample': 0.9, 'random_state': 10, 'criterion': 'friedman_mse'}

        clf2 = GradientBoostingClassifier(**best_param_temp2[0]).fit(x_train, y_train)

        # self.best_model = .fit(x_train, y_train)

        """
        # save the variable of importance for the best model
        importance = pd.DataFrame(self.best_model.feature_importances_, columns=['importance'])
        variable = pd.DataFrame(self.df_train.drop(self.str_resp, axis=1).columns, columns=['variable'])
        features = pd.concat([importance, variable], axis=1)
        self.top_variable = features[(features['importance'] != 0)].sort_values(by=['importance'], ascending=0) \
            .reset_index(drop=True)
        
        """

        print('Overall execution time in seconds: ' + str(time.time() - overall_start_time))
        return best_param_temp2

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
