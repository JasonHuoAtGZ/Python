# Gradient Boosting Learner
# this module inherits sklearn.ensemble.GradientBoostingClassifier by incorporating C-stat, Maximum KS & decile lift
# which are the most important statistics to validate model prediction for campaign targeting / order ranking
# Developed by Jason Huo
# Email: jason_huo1983@hotmail.com

import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

class MyGBC(GradientBoostingClassifier):

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100,
                 subsample=1.0, criterion='friedman_mse', min_samples_split=2,
                 min_samples_leaf=1, min_weight_fraction_leaf=0.,
                 max_depth=3, min_impurity_decrease=0.,
                 min_impurity_split=None, init=None,
                 random_state=None, max_features=None, verbose=0,
                 max_leaf_nodes=None, warm_start=False, presort='auto',
                 df_in=None, str_group=None, str_score=None, str_resp=None
                 ):

        super(GradientBoostingClassifier, self).__init__(
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
            warm_start=warm_start,
            presort=presort)

        self.str_score='score_1'
        self.str_resp=str_resp
        self.str_group='group'

    def decile_lift(self, df_scored):
        # df_in: a dataframe that contains both group & response columns
        # str_group: a string that specifies grouping name
        # str_resp: a string that specifies response name

        if df_scored is None:
            print("Error: no scored file for decile_lift() !!!!")
        else:
            # group by decile
            deciles=df_scored.groupby([self.str_group]).agg({self.str_resp: [np.size(), np.mean]})


