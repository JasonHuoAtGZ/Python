# Gradient Boosting Learner
# this module inherits sklearn.ensemble.GradientBoostingClassifier by incorporating C-stat, Maximum KS & decile lift
# which are the most important statistics to validate model prediction for campaign targeting / order ranking
# Developed by Jason Huo
# Email: jason_huo1983@hotmail.com

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

class MyRFC(RandomForestClassifier):

    def __init__(self,
                 n_estimators=10,
                 criterion="gini",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features="auto",
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 bootstrap=True,
                 oob_score=False,
                 n_jobs=1,
                 random_state=None,
                 verbose=0,
                 warm_start=False,
                 class_weight=None,
                 df_in = None,
                 str_group = None,
                 str_score = None,
                 str_resp = None
                 ):

        super(RandomForestClassifier, self).__init__(
            base_estimator=DecisionTreeClassifier(),
            n_estimators=n_estimators,
            estimator_params=("criterion", "max_depth", "min_samples_split",
                              "min_samples_leaf", "min_weight_fraction_leaf",
                              "max_features", "max_leaf_nodes",
                              "min_impurity_decrease", "min_impurity_split",
                              "random_state"),
            bootstrap=bootstrap,
            oob_score=oob_score,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose,
            warm_start=warm_start,
            class_weight=class_weight)

        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split

        # additional initial variables added
        self.str_score='score_1'
        self.str_resp=str_resp
        self.str_group='group'

    def decile_lift(self, df_scored):
        # df_in: a dataframe that contains both group & response columns
        # str_group: a string that specifies grouping name
        # str_resp: a string that specifies response name

        if df_scored is None:
            print("Error: no scored file for decile_lift() !!!!")
            return
        else:
            # group by decile
            deciles=df_scored.groupby([self.str_group]).agg({self.str_resp: [np.size, np.mean]})
            deciles.columns=deciles.columns.droplevel(level=0)
            deciles['lift']=deciles['mean']/df_scored[self.str_resp].mean()
            pd_group=pd.DataFrame(deciles.index)
            deciles=deciles.reset_index(drop=True)
            deciles[self.str_group]=pd_group
            deciles['temp_lift']='decile_lift_'
            deciles['temp_count']='decile_count_'

            deciles['decile_lift']=deciles.temp_lift.str.cat(deciles[self.str_group].astype(str))
            deciles['decile_group']=deciles.temp_count.str.cat(deciles[self.str_group].astype(str))
            deciles=deciles.T

            # get decile count
            count_part=deciles[deciles.index=='size']
            count_part.columns=deciles.loc['decile_count']
            count_part_1=count_part.reset_index(drop=True)

            # get decile lift
            lift_part=deciles[deciles.index=='lift']
            lift_part.columns=deciles.loc['decile_lift']
            lift_part_1=lift_part.reset_index(drop=True)

            deciles=pd.concat([count_part_1, lift_part_1], axis=1)

            return deciles

    def maximum_ks(self, df_scored):
        # df_in: a dataframe that contains both group & response columns
        # str_group: a string that specifies grouping name
        # str_resp: a string that specifies response name
        if df_scored is None:
            print("Error: no scored file for maximum_ks() !!!!")
            return
        else:
            # calculate Maximum KS
            max_ks_sort=df_scored.sort_values([self.str_score], ascending=1)
            max_ks_sort['good']=max_ks_sort[self.str_resp]
            max_ks_sort['bad']=1-max_ks_sort[self.str_resp]
            max_ks_sort['t_resp1']=max_ks_sort.good.cumsum()
            max_ks_sort['t_resp0']=max_ks_sort.bad.cumsum()
            max_ks_sort['c_resp1']=max_ks_sort.t_resp1/max_ks_sort.good.sum()
            max_ks_sort['c_resp0']=max_ks_sort.t_resp0/max_ks_sort.bad.sum()
            max_ks_sort['max_ks']=abs(max_ks_sort.c_resp1-max_ks_sort.c_resp0)

            max_ks_score=max_ks_sort[(max_ks_sort.max_ks==max_ks_sort.max_ks.max())]

            max_ks_score=max_ks_score.rename(columns={self.str_score:'max_ks_score'})
            max_ks_score=max_ks_score[['max_ks_score','max_ks']]
            max_ks_score=max_ks_score.reset_index(drop=True)

            return max_ks_score

    def c_stat(self, df_scored):
        # df_in: a dataframe that contains both group & response columns
        # str_group: a string that specifies grouping name
        # str_resp: a string that specifies response name
        if df_scored is None:
            print("Error: no scored file for c_stat() !!!!")
            return
        else:
            # C-stat / concordant %
            c_stat_sort=df_scored.sort_values([self.str_score], ascending=0)
            c_stat_sort=c_stat_sort.reset_index(drop=True)
            c_stat_sort['rp']=c_stat_sort.index
            num_resp=c_stat_sort.response.sum()
            rp_sum=sum(c_stat_sort.response*c_stat_sort.rp)
            row_count=c_stat_sort[self.str_resp].count()

            c_stat=1-((rp_sum-0.5*num_resp*(num_resp+1))/(num_resp*(row_count-num_resp)))
            c_stat_score=pd.DataFrame([c_stat], columns=['c_stat'])

            return c_stat_score

    def get_result(self, df_scored):
        deciles=self.decile_lift(df_scored)
        maxks=self.maximum_ks(df_scored)
        cstat=self.c_stat(df_scored)

        param=pd.concat([
            pd.DataFrame(['Random Forest'], columns=['model_type']),
            pd.DataFrame([self.n_estimators], columns=['n_estimators']),
            pd.DataFrame([self.min_samples_split], columns=['min_samples_split']),
            pd.DataFrame([self.min_samples_leaf], columns=['min_samples_leaf']),
            pd.DataFrame([self.max_depth], columns=['max_depth']),
            pd.DataFrame([self.max_features], columns=['max_features']),
            pd.DataFrame([self.random_state], columns=['random_state']),
            pd.DataFrame([self.criterion], columns=['criterion']),
            pd.DataFrame([self.bootstrap], columns=['bootstrap']),
            pd.DataFrame([self.n_jobs], columns=['n_jobs']),
            cstat,
            maxks,
            deciles], axis=1)

        return param

class RFLearner:
    def __init__(self, mode=None, df_train=None, df_valid=None, str_resp=None):
        self.df_train=df_train
        self.mode=mode
        self.df_valid=df_valid
        self.str_resp=str_resp  # response variable

        self.train_base=self.df_train  # save the original training dataframe
        self.param=pd.DataFrame()  # full set of parameters
        self.best_param=pd.DataFrame()  # best parameters
        self.best_model=None  # trained model with the best parameter
        self.feature_importances_=None  # variable of importance

        # to grid search all parameters and return the full set of parameters and KPIs
        if self.mode is None or self.mode=='default':
            self.n_estimators=[]
            self.learning_rate=[]
            self.min_samples_split=[]
            self.min_samples_leaf=[]
            self.max_depth=[]
            self.max_features=[]
            self.subsample=[]
            self.random_state=[]
            self.criterion=[]
        elif self.mode=='superfast':
            self.n_estimators=[]
            self.learning_rate=[]
            self.min_samples_split=[]
            self.min_samples_leaf=[]
            self.max_depth=[]
            self.max_features=[]
            self.subsample=[]
            self.random_state=[]
            self.criterion=[]
        elif self.mode=='fast':
            self.n_estimators=[]
            self.learning_rate=[]
            self.min_samples_split=[]
            self.min_samples_leaf=[]
            self.max_depth=[]
            self.max_features=[]
            self.subsample=[]
            self.random_state=[]
            self.criterion=[]
        elif self.mode=='medium':
            self.n_estimators=[]
            self.learning_rate=[]
            self.min_samples_split=[]
            self.min_samples_leaf=[]
            self.max_depth=[]
            self.max_features=[]
            self.subsample=[]
            self.random_state=[]
            self.criterion=[]
        elif self.mode=='slow':
            self.n_estimators=[]
            self.learning_rate=[]
            self.min_samples_split=[]
            self.min_samples_leaf=[]
            self.max_depth=[]
            self.max_features=[]
            self.subsample=[]
            self.random_state=[]
            self.criterion=[]
        elif self.mode=='superslow':
            self.n_estimators=[]
            self.learning_rate=[]
            self.min_samples_split=[]
            self.min_samples_leaf=[]
            self.max_depth=[]
            self.max_features=[]
            self.subsample=[]
            self.random_state=[]
            self.criterion=[]

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
        for i in self.n_estimators:
            for j in self.min_samples_split:
                for k in self.min_samples_leaf:
                    for l in self.max_depth:
                        for m in self.max_features:
                            for n in self.random_state:
                                for o in self.criterion:
                                    for p in self.bootstrap:
                                        for q in self.n_jobs:
                                            # model training
                                            clf=MyRFC(
                                                n_estimators       =i,
                                                min_samples_split  =j,
                                                min_samples_leaf   =k,
                                                max_depth          =l,
                                                max_features       =m,
                                                random_state       =n,
                                                criterion          =o,
                                                bootstrap          =p,
                                                n_jobs             =q,
                                                str_resp           =self.str_resp
                                            ).fit(x_train, y_train)

                                            # model validation
                                            p_predict=pd.DataFrame(clf.predict_proba(x_valid))
                                            p_actual=self.df_valid[self.str_resp]
                                            p_valid=pd.concat([p_predict, p_actual], axis=1)
                                            p_valid.rename(columns={0:'score_0',1:'score_1'}, inplace=True)

                                            p_rank=pd.DataFrame(pd.qcut(p_valid['score_1'], 10, labels=False, duplicates='drop'))
                                            p_rank.rename(columns={'score_1':'group'}, inplace=True)
                                            p_valid2=pd.concat([p_valid, p_rank], axis=1)

                                            # get parameters and KPIs
                                            df_param=clf.get_result(p_valid2)
                                            self.param=self.param.append(df_param)

        # find best parameter, will enhance selection mode to define best parameter
        self.best_param=self.param[(self.param['c_stat']==self.param['c_stat'].max())]
        self.best_param=pd.DataFrame(self.best_param.iloc[0])
        self.best_param=self.best_param.T

        # save the best model
        self.best_model=MyGBC(
            n_estimators=self.best_param['n_estimators'].values[0],
            min_samples_split=self.best_param['min_samples_split'].values[0],
            min_samples_leaf=self.best_param['min_samples_leaf'].values[0],
            max_depth=self.best_param['max_depth'].values[0],
            max_features=self.best_param['max_features'].values[0],
            random_state=self.best_param['random_state'].values[0],
            criterion=self.best_param['criterion'].values[0],
            bootstrap=self.best_param['bootstrap'].values[0],
            n_jobs=self.best_param['n_jobs'].values[0],
            str_resp=self.str_resp
        )

        # save the variable of importance for the best model
        importance=pd.DataFrame(self.best_model.feature_importances_, columns=['importance'])
        variable=pd.DataFrame(self.df_train.drop(self.str_resp, axis=1).columns, columns=['variable'])
        features=pd.concat([importance, variable], axis=1)
        self.feature_importances_=features[(features['importance']!=0)].sort_values(by=['importance'],ascending=0)\
            .reset_index(drop=True)

        return

    def _validating(self, df_to_valid):
        return

    def _validating_out(self, df_to_valid):
        return

    def _score(self, df_to_score, str_ID):
        if df_to_score is None:
            print("No scoring file is specified !!!")
            return
        elif len(df_to_score)==0:
            print("No records found from scoring file !!!")
            return
        else:
            df_ID=pd.DataFrame(df_to_score[str_ID])
            to_score1=df_to_score.drop([str_ID], axis=1)
            to_score2=pd.DataFrame(self.best_model.predict_proba(to_score1.values))
            df_scored=pd.concat([df_ID, to_score2], axis=1)
            df_scored=df_scored.rename(columns={0:'score_0', 1:'score_1'})
            return df_scored
