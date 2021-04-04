from xgboost import XGBClassifier

print(xgboost.__version__)

clf = xgb.XGBClassifier()
clf.fit
clf.predict_proba()
clf.feature_importances_
