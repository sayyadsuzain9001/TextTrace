Copy your trained artifacts here:
- ensemble_config.json
- tfidf_logistic_regression.pkl
- tfidf_calibrated_linear_svm.pkl
- xgboost_linguistic.pkl

Note: the current adapter uses available LR/SVM probabilities and intentionally skips XGBoost until the exact linguistic feature extraction function from training is added. It never guesses the XGBoost feature schema.
