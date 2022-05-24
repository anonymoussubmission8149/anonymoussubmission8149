from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score as acc
def auc(y, probas):
    from sklearn.metrics import roc_auc_score
    if probas.shape[-1] == 2:
        return roc_auc_score(y, probas[:,-1])
    else:
        return roc_auc_score(y, probas, multi_class ="ovr")
    
#Catboost Catfast params
catfast_params = {"iterations":100,
                "depth":3,
                "iterations":100,
                "subsample": 0.1,
                "max_bin": 32,
                "bootstrap_type":"Bernoulli",
                "task_type":"GPU"}

def run_xgb(X_train, X_test, y_train, y_test, method_name, seed, hyper_parameters = {}, regression = True):
    try:
        if regression:
            if method_name == "XGBoost":
                from xgboost import XGBRegressor as XGB
                model = XGB(random_state = seed, objective ='reg:squarederror', verbose = False, **hyper_parameters)

            elif method_name in ["CAT", "CATfast"]:
                from catboost import CatBoostRegressor as CAT
                updated_hyper_parameters = {}
                if method_name == "CATfast":
                    updated_hyper_parameters.update(catfast_params)
                updated_hyper_parameters.update(hyper_parameters)
                model = CAT(random_seed=seed, logging_level='Silent', **updated_hyper_parameters)                

            elif method_name == "LGBM":
                from lightgbm.sklearn import LGBMRegressor as LGBM
                model = LGBM(random_state=seed, **hyper_parameters)
        else:
            if method_name == "XGBoost":
                from xgboost import XGBClassifier as XGB
                model = XGB(random_state = seed, verbose = False, **hyper_parameters)

            elif method_name in ["CAT", "CATfast"]:
                from catboost import CatBoostClassifier as CAT
                updated_hyper_parameters = {}
                if method_name == "CATfast":
                    updated_hyper_parameters.update(catfast_params)
                updated_hyper_parameters.update(hyper_parameters)
                model = CAT(random_seed=seed, logging_level='Silent', **updated_hyper_parameters)

            elif method_name == "LGBM":
                from lightgbm.sklearn import LGBMClassifier as LGBM
                model = LGBM(random_state=seed, **hyper_parameters)

        model.fit(X_train,y_train)
        prediction = model.predict(X_test) if regression else model.predict_proba(X_test)
        results = [r2_score(y_test, prediction)] if regression else [acc(y_test, model.predict(X_test)), auc(y_test,prediction)]
        if regression:
            results += [r2_score(y_train, model.predict(X_train))]
        else:
            results += [acc(y_train, model.predict(X_train)), auc(y_train,model.predict_proba(X_train))]
        success = True
    except: 
        prediction = None
        results = [None, None] if regression else [None, None, None, None]
        success = False
    return success, results, prediction