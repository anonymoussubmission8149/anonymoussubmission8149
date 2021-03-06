import numpy as np
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score as acc
def auc(y, probas):
    from sklearn.metrics import roc_auc_score
    if probas.shape[-1] == 2:
        return roc_auc_score(y, probas[:,-1])
    else:
        return roc_auc_score(y, probas, multi_class ="ovr")

deterministic_methods = ["Intercept",
                             "Ridge",
                             "Lasso",
                             "Enet",
                             "LinearRidge"
                             "CART",
                             "XCART",
                             "Kernel",
                             "NuSVM",
                             "MARS",
                             "QDA",
                             "LinearRidge"]


def run_sklearn(X_train, X_test, y_train, y_test, method_name, seed, hyper_parameters = {}, regression = True):
    if method_name not in deterministic_methods: hyper_parameters.update({"random_state" : seed})

    #rename sklearn classes such that classifier and regressor have the same name
    if regression:
        from sklearn.dummy import DummyRegressor as Intercept
        from sklearn.linear_model import RidgeCV as Ridge
        from sklearn.linear_model import LassoCV as Lasso
        from sklearn.linear_model import ElasticNetCV as Enet
        from sklearn.tree import DecisionTreeRegressor as CART
        from sklearn.ensemble import RandomForestRegressor as RF
        from sklearn.ensemble import ExtraTreesRegressor as XRF
        from sklearn.ensemble import GradientBoostingRegressor as xgb_sklearn
        from sklearn.kernel_ridge import KernelRidge as Kernel
        from sklearn.svm import NuSVR as NuSVM
        from sklearn.neural_network import MLPRegressor as MLP_sklearn
        if method_name == "MARS": from pyearth import Earth as MARS

    else:                   
        from functools import partial
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.dummy import DummyClassifier as Intercept
        from sklearn.linear_model import LogisticRegressionCV as LogitCV
        Ridge = partial(LogitCV,penalty = "l2", solver = 'liblinear')
        Lasso = partial(LogitCV,penalty = "l1", solver = 'liblinear')
        Enet = partial(LogitCV,penalty = "elasticnet", l1_ratios = [0.5,0.9,.95], solver = 'saga')
        from sklearn.linear_model import RidgeClassifierCV
        from sklearn.utils.extmath import softmax
        class LinearRidge(RidgeClassifierCV):
            def predict_proba(self, X):
                d = self.decision_function(X)
                d_2d = np.c_[-d, d]
                return softmax(d_2d)
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
        from sklearn.tree import DecisionTreeClassifier as CART
        from sklearn.tree import ExtraTreeClassifier as XCART
        from sklearn.ensemble import RandomForestClassifier as RF
        from sklearn.ensemble import ExtraTreesClassifier as XRF
        from sklearn.ensemble import BaggingClassifier as Bagging
        from sklearn.ensemble import AdaBoostClassifier as ADABoost
        from sklearn.ensemble import GradientBoostingClassifier as xgb_sklearn
        from sklearn.neural_network import MLPClassifier as MLP_sklearn 
  
    try:
        model = eval(method_name)(**hyper_parameters)
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