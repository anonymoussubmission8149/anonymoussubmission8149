import AdaCapNet
from benchmarked_architectures import *
import torch
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score as acc
def auc(y, probas):
    from sklearn.metrics import roc_auc_score
    if probas.shape[-1] == 2:
        return roc_auc_score(y, probas[:,-1])
    else:
        return roc_auc_score(y, probas, multi_class ="ovr")
    
def run_adacapnet(X_train, X_test, y_train, y_test, method_name, seed, hyper_parameters = {}, regression = True):
    kwargs = eval(method_name)#from benchmarked_architectures.py
    kwargs.update(hyper_parameters)
    kwargs["hidden_params"]["n_features"] = X_train.shape[-1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs["hidden_params"]["device"] = device 
    try:
        method_class = AdaCapNet.AdaCapNetRegressor if regression else AdaCapNet.AdaCapNetClassifier
        model = method_class(random_state = seed, **kwargs)
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
    try: model.delete_model_weights()
    except: success = False
    try: del model
    except: success = False
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return success, results, prediction