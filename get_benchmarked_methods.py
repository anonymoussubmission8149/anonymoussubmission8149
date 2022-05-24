import importlib
def get_benchmarked_methods(methods, regression = True):
    #methods format: Category Name (str), Method Name (str), Method Function (str), Hyparameters name (str), Hyperparameters (dict)
    
    validate_only = "validate_only" in methods
    bagging_only = "bagging_only" in methods
    implementation_not_needed = validate_only or bagging_only
    
    #else excluded from the benchmark
    torch_available = implementation_not_needed or importlib.util.find_spec('torch') is not None #conda install -c pytorch pytorch
    xgboost_available = implementation_not_needed or importlib.util.find_spec('xgboost') is not None #conda install -c conda-forge xgboost
    catboost_available = implementation_not_needed or importlib.util.find_spec('catboost') is not None #conda install -c conda-forge catboost
    lgbm_available = implementation_not_needed or importlib.util.find_spec('lightgbm') is not None #conda install -c conda-forge lightgbm
    mars_available = implementation_not_needed or importlib.util.find_spec('pyearth') is not None #conda install -c conda-forge sklearn-contrib-py-earth
    test_cpu = "GPU_only" not in methods or implementation_not_needed
    test_gpu = "CPU_only" not in methods or implementation_not_needed
    test_slow = "slow" in methods
    test_additional_architectures = False #architectures not included in benchmark
    
    test_sklearn = "sklearn" in methods
    test_rf = "RF" in methods or test_sklearn
    test_adacapnet = "adacapnet" in methods and torch_available
    test_regularnet = "regularnet" in methods and torch_available
    test_xgboost = "xgboost" in methods and xgboost_available
    test_catboost = "catboost" in methods and catboost_available
    test_lgbm = "lgbm" in methods and lgbm_available
    test_mars = "mars" in methods and mars_available
    
    test_bagging = "bagging" in methods
    test_ablation_width = "ablation_width" in methods and torch_available and test_gpu
    test_ablation_depth = "ablation_depth" in methods and torch_available and test_gpu
    test_ablation_batch = "ablation_batch" in methods and torch_available and test_gpu
    test_ablation_dropout = "ablation_dropout" in methods and torch_available and test_gpu
    test_replication = "replication" in methods and torch_available and test_gpu

    #HPO not evaluated for time reasons
    test_nohpo = "HPO_only" not in methods
    
    #fast benchmark options
    test_fast_adacapnet = ("fast_adacapnet" in methods and torch_available) or test_adacapnet
    test_fast_regularnet = ("fast_regularnet" in methods and torch_available) or test_regularnet
    test_batch_adacapnet = ("batch" in methods and torch_available and test_slow) or test_adacapnet and test_slow
    test_batch_regularnet = ("batch" in methods and torch_available and test_slow) or test_regularnet and test_slow
    test_fast_catboost = ("fast_catboost" in methods and catboost_available) or test_catboost
    
    #Method category names
    baseline_name = "Baseline"
    lm_name = "GLM"
    QDA_name = "QDA"
    tree_name = "TREE"
    ensemble_name = "RF"
    spline_name = "MARS"
    svm_name = "SVM"
    nn_name = "NN"
    xgb_name = "GBDT"
    adacapnet_name = "ADACAP"
    
    #run_method name
    xgb_experiment = "run_xgb"
    sklearn_experiment = "run_sklearn"
    adacapnet_experiment = "run_adacapnet"
    regularnet_experiment = "run_regularnet"

    methods = []
    methods += [(baseline_name, "Intercept", sklearn_experiment, "nohp", {}),
                         (lm_name, "Ridge", sklearn_experiment, "nohp", {}),
                         (lm_name, "Lasso", sklearn_experiment, "nohp", {}),
                         (lm_name, "Enet", sklearn_experiment, "nohp", {}),
                         (tree_name, "CART", sklearn_experiment, "nohp", {}),
                         (ensemble_name, "XRF", sklearn_experiment, "nohp", {}),
                         (xgb_name, "xgb_sklearn", sklearn_experiment, "nohp", {})] * test_sklearn * test_nohpo * test_cpu
                             
    methods += [(svm_name, "Kernel", sklearn_experiment, "nohp", {}),
                         (svm_name, "NuSVM", sklearn_experiment, "nohp", {})] * test_sklearn * test_nohpo * test_cpu
    
    methods += [(ensemble_name, "RF", sklearn_experiment, "nohp", {})] * test_rf * test_nohpo * test_cpu
    methods += [(xgb_name, "XGBoost", xgb_experiment, "nohp", {})] * test_xgboost * test_nohpo * test_gpu
    methods += [(xgb_name, "CAT", xgb_experiment, "nohp", {})] * test_catboost * test_nohpo * test_gpu * test_slow
    methods += [(xgb_name, "CATfast", xgb_experiment, "nohp", {})] * test_fast_catboost * test_nohpo * test_gpu
    methods += [(xgb_name, "LGBM", xgb_experiment, "nohp", {})] * test_lgbm * test_nohpo * test_gpu
    methods += [(spline_name, "MARS", sklearn_experiment, "nohp", {})] * test_mars * test_nohpo * test_cpu
    
    methods += [(adacapnet_name, "adacapnetfast", adacapnet_experiment, "nohp", {})] * test_fast_adacapnet * test_nohpo * test_gpu
    methods += [(nn_name, "regularnetfast", regularnet_experiment, "nohp", {})] * test_fast_regularnet * test_nohpo * test_gpu
    for architecture_name in ["standard","resblock", "glu", "selu", "fastselu"]:
        methods += [(adacapnet_name, "adacapnet"+architecture_name, adacapnet_experiment, "nohp", {})] * test_adacapnet * test_nohpo * test_gpu
        methods += [(nn_name, "regularnet"+architecture_name, regularnet_experiment, "nohp", {})] * test_regularnet * test_nohpo * test_gpu
        
    methods += [(adacapnet_name, "adacapnetbatchstandard", adacapnet_experiment, "nohp", {})] * test_batch_adacapnet * test_nohpo * test_gpu * test_slow
    methods += [(adacapnet_name, "adacapnetbatchresblock", adacapnet_experiment, "nohp", {})] * test_batch_adacapnet * test_nohpo * test_gpu * test_slow
    methods += [(nn_name, "regularnetbatchstandard", regularnet_experiment, "nohp", {})] * test_batch_regularnet * test_nohpo * test_gpu
    methods += [(nn_name, "regularnetbatchresblock", regularnet_experiment, "nohp", {})] * test_batch_regularnet * test_nohpo * test_gpu
    
    if test_bagging:
        methods += [(lm_name, "Ridge", sklearn_experiment, "nohp", {}),
                    (lm_name, "Lasso", sklearn_experiment, "nohp", {}),
                    (lm_name, "Enet", sklearn_experiment, "nohp", {}),
                    (ensemble_name, "XRF", sklearn_experiment, "nohp", {}),
                    (xgb_name, "xgb_sklearn", sklearn_experiment, "nohp", {})] * test_cpu
        methods += [(xgb_name, "LGBM", xgb_experiment, "nohp", {})]* lgbm_available * test_gpu
        methods += [(xgb_name, "XGBoost", xgb_experiment, "nohp", {})]* xgboost_available * test_gpu
        methods += [(xgb_name, "CATfast", xgb_experiment, "nohp", {})]* catboost_available * test_gpu
        methods += [(xgb_name, "CAT", xgb_experiment, "nohp", {})] * catboost_available * test_gpu
        methods += [(adacapnet_name, "adacapnetfast", adacapnet_experiment, "nohp", {})]* torch_available * test_gpu
        methods += [(nn_name, "regularnetfast", regularnet_experiment, "nohp", {})]* torch_available * test_gpu
        methods += [(adacapnet_name, "adacapnetselu", adacapnet_experiment, "nohp", {})]* torch_available * test_gpu
        methods += [(nn_name, "regularnetselu", regularnet_experiment, "nohp", {})]* torch_available * test_gpu
        methods += [(adacapnet_name, "adacapnetglu", adacapnet_experiment, "nohp", {})]* torch_available * test_gpu
        methods += [(nn_name, "regularnetglu", regularnet_experiment, "nohp", {})]* torch_available * test_gpu
        if regression:
            methods += [(svm_name, "Kernel", sklearn_experiment, "nohp", {})] * test_cpu
            methods += [(spline_name, "MARS", sklearn_experiment, "nohp", {})] * mars_available * test_cpu
            
    if test_replication:
        architecture = "fast"
        methods += [(nn_name, "regularnet"+architecture+"weightseed0", regularnet_experiment, "nohp", {})]
        methods += [(adacapnet_name, "adacapnet"+architecture+"weightseed0npermut0", adacapnet_experiment, "nohp", {})]
        methods += [(adacapnet_name, "adacapnet"+architecture+"weightseed0npermut1", adacapnet_experiment, "nohp", {})]
        methods += [(adacapnet_name, "adacapnet"+architecture+"weightseed0npermut16", adacapnet_experiment, "nohp", {})]
        methods += [(nn_name, "regularnet"+architecture+"weightseed0DO", regularnet_experiment, "nohp", {})]
            
    if test_ablation_width:
        width_range = [16,32,64,258,256,512,1024,2048]
        for param_value in width_range:
            for architecture in ["fast"] + ["standard","resblock","batchstandard","batchresblock", "selu", "fastselu"] * test_ablation_all:
                methods += [(adacapnet_name, "adacapnet"+architecture+"width"+str(param_value), adacapnet_experiment, "nohp", {})]
                methods += [(nn_name, "regularnet"+architecture+"width"+str(param_value), regularnet_experiment, "nohp", {})]
                
                
    if test_ablation_dropout:
        dropout_range = [0., 0.1, 0.2, 0.3, 0.4, 0.5]
        for param_value in dropout_range:
            for architecture in ["fast"] + ["standard","resblock","batchstandard","batchresblock", "selu", "fastselu"] * test_ablation_all:
                suffix = architecture+"dropout"+str(param_value).replace(".","_").replace("-","m")
                suffixbn = architecture+"dropoutbatchnorm"+str(param_value).replace(".","_").replace("-","m")
                methods += [(adacapnet_name,"adacapnet"+suffix, adacapnet_experiment, "nohp", {})]
                methods += [(adacapnet_name,"adacapnet"+suffixbn, adacapnet_experiment, "nohp", {})]
                
    if test_ablation_depth:
        depth_range = [1,2,3,4,5,6]
        for param_value in depth_range:
            for architecture in ["fast"] + ["standard","resblock","batchstandard","batchresblock", "selu", "fastselu"] * test_ablation_all:
                methods += [(adacapnet_name, "adacapnet"+architecture+"depth"+str(param_value), adacapnet_experiment, "nohp", {})] 
                methods += [(nn_name, "regularnet"+architecture+"depth"+str(param_value), regularnet_experiment, "nohp", {})]   
                
    if test_ablation_batch:
        batch_range = [16,32,64,128,258,256,512,1024,2048]
        for param_value in batch_range:
            for architecture in ["fast"] + ["standard","resblock","batchstandard","batchresblock"] * test_ablation_all:
                methods += [(adacapnet_name, "adacapnet"+architecture+"bs"+str(param_value), adacapnet_experiment, "nohp", {})] 
                methods += [(nn_name, "regularnet"+architecture+"bs"+str(param_value), regularnet_experiment, "nohp", {})]
                
    if test_additional_architectures:
        for architecture_name in [ "resglu", "wide", "resblockwide", "seludeep", "deep", "resblockdeep", "resblockdeepselu", "reglu", "resreglu"]:
            methods += [(adacapnet_name, "adacapnet"+architecture_name, adacapnet_experiment, "nohp", {})] * test_adacapnet * test_nohpo * test_gpu * test_slow 
            methods += [(nn_name, "regularnet"+architecture_name, regularnet_experiment, "nohp", {})] * test_regularnet * test_nohpo * test_gpu * test_slow
                
    if validate_only:
        methods = [[method_category, method_name, "run_validevaluation", "validation", {}] for method_category, method_name, function, hp_name, hps in methods]
    
    return methods
def get_run_(function):
    if function == "run_sklearn": from run_sklearn import run_sklearn
    if function == "run_xgb": from run_xgb import run_xgb
    if function == "run_regularnet": from run_regularnet import run_regularnet
    if function == "run_adacapnet": from run_adacapnet import run_adacapnet
    if function == "run_catfast": from run_catfast import run_catfast
    if function == "run_validevaluation": pass
    return eval(function)

#utilities for metalearning techniques
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score as acc
def auc(y, probas):
    from sklearn.metrics import roc_auc_score
    if probas.shape[-1] == 2:
        return roc_auc_score(y, probas[:,-1])
    else:
        return roc_auc_score(y, probas, multi_class ="ovr")
def proba2class(prediction):
    if prediction.shape[-1] <=2:
        return prediction[:,-1]>0.5
    else:
        return np.argmax(prediction,axis= -1)
def evaluate_prediction(prediction, target, regression = True):
    results = [r2_score(target, prediction)] if regression else [acc(target, proba2class(prediction)), auc(target, prediction)]
    return True, results, prediction

def run_validevaluation(X_train, X_test, y_train, y_test, method_name, seed, hyper_parameters = {}, regression = True):
    try:
        success_test, results_test, prediction = evaluate_prediction(X_test, y_test, regression = regression)
        success_train, results_train, train_prediction = evaluate_prediction(X_train, y_train, regression = regression)
        success = success_test * success_train
        results = results_test + results_train
    except: 
        prediction = None
        results = [None, None] if regression else [None, None, None, None]
        success = False
    return success, results, prediction
