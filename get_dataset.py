import numpy as np
from sklearn.preprocessing import StandardScaler as normalize
from sklearn.model_selection import train_test_split as tts

MIN_SAMPLES = 10
MAX_SAMPLES = 0.8
def dataset_loader(dataset_id, name, repository):
    return np.load(repository + name + str(dataset_id) + ".npy")
def prepare_dataset(dataset, train_size = 0.8, n_features = None, should_stratify = False, seed= False):
    kwargs = {}
    if seed or seed == 0:
        kwargs["random_state"] = seed
    X, y = dataset[:, :-1], dataset[:, -1]
    X = normalize().fit_transform(X)
    n, p = X.shape
    
    if train_size in [None, False]:
        train_size = 0.8
    if train_size >= 1:#convert absolute value to percentage
        train_size = float(train_size/n)
    if train_size * n < MIN_SAMPLES:
        train_size = float(MIN_SAMPLES/n)
    if train_size > MAX_SAMPLES:
        train_size = MAX_SAMPLES
    if type(n_features) == type(None):#all features ([:None] = [:])
        n_features = p 
    elif type(n_features) == type(1.):#percentage of all features 
        n_features = int(p * n_features)
    if n_features <= 0:#exclude last features
        n_features = p + n_features
    n_features = max(1, min(n_features, p)) #at least 1, at most p
    X = X[:,:n_features]
    if should_stratify:
        if should_stratify == "regression":
            stratify = stratify_continuous_target(y, seed = seed)
        else:
            stratify = y
    else:
        stratify = None
    X_train, X_test, y_train, y_test = tts(X, y, train_size = train_size, stratify = stratify, **kwargs)
    return X_train, X_test, y_train, y_test

def get_dataset(dataset_id, name, repository, train_size = 0.8, n_features = None, should_stratify = False, seed = False):
    return prepare_dataset(dataset_loader(dataset_id, name, repository), train_size = train_size, n_features = n_features, should_stratify = name * should_stratify, seed = seed)

def get_weak_learners_predictions(input_prediction_repository, task_tag, weak_learners, weak_learner_seeds, dataset_id, input_name, input_repository, train_size, n_features, was_stratified, dataset_seed):
    _X_train, __, ___, y_true = get_dataset(dataset_id, input_name, input_repository, train_size = train_size, n_features = n_features,should_stratify = was_stratified,seed = dataset_seed)
    n_test = len(y_true)
    n, p = _X_train.shape
    dataset = np.zeros((n_test,0))
    coef_learner_name = []
    for method_category, method_name, function, hp_name, hps in weak_learners:
        for method_seed in weak_learner_seeds:
            pred_id = "_".join(map(str,[task_tag, dataset_id, n, p, dataset_seed, was_stratified, method_category, method_name,method_seed, hp_name]))
            dataset = np.concatenate([dataset, np.load(input_prediction_repository + pred_id + ".npy").reshape((n_test,-1))], axis=-1)
            coef_learner_name.append(method_name)
    dataset = np.concatenate([dataset, y_true.reshape((n_test,-1))], axis = -1)
    return dataset, n, p, coef_learner_name

def average_weak_learners_predictions(input_prediction_repository, task_tag, weak_learners, weak_learner_seeds, dataset_id, input_name, input_repository, train_size, n_features, was_stratified, dataset_seed):
    _X_train, __, ___, y_test = get_dataset(dataset_id, input_name, input_repository, train_size = train_size, n_features = n_features,should_stratify = was_stratified,seed = dataset_seed)
    n_test = len(y_test)
    n, p = _X_train.shape
    prediction = 0.
    for method_category, method_name, function, hp_name, hps in weak_learners:
        for method_seed in weak_learner_seeds:
            pred_id = "_".join(map(str,[task_tag, dataset_id, n, p, dataset_seed, was_stratified, method_category, method_name,method_seed, hp_name]))
            prediction += np.load(input_prediction_repository + pred_id + ".npy").reshape((n_test,-1))
    prediction /= len(weak_learners) * len(weak_learner_seeds)
    return prediction, n, p, y_test

#You could find this usefull in other contexts
def stratify_continuous_target(y, seed = False):
    from sklearn.tree import DecisionTreeRegressor as tree_binarizer
    MAX_TRAINING_SAMPLES, MAX_TREE_SIZE = int(1e4), 50 #CPU memory and runtime safeguards
    tree_binarizer_params = {"criterion":'friedman_mse', 
               "splitter":'best', 
               "max_depth":None, 
               "min_samples_split":2, 
               "min_weight_fraction_leaf":0.0, 
               "max_features":None,  
               "min_impurity_decrease":0.2,
               "min_impurity_split":None, 
               "ccp_alpha":0.0}

    tree_size = min(int(np.sqrt(len(y))), MAX_TREE_SIZE)
    fit_sample_size = min(len(y), MAX_TRAINING_SAMPLES)
    tree_binarizer_params["max_leaf_nodes"] = tree_size
    tree_binarizer_params["min_samples_leaf"] = tree_size
    if seed or seed == 0:
        tree_binarizer_params["random_state"] = seed
    return tree_binarizer(**tree_binarizer_params).fit(y[:fit_sample_size].reshape((-1,1)), y[:fit_sample_size]).apply(y.reshape((-1,1))) 