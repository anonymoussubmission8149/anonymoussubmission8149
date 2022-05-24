import numpy as np
import pandas as pd
from copy import deepcopy
import os
import urllib.request as req
from dataset_preprocessing_utilities import *
import shutil

infos_rep = "./"
uciopenml_raw_files_infos = "download_format_uciopenml.csv"
kaggle_raw_files_infos = "download_format_kaggle.csv"
description_task_target_infos = "description_task_target.csv"
benchmark_infos = "benchmark_infos.csv"

raw_files_rep = "./raw_files/"
ds_repository = "./unrefined_datasets/"
np_repository = "./raw_matrices/"
preprocessed_datasets_repository = "../preprocessed_datasets/"

download_format_uciopenml = True
download_format_kaggle = True
rerun_download_format = True
rerun_preprocess = True
verbose = False

####################
os.makedirs(raw_files_rep, exist_ok = True)
os.makedirs(ds_repository, exist_ok = True)
os.makedirs(np_repository, exist_ok = True)
os.makedirs(preprocessed_datasets_repository, exist_ok = True)

df_raw_infos_uciopenml = pd.read_csv(infos_rep+uciopenml_raw_files_infos, index_col = 0)
df_raw_infos_uciopenml["new_name"] = df_raw_infos_uciopenml["pretty_name"] + df_raw_infos_uciopenml["original_file_name"]
df_raw_infos_kaggle = pd.read_csv(infos_rep+kaggle_raw_files_infos, index_col = 0)
df_raw_infos_kaggle["sep"]= list(map(convert_float_nan,df_raw_infos_kaggle["sep"].values))
df_raw_infos_kaggle["encoding"]= list(map(convert_float_nan,df_raw_infos_kaggle["encoding"].values))

####################
if download_format_uciopenml:
    df_raw_infos_uciopenml = pd.read_csv(infos_rep+uciopenml_raw_files_infos, index_col = 0)
    df_raw_infos_uciopenml["new_name"] = df_raw_infos_uciopenml["pretty_name"] + df_raw_infos_uciopenml["original_file_name"]
    for row_name, row in df_raw_infos_uciopenml.iterrows():
        full_url = row["url_repository"] + row["original_file_name"]
        new_name = row["new_name"]
        if not check_file_exists(new_name, raw_files_rep) or rerun_download_format:
            if verbose: print(full_url)
            req.urlretrieve(full_url, raw_files_rep + new_name)
            if verbose: print(row_name, "/", len(df_raw_infos_uciopenml), row["pretty_name"], ": Done!")
        elif verbose: print(row["pretty_name"], ": Already there")

####################
if download_format_kaggle:
    from kaggle.api.kaggle_api_extended import KaggleApi
    import zipfile
    api = KaggleApi()
    api.authenticate()

    for row_name, row in df_raw_infos_kaggle.iterrows():
        kaggle_path = row["kaggle_api_path"]
        kaggle_name = row["kaggle_api_name"]
        new_name = row["new_name"]
        download_name = row["original_file_name"]
        unzipped_name = row["unzipped_name"]
        should_unzip = row["zipped"] 
        if not check_file_exists(new_name, raw_files_rep) or rerun_download_format:
            if not check_file_exists(download_name, raw_files_rep):
                print('downloading',kaggle_path,kaggle_name, new_name)
                if row["origine"] == 'dataset':
                    new = api.dataset_download_file(kaggle_path,kaggle_name, path = raw_files_rep)
                else:
                    new =  api.competition_download_file(kaggle_path,kaggle_name, path = raw_files_rep)
            if should_unzip:
                print("unzipping")
                if check_file_exists(download_name, raw_files_rep):
                    with zipfile.ZipFile(raw_files_rep+download_name, 'r') as zip_ref:
                        zip_ref.extractall(raw_files_rep)
                elif check_file_exists(download_name+".zip", raw_files_rep):
                    with zipfile.ZipFile(raw_files_rep+download_name+".zip", 'r') as zip_ref:
                        zip_ref.extractall(raw_files_rep)
            if check_file_exists(download_name, raw_files_rep) and download_name!=unzipped_name:
                os.remove(raw_files_rep+download_name)
            elif check_file_exists(download_name+".zip", raw_files_rep) and download_name+".zip" !=unzipped_name:
                os.remove(raw_files_rep+download_name+".zip")
            os.rename(raw_files_rep+unzipped_name, raw_files_rep+new_name)

####################
df_raw_infos = df_raw_infos_uciopenml.append(df_raw_infos_kaggle[df_raw_infos_uciopenml.columns])

#"raw_file_size"
df_raw_infos["raw_file_size"] = np.zeros(len(df_raw_infos))
for row_name, row in df_raw_infos.iterrows():
    df_raw_infos["raw_file_size"][row_name] = os.path.getsize(raw_files_rep + row["new_name"])
df_raw_infos.sort_values("raw_file_size", inplace = True)

#populate infos for each task and target combination of every datasets
row_names, row_list = [], []
for row_name, row in df_raw_infos.iterrows():
    tasks = list(row["task"])
    targets = str(row["y"]).split(";")
    if len(tasks) == len(targets):
        targetntasks = list(zip(tasks,targets))
    else:
        targetntasks = [(task, target) for target in targets for task in tasks]
    for task, target in targetntasks:
        new_row = row.copy()
        new_row["y"] = target
        new_row["task"] = task
        new_row["excluded_columns"] = ";".join(targets)
        new_row["name_task_target"] = str(new_row["new_name"]) + "_task_"+ str(task)+ "_target_" + str(target)
        row_list.append(new_row)
        row_names.append(row_name)
dataset_by_task_target = pd.DataFrame(row_list)

#manually reformat for unusual separators
file_name = 'Yacht Hydrodynamicsyacht_hydrodynamics.data'
if check_file_exists(file_name, raw_files_rep):
    with open(raw_files_rep+file_name, "r") as file:
        data = file.readlines()
    for index,line in enumerate(data):
        data[index] = line.replace(" \n", "\n").replace("  ", " ")
    with open(raw_files_rep+file_name, 'w') as file:
        file.writelines( data )
    del data
for row_name, ds_infos in dataset_by_task_target.iterrows():
    if ds_infos["sep"] == ";":
        with open(raw_files_rep+ds_infos["new_name"], "r") as file:
            data = file.readlines()
        for index,line in enumerate(data):
            data[index] = line.replace(",", ".")
        with open(raw_files_rep+ds_infos["new_name"], 'w') as file:
            file.writelines( data )
        del data
        
#put in standard csv format while keeping only relevant rows
dataset_by_task_target = dataset_by_task_target.where(pd.notnull(dataset_by_task_target), None)
for row_name, ds_infos in dataset_by_task_target.iterrows():
    if verbose: print(row_name,ds_infos["pretty_name"], ds_infos["raw_file_size"])
    if not check_file_exists(ds_infos["name_task_target"],np_repository) or rerun_download_format:
        raw_np_mat = panda_read_raw_file(raw_files_rep, ds_infos)
        raw_np_mat.to_csv(np_repository+ds_infos["name_task_target"])
        if verbose: print(row_name, raw_np_mat.shape)
    else:
        if verbose: 
            print("Already There")
            print()
dataset_by_task_target["target_reformated"] = np.zeros(len(dataset_by_task_target)).astype(bool)

####################
#generate .npy file and preprocess features
regressions, classifications, multiclasses = 0, 0, 0
ds_full_name = []
ds_new_name = []
has_failed = []
y_means = []
y_stds = []

conti_count = []
categ_count = []
ds_index_count = []

for row_name, ds_infos in dataset_by_task_target.iterrows():
    ds_full_name.append(ds_infos["name_task_target"])
    if ds_infos["task"] == "R":
        task_name = "regression"
        index = regressions
        regressions += 1
    if ds_infos["task"] == "C":
        task_name = "classification"
        index = classifications
        classifications += 1

    if ds_infos["task"] == "M":#Not Needed
        task_name = "multiclass"
        index = multiclasses
        multiclasses += 1
    dataset_name = task_name+str(index)
    if not check_file_exists(dataset_name+".npy",preprocessed_datasets_repository) or rerun_preprocess:
        matrix = pd.read_csv(np_repository+ds_infos["name_task_target"], index_col = 0).values

        data = process_matrix(matrix, ds_infos)
        np.save(preprocessed_datasets_repository+dataset_name, data["data"])
        matrix = data.pop("data")

        n, p = matrix.shape
        if data["info"]["y_info"]["type"] == "numeric" and ds_infos["task"] == "R":
            y_means.append(data["info"]["y_info"]["mean"])
            y_stds.append(data["info"]["y_info"]["std"])
        else:
            y_means.append(None)
            y_stds.append(None)
        ds_infos["info"] = data["info"]
        
        feature_info = data["info"]["X_info"]
        n_continuous = len([key for key, value in feature_info.items() if value["type"]=="quantitative"])
        n_categorical = len(set([value["orig"] for key, value in feature_info.items() if value["type"]!="quantitative"]))
        ds_last_index = regressions if task_name == "regression" else classifications if task_name == "classification" else multiclasses
        ds_last_index = ds_last_index - 1
        
        conti_count.append(n_continuous)
        categ_count.append(n_categorical)
        ds_index_count.append(ds_last_index)
        
        ds_new_name.append(preprocessed_datasets_repository+dataset_name+".npy")
        has_failed.append(False)

    if verbose: 
        print(ds_infos["name_task_target"],dataset_name)
del data
del matrix

dataset_by_task_target["np_matrix_name"] = ds_new_name
dataset_by_task_target["processing_failed"] = has_failed
dataset_by_task_target["y_mean"] = y_means
dataset_by_task_target["y_std"] = y_stds
dataset_by_task_target["n_continuous"] = conti_count
dataset_by_task_target["n_categorical"] = categ_count
dataset_by_task_target["ds_last_index"] = ds_index_count


####################
# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'Seoul Bike Sharing DemandSeoulBikeData.csv_task_R_target_1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = -np.log(y + 1)
    y_ = np.log(y_ - y_.min() + 2)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True
    
orig_name = 'Servoservo.data_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    mat[:,-1] = renorma(y_)
    # <<< specific
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

orig_name = 'Computer Hardwaremachine.data_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    mat[:,-1] = renorma(y_)
    # <<< specific
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True
    
orig_name = 'Yacht Hydrodynamicsyacht_hydrodynamics.data_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    mat[:,-1] = renorma(y_)
    # <<< specific
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True
    
orig_name = 'Communities and Crimecommunities.data_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y + 1)
    mat[:,-1] = renorma(y_)
    # <<< specific
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'restaurant-revenue-predictiontrain.csv_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'mirichoi0218_insuranceinsurance.csv_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

    
# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'snooptosh_bangalore-real-estate-priceblr_real_estate_prices.csv_task_R_target_-1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'arashnic_marketing-seris-customer-lifetime-valuesquark_automotive_CLV_training_data.csv_task_R_target_2'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True

# specific target rescaling for some regression datasets (mostly log rescale)
orig_name = 'mercedes-benz-greener-manufacturingtrain.csv_task_R_target_1'
if not dataset_by_task_target.set_index("name_task_target").loc[orig_name]["target_reformated"]:
    mat_name = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["np_matrix_name"]
    mean = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_mean"]
    std = dataset_by_task_target.reset_index().set_index("name_task_target").loc[orig_name]["y_std"]
    mat = np.load(mat_name)
    
    #specific >>>
    y = mat[:,-1] * std + mean
    y_ = np.log(y)
    # <<< specific
    mat[:,-1] = renorma(y_)
    np.save(mat_name[:-4], mat)
    dataset_by_task_target.iloc[np.where(dataset_by_task_target["name_task_target"] == orig_name)[0][0]]["target_reformated"] = True
    
####################
#adding extra infos and removing intermediary files
benchmark_infos_df = pd.read_csv(infos_rep + benchmark_infos,index_col = 0).set_index("name_task_target")
dataset_by_task_target.set_index("name_task_target", inplace = True)
for col in ["bagging_benchmark", "minRMSE<0.25_benchmark", "Concrete_ablation", "Abalone_ablation"]:
    dataset_by_task_target[col] = benchmark_infos_df[col]
dataset_by_task_target.reset_index(inplace = True)
dataset_by_task_target.to_csv(infos_rep+description_task_target_infos)

shutil.rmtree(raw_files_rep, ignore_errors = True)
shutil.rmtree(ds_repository, ignore_errors = True)
shutil.rmtree(np_repository, ignore_errors = True)