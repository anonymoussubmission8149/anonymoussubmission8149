# AdaCap Supplementary material
This repository contains code to replicate all experiments featured in the submission.
### Clone repository and set up
```bash
export REPO_DIR=<ABSOLUTE path to the desired repository directory>
git clone 'https://anonymoussubmission8149/anonymoussubmission8149' $REPO_DIR
cd $REPO_DIR
mkdir outputs figures weights predictions
```

# Create Envs
### generate datasets
```bash
conda create -n dataset_generation python=3.9
conda activate dataset_generation
conda install ipykernel numpy=1.21.2 pandas=1.3.4 scikit-learn=1.0.1 -y
conda install -c conda-forge kaggle=1.5.12 -y
conda install -c anaconda scipy=1.7.1 -y
conda deactivate
```

### complete benchmark
```bash
conda create -n adacap_benchmark python=3.7
conda activate mlrnet_benchmark
conda install ipykernel pandas=1.1.3 scikit-learn=0.23.2 -y
conda install -c anaconda scipy=1.6.2 -y
conda install -c conda-forge sklearn-contrib-py-earth=0.1.0 xgboost=1.3.3 catboost=0.26.1 lightgbm=3.2.1 matplotlib=3.4.3 -y
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda deactivate
```

### to run AdaCap alone you mostly need pytorch
```bash
conda create -n adacap_demo python=3.7
conda activate adacap_demo
conda install ipykernel pandas=1.1.3 scikit-learn=0.23.2 -y
conda install -c conda-forge matplotlib=3.4.3 -y
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda deactivate
```

### if no gpu available:
- replace 
```bash
conda install pytorch=1.9.1 cudatoolkit=11.1 -c pytorch -c conda-forge -y
```
- with 
```bash
conda install pytorch=1.9.1 -c pytorch -y
```
- It might be very very very slow...

# AdaCapNet as a standalone
## Quick Demo on Boston
Run AdaCapDemo.ipynb with the env adacap_demo (you only need AdaCapNet.py, architectures.py and benchmarked_architectures.py in the current repository)
## Use AdaCapNet in other projects 
- Use AdaCapNetRegressor & AdaCapNetClassifier from AdaCapNet.py (init/fit/predict/score behavior like scikit-learn)
- architectures.py provides simple pytorch FFNN implementations for ConvNet, MLP, ResBlock, GLU 
- benchmarked_architectures.py provides sensible sets of hyperparameters for AdaCapNet

# Generate Figures
You can use either adacap_benchmark or adacap_demo virtual envs and run all cells of the notebooks
## Figure 1:
run MLRvsCV_Criterion_Comparison.ipynb with adacap_benchmark env (will need about a minute to complete, does not require a GPU)
## Figure 2: 
run Ablation_Weights_Correlations.ipynb with adacap_benchmark env (will need about a minute to complete with a GPU)
## Figure 3:
run Ablation_Learning_Dynamic.ipynb with adacap_benchmark env (will need about a minute to complete with a GPU)

# Benchmark and Ablation
## Generate datasets

### To generate datasets you need to create a kaggle account, set up an api token, and agree to competition rules for those datasets:

- https://www.kaggle.com/c/restaurant-revenue-prediction
- https://www.kaggle.com/c/mercedes-benz-greener-manufacturing


### Run dataset_generation.py script
```bash
cd dataset_generation
conda activate dataset_generation
$(which python) ./download_format_preprocess_datasets.py
conda deactivate
cd ..
```
## run global benchmark
```bash
./launch_benchmark.sh
```

## run bagging benchmark
```bash
./launch_bagging_benchmark.sh
```

## run ablation
```bash
./launch_ablation.sh
```
