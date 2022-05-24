conda activate adacap_benchmark

SEEDS=1
ABLSEEDS=10
REPSEEDS=1000

OUTPATH='_ablation.csv'
METHODS='ablation_all'
REPMETHODS='replication'
INTER='interrupt_ablation.txt'
ONGO='ongoing_ablation_experiments.csv'

DATAPATH=./preprocessed_datasets/
PYPATH=$(which python)
RETRY=FALSE

$PYPATH ./launch_experiment.py --method $METHODS --dataset_id 15 --task_name regression --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER --retry_failed_exp $RETRY --ongoing_file_path $ONGO --method_seeds $ABLSEEDS

$PYPATH ./launch_experiment.py --method $METHODS --dataset_id 12 --task_name regression --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER --retry_failed_exp $RETRY --ongoing_file_path $ONGO --method_seeds $ABLSEEDS

$PYPATH ./launch_experiment.py --method $REPMETHODS --dataset_id 15 --task_name regression --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER --retry_failed_exp $RETRY --ongoing_file_path $ONGO --method_seeds $REPSEEDS