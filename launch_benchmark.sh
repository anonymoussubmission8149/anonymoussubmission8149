conda activate adacap_benchmark

NDATAREG=26
SEEDS=10

OUTPATH='_benchmark.csv'
METHODS='all'
INTER='interrupt_benchmark.txt'
ONGO='ongoing_benchmark_experiments.csv'

DATAPATH=./preprocessed_datasets/
PYPATH=$(which python)
RETRY=FALSE

for (( i=0; i<$NDATAREG; i++ )); do
    $PYPATH ./launch_experiment.py --method $METHODS --dataset_id $i --task_name regression --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER --retry_failed_exp $RETRY --ongoing_file_path $ONGO
done
