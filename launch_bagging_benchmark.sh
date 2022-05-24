conda activate adacap_benchmark

SEEDS=10
BAGSEEDS=10

OUTPATH='_bagging_benchmark.csv'
PREDPATH=./predictions/
METHODS='bagging'
INTER='interrupt_bagging_benchmark.txt'
ONGO='ongoing_bagging_benchmark_experiments.csv'

DATAPATH=./preprocessed_datasets/
PYPATH=$(which python)
RETRY=FALSE

for i in 0 1 3 5 6 7 9 10 11 12 13 15 16 17 22; do
    $PYPATH ./launch_experiment.py --method $METHODS --dataset_id $i --task_name regression --benchmark_output_file $OUTPATH --dataset_seeds $SEEDS --input_repository $DATAPATH --interrupt_file_path $INTER --retry_failed_exp $RETRY --ongoing_file_path $ONGO --method_seeds $BAGSEEDS --prediction_output_file $PREDPATH
done

for i in 0 1 3 5 6 7 9 10 11 12 13 15 16 17 22; do
    srun $PYPATH ./launch_bagging_experiment.py.py --method $METHODS --dataset_id $i --task_name regression --benchmark_output_file $OUTPATH --dataset_seeds $SEEDONE --input_repository $DATAPATH --interrupt_file_path $INTER --retry_failed_exp $RETRY --ongoing_file_path $ONGO --prediction_input_file $PREDPATH --method_seeds $BAGSEEDS
done