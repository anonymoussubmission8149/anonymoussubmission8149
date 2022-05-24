import pandas as pd
import numpy as np
def write_results(results, output_file, output_repository, header = ["id","success"]):
    import os
    if not check_file_exists(output_file, output_repository):
        with open(output_repository + output_file, "w") as file:
            file.write(",".join(header))
            file.close()
    with open(output_repository + output_file, "a") as file:
        file.write("\n"+",".join(map(str,results)))
        file.close()
def check_file_exists(file, repository):
    import os
    return file in os.listdir(repository)
def check_exp_exists(exp_id, input_file, input_repository, check_success = False):
    if not check_file_exists(input_file, input_repository):
        return False
    results_df = pd.read_csv(input_repository+input_file, index_col = "id", usecols = ["id", "success"])
    if exp_id not in results_df.index.values:
        return False
    else:
        results_df["success"] = [success in [True, "True"] for success in results_df["success"].values]
        return results_df.loc[exp_id]["success"].any() or not check_success