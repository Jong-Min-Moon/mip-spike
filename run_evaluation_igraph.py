import os
import json
import numpy as np
import traceback
from cascade2p import utils
from util_mip import tune_lambda_via_noise_constraint, spp_igraph, evaluate_spike_inference_with_slack

def run_evaluation():
    # 1. Create results directory if it doesn't exist
    results_dir = 'evaluation_results_igraph'
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # 2. Load datasets
    print("Loading datasets...")
    # Assuming Ground_truth is in the current directory
    datasets = utils.load_all_ground_truth(ground_truth_folder='Ground_truth')
    dataset_names = sorted(list(datasets.keys()))

    print(f"Found {len(dataset_names)} datasets.")

    # 3. Iterate through datasets
    for name in dataset_names:
        result_path = os.path.join(results_dir, f"{name}_igraph.json")
        
        # Skip if already processed
        if os.path.exists(result_path):
            print(f"Skipping {name}, result already exists.")
            continue

        print(f"\nProcessing dataset: {name}")
        
        try:
            # Get the data (index 1 as requested)
            y = datasets[name][1]
            
            # Tune lambda using spp_igraph
            # Note: spp_igraph is imported from util_mip
            print(f"Tuning lambda for {name} using igraph...")
            optimal_lambda, best_s, best_z = tune_lambda_via_noise_constraint(y, spp_igraph)
            
            # Evaluate
            print(f"Evaluating {name}...")
            eval_result = evaluate_spike_inference_with_slack(best_z, y, 0.2)
            
            # Add metadata to result
            eval_result['dataset_name'] = name
            eval_result['optimal_lambda'] = float(optimal_lambda)
            
            # Convert any numpy types to python types for JSON serialization
            for key, value in eval_result.items():
                if isinstance(value, (np.int64, np.int32)):
                    eval_result[key] = int(value)
                elif isinstance(value, (np.float64, np.float32)):
                    eval_result[key] = float(value)

            # Save result
            with open(result_path, 'w') as f:
                json.dump(eval_result, f, indent=4)
            
            print(f"Successfully saved igraph results for {name}")

        except Exception as e:
            print(f"Error processing {name}: {str(e)}")
            traceback.print_exc()
            continue

if __name__ == "__main__":
    run_evaluation()
