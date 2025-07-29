# evaluate_ensemble.py

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import inspect
from tqdm import tqdm
import logging

# --- Import your project's modules ---
from enhanced_asl_network import EnhancedASLNet
from enhanced_simulation import RealisticASLSimulator, ASLParameters
from utils import engineer_signal_features
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from comparison_framework import apply_normalization_to_input_flat, denormalize_predictions
from utils import engineer_signal_features

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_artifacts(results_dir: Path) -> tuple:
    """Loads the model ensemble, configuration, and normalization stats."""
    logging.info(f"Loading artifacts from: {results_dir}")

    # Load and update config with Optuna best params if available
    config_path = results_dir / 'research_config.json'
    if not config_path.exists(): raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f: config = json.load(f)
    logging.info("Loaded base research_config.json")

    final_results_path = results_dir / 'final_research_results.json'
    if final_results_path.exists():
        with open(final_results_path, 'r') as f: final_results = json.load(f)
        if 'optuna_best_params' in final_results and final_results['optuna_best_params']:
            logging.info("Updating config with Optuna best params for evaluation.")
            best_params = final_results['optuna_best_params']
            config['hidden_sizes'] = [
                best_params.get('hidden_size_1'), best_params.get('hidden_size_2'), best_params.get('hidden_size_3')
            ]
            config['dropout_rate'] = best_params.get('dropout_rate')
    logging.info(f"Using hidden_sizes: {config['hidden_sizes']}")

    # Load normalization stats
    norm_stats_path = results_dir / 'norm_stats.json'
    if not norm_stats_path.exists(): raise FileNotFoundError(f"Norm stats not found: {norm_stats_path}")
    with open(norm_stats_path, 'r') as f: norm_stats = json.load(f)
    logging.info("Loaded norm_stats.json")

    # Load all models in the ensemble
    models_dir = results_dir / 'trained_models'
    ensemble_models = []
    num_plds = len(config.get('pld_values', []))
    base_input_size = num_plds * 2 + 4
    
    # The original filtering of kwargs was too restrictive and removed physics parameters.
    # Pass the full config dict instead; the model's __init__ will pick what it needs.
    # model_param_keys = inspect.signature(EnhancedASLNet).parameters.keys()
    # filtered_kwargs = {k: v for k, v in config.items() if k in model_param_keys}
    
    for i in range(config.get('n_ensembles', 5)):
        model_path = models_dir / f'ensemble_model_{i}.pt'
        if not model_path.exists():
            logging.warning(f"Model file not found: {model_path}. Skipping.")
            continue
        model = EnhancedASLNet(input_size=base_input_size, **config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        ensemble_models.append(model)
    if not ensemble_models: raise FileNotFoundError("No models could be loaded.")
    logging.info(f"Successfully loaded {len(ensemble_models)} models for the ensemble.")
        
    return ensemble_models, config, norm_stats

def analyze_and_report_full_grid(results_collector: dict, true_params_grid: np.ndarray) -> pd.DataFrame:
    """
    Analyzes results for each point on the full CBF/ATT grid without aggregation.

    Returns:
        A detailed DataFrame with one row per (method, grid_point).
    """
    n_points = true_params_grid.shape[0]
    full_results_list = []

    for i in range(n_points):
        true_cbf = true_params_grid[i, 0]
        true_att = true_params_grid[i, 1]

        for method, all_preds in results_collector.items():
            pred_cbf, pred_att = all_preds[i] # Get the prediction for this specific point

            if np.isnan(pred_cbf) or np.isnan(pred_att):
                # Handle cases where the fitting failed (NaN prediction)
                cbf_error = np.nan
                att_error = np.nan
                cbf_rel_error = np.nan
                att_rel_error = np.nan
            else:
                cbf_error = pred_cbf - true_cbf
                att_error = pred_att - true_att
                # Avoid division by zero, though true values shouldn't be zero here
                cbf_rel_error = (cbf_error / true_cbf) * 100 if true_cbf != 0 else np.nan
                att_rel_error = (att_error / true_att) * 100 if true_att != 0 else np.nan

            full_results_list.append({
                'True CBF': true_cbf,
                'True ATT': true_att,
                'Method': method,
                'Predicted CBF': pred_cbf,
                'Predicted ATT': pred_att,
                'CBF Error': cbf_error,
                'ATT Error (ms)': att_error,
                'CBF Rel Error %': cbf_rel_error,
                'ATT Rel Error %': att_rel_error
            })

    return pd.DataFrame(full_results_list)

def run_evaluation(models: list, config: dict, norm_stats: dict):
    """Generates data and runs the 4 specified comparison scenarios across the landscape."""
    
    # 1. Setup
    sim_params = ASLParameters(**{k: v for k, v in config.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=sim_params)
    plds_np = np.array(config['pld_values'])
    num_plds = len(plds_np)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device)

    # Define the test landscape to cover the full physiological range
    cbf_grid = np.linspace(20, 100, 8)
    att_grid = np.linspace(500, 4000, 10)
    true_params_grid = np.array(np.meshgrid(cbf_grid, att_grid)).T.reshape(-1, 2)
    n_points = len(true_params_grid)
    logging.info(f"Testing on a systematic {len(cbf_grid)}x{len(att_grid)} CBF/ATT grid ({n_points} points).")
    
    # 2. Data Generation and Prediction Loop
    results_collector = {
        'LS (4-repeat)': [], 
        'LS (1-repeat)': [],
        'NN (4-repeat)': [], 
        'NN (1-repeat)': []
    }
    
    for i in tqdm(range(n_points), desc="Processing Grid Points"):
        true_cbf, true_att = true_params_grid[i]
        
        repeats_data = []
        for _ in range(4):
            data_dict = simulator.generate_synthetic_data(
                plds_np, np.array([true_att]), n_noise=1, tsnr=8.0, cbf_val=true_cbf
            )
            repeats_data.append(data_dict['MULTIVERSE'][0, 0, :, :])

        # --- Process the 4 Scenarios ---
        avg_signal_ls = np.mean(repeats_data, axis=0)
        cbf_ls_4, att_ls_4 = fit_conventional(avg_signal_ls, plds_np, simulator.params)
        results_collector['LS (4-repeat)'].append([cbf_ls_4, att_ls_4])

        single_signal_ls = repeats_data[0]
        cbf_ls_1, att_ls_1 = fit_conventional(single_signal_ls, plds_np, simulator.params)
        results_collector['LS (1-repeat)'].append([cbf_ls_1, att_ls_1])
        
        avg_signal_nn = np.mean(repeats_data, axis=0).flatten(order='F')
        cbf_nn_4, att_nn_4 = predict_nn(models, avg_signal_nn, num_plds, norm_stats, device)
        results_collector['NN (4-repeat)'].append([cbf_nn_4, att_nn_4])

        single_signal_nn = repeats_data[0].flatten(order='F')
        cbf_nn_1, att_nn_1 = predict_nn(models, single_signal_nn, num_plds, norm_stats, device)
        results_collector['NN (1-repeat)'].append([cbf_nn_1, att_nn_1])

    # 3. Collate and Analyze Full Grid Results
    logging.info("Collating full grid results without aggregation...")
    detailed_df = analyze_and_report_full_grid(results_collector, true_params_grid)
            
    return detailed_df

def fit_conventional(signal_reshaped: np.ndarray, plds: np.ndarray, params: ASLParameters) -> tuple:
    """Helper function to run the conventional NLLS fit."""
    pldti = np.column_stack([plds, plds])
    init_cbf_ls, init_att_ls = 50.0 / 6000.0, 1500.0
    try:
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
            pldti, signal_reshaped, [init_cbf_ls, init_att_ls],
            T1_artery=params.T1_artery, T_tau=params.T_tau, T2_factor=params.T2_factor,
            alpha_BS1=params.alpha_BS1, alpha_PCASL=params.alpha_PCASL, alpha_VSASL=params.alpha_VSASL
        )
        return beta[0] * 6000.0, beta[1]
    except Exception:
        return np.nan, np.nan

def predict_nn(models: list, signal_flat: np.ndarray, num_plds: int, norm_stats: dict, device: torch.device) -> tuple:
    """Helper function to run NN prediction."""
    engineered_features = engineer_signal_features(signal_flat.reshape(1, -1), num_plds)
    nn_input_unnorm = np.concatenate([signal_flat, engineered_features.flatten()])
    
    nn_input_norm = apply_normalization_to_input_flat(nn_input_unnorm, norm_stats, num_plds, has_m0=False)
    input_tensor = torch.FloatTensor(nn_input_norm).unsqueeze(0).to(device)
    
    cbf_preds_norm, att_preds_norm = [], []
    with torch.no_grad():
        for model in models:
            # The model returns 6 values; we only need the first two mean predictions here.
            cbf_m, att_m, _, _, _, _ = model(input_tensor)
            cbf_preds_norm.append(cbf_m.item())
            att_preds_norm.append(att_m.item())
            
    ensemble_cbf_norm = np.mean(cbf_preds_norm)
    ensemble_att_norm = np.mean(att_preds_norm)
    
    cbf_denorm, att_denorm, _, _ = denormalize_predictions(
        ensemble_cbf_norm, ensemble_att_norm, None, None, norm_stats
    )
    return cbf_denorm, att_denorm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained ASL model ensemble against conventional Least-Squares (LS) fitting. "
                                                 "This script compares performance under two conditions for each method: "
                                                 "1) using a single, noisy data repeat, and 2) using data averaged over 4 repeats.")
    parser.add_argument("results_dir", type=str, help="Path to the results directory containing models, config, and norm_stats.")
    args = parser.parse_args()

    # 1. Load artifacts
    models, config, norm_stats = load_artifacts(Path(args.results_dir))
    
    # 2. Run the specific evaluation scenarios and get the detailed, un-aggregated DataFrame
    results_df = run_evaluation(models, config, norm_stats)
    
    # 3. Display the final detailed comparison table
    print("\n" + "="*120)
    print("--- Full Landscape Benchmark Results (Disaggregated by CBF and ATT) ---")
    print("="*120)
    if not results_df.empty:
        # Sort for better readability
        sorted_df = results_df.sort_values(by=['True ATT', 'True CBF', 'Method'])
        # Set pandas options to display the full DataFrame without truncation
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(sorted_df.to_string(index=False, float_format="%.2f"))
        # Save the detailed report to a CSV file for further analysis
        output_file = Path(args.results_dir) / 'full_landscape_evaluation.csv'
        sorted_df.to_csv(output_file, index=False, float_format='%.3f')
        print(f"\nDetailed report saved to: {output_file}")
    else:
        print("No results were generated.")
    print("="*120)