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
from main import engineer_signal_features
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from comparison_framework import apply_normalization_to_input_flat, denormalize_predictions

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
    model_param_keys = inspect.signature(EnhancedASLNet).parameters.keys()
    filtered_kwargs = {k: v for k, v in config.items() if k in model_param_keys}
    
    for i in range(config.get('n_ensembles', 5)):
        model_path = models_dir / f'ensemble_model_{i}.pt'
        if not model_path.exists():
            logging.warning(f"Model file not found: {model_path}. Skipping.")
            continue
        model = EnhancedASLNet(input_size=base_input_size, **filtered_kwargs)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        ensemble_models.append(model)
    if not ensemble_models: raise FileNotFoundError("No models could be loaded.")
    logging.info(f"Successfully loaded {len(ensemble_models)} models for the ensemble.")
        
    return ensemble_models, config, norm_stats

def run_evaluation(models: list, config: dict, norm_stats: dict):
    """Generates data and runs the 4 specified comparison scenarios across the landscape."""
    
    # 1. Setup
    sim_params = ASLParameters(**{k: v for k, v in config.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=sim_params)
    plds_np = np.array(config['pld_values'])
    num_plds = len(plds_np)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models: model.to(device)

    # Define the test landscape
    cbf_grid = np.linspace(20, 80, 8)
    att_grid = np.linspace(800, 3500, 10)
    true_params_grid = np.array(np.meshgrid(cbf_grid, att_grid)).T.reshape(-1, 2)
    n_points = len(true_params_grid)
    logging.info(f"Testing on a systematic {len(cbf_grid)}x{len(att_grid)} CBF/ATT grid ({n_points} points).")
    
    # 2. Data Generation and Prediction Loop
    results_collector = {
        'NLLS (4-repeat avg)': [], 'NLLS (1-repeat)': [],
        'NN (4-repeat avg)': [], 'NN (1-repeat)': []
    }
    
    for i in tqdm(range(n_points), desc="Processing Grid Points"):
        true_cbf, true_att = true_params_grid[i]
        
        # Generate 4 independent noisy repeats
        repeats_data = []
        for _ in range(4):
            data_dict = simulator.generate_synthetic_data(
                plds_np, np.array([true_att]), n_noise=1, tsnr=8.0, cbf_val=true_cbf
            )
            repeats_data.append(data_dict['MULTIVERSE'][0, 0, :, :])

        # --- Process the 4 Scenarios ---
        avg_signal_nlls = np.mean(repeats_data, axis=0)
        cbf_nlls_4, att_nlls_4 = fit_conventional(avg_signal_nlls, plds_np, simulator.params)
        results_collector['NLLS (4-repeat avg)'].append([cbf_nlls_4, att_nlls_4])

        single_signal_nlls = repeats_data[0]
        cbf_nlls_1, att_nlls_1 = fit_conventional(single_signal_nlls, plds_np, simulator.params)
        results_collector['NLLS (1-repeat)'].append([cbf_nlls_1, att_nlls_1])
        
        avg_signal_nn = np.mean(repeats_data, axis=0).flatten()
        cbf_nn_4, att_nn_4 = predict_nn(models, avg_signal_nn, num_plds, norm_stats, device)
        results_collector['NN (4-repeat avg)'].append([cbf_nn_4, att_nn_4])

        single_signal_nn = repeats_data[0].flatten()
        cbf_nn_1, att_nn_1 = predict_nn(models, single_signal_nn, num_plds, norm_stats, device)
        results_collector['NN (1-repeat)'].append([cbf_nn_1, att_nn_1])

    # 3. Collate and Analyze Results by ATT Range
    final_results = []
    att_ranges = config.get('att_ranges_config', [])
    
    for att_min, att_max, range_name in att_ranges:
        # Create a boolean mask to select points within the current ATT range
        mask = (true_params_grid[:, 1] >= att_min) & (true_params_grid[:, 1] < att_max)
        if not np.any(mask):
            continue

        for method, preds in results_collector.items():
            preds_np = np.array(preds)[mask] # Apply mask to get predictions for this range
            true_params_range = true_params_grid[mask]
            
            pred_cbf, pred_att = preds_np[:, 0], preds_np[:, 1]
            valid_mask = ~np.isnan(pred_cbf) & ~np.isnan(pred_att)
            
            # Calculate metrics only on valid predictions within the range
            cbf_rmse = np.sqrt(np.mean((pred_cbf[valid_mask] - true_params_range[valid_mask, 0])**2))
            att_rmse = np.sqrt(np.mean((pred_att[valid_mask] - true_params_range[valid_mask, 1])**2))
            cbf_bias = np.mean(pred_cbf[valid_mask] - true_params_range[valid_mask, 0])
            att_bias = np.mean(pred_att[valid_mask] - true_params_range[valid_mask, 1])
            success_rate = np.mean(valid_mask) * 100
            
            final_results.append({
                'ATT Range': range_name,
                'Method': method,
                'CBF Bias': cbf_bias,
                'CBF RMSE': cbf_rmse,
                'ATT Bias (ms)': att_bias,
                'ATT RMSE (ms)': att_rmse,
                'Success Rate (%)': success_rate
            })
            
    return pd.DataFrame(final_results)

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
            cbf_m, att_m, _, _ = model(input_tensor)
            cbf_preds_norm.append(cbf_m.item())
            att_preds_norm.append(att_m.item())
            
    ensemble_cbf_norm = np.mean(cbf_preds_norm)
    ensemble_att_norm = np.mean(att_preds_norm)
    
    cbf_denorm, att_denorm, _, _ = denormalize_predictions(
        ensemble_cbf_norm, ensemble_att_norm, None, None, norm_stats
    )
    return cbf_denorm, att_denorm

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a pre-trained ASL model ensemble against conventional methods in 1-repeat vs 4-repeat scenarios across the full ATT landscape.")
    parser.add_argument("results_dir", type=str, help="Path to the results directory containing models, config, and norm_stats.")
    args = parser.parse_args()

    # 1. Load artifacts
    models, config, norm_stats = load_artifacts(Path(args.results_dir))
    
    # 2. Run the specific evaluation scenarios
    results_df = run_evaluation(models, config, norm_stats)
    
    # 3. Display final comparison table
    print("\n" + "="*80)
    print("--- 1-Repeat vs. 4-Repeat Benchmark Results (Across ATT Ranges) ---")
    print("="*80)
    if not results_df.empty:
        # Sort for better readability
        sorted_df = results_df.sort_values(by=['ATT Range', 'Method'])
        print(sorted_df.to_string(index=False, float_format="%.2f"))
    else:
        print("No results were generated.")
    print("="*80)