# final_benchmark.py
import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# --- Self-contained project modules ---
# These are included to make the script standalone and easy to run.

from asl_simulation import ASLParameters
from enhanced_simulation import RealisticASLSimulator, PhysiologicalVariation
from enhanced_asl_network import EnhancedASLNet
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from utils import engineer_signal_features

# --- Standalone Artifact & Prediction Functions ---

def load_artifacts(artifacts_dir: Path) -> Tuple[List[EnhancedASLNet], Dict, Dict]:
    """Loads the model ensemble, config, and normalization stats."""
    print(f"--> Loading artifacts from: {artifacts_dir}")
    config_path = artifacts_dir / 'research_config.json'
    if not config_path.exists(): raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, 'r') as f: config = json.load(f)

    norm_stats_path = artifacts_dir / 'norm_stats.json'
    if not norm_stats_path.exists(): raise FileNotFoundError(f"Norm stats not found: {norm_stats_path}")
    with open(norm_stats_path, 'r') as f: norm_stats = json.load(f)

    models_dir = artifacts_dir / 'trained_models'
    if not models_dir.exists(): raise FileNotFoundError(f"Trained models directory not found: {models_dir}")
        
    models = []
    num_plds = len(config.get('pld_values', []))
    base_input_size = num_plds * 2 + 4

    for model_path in models_dir.glob('ensemble_model_*.pt'):
        model = EnhancedASLNet(input_size=base_input_size, **config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        models.append(model)
        
    if not models: raise FileNotFoundError(f"No '.pt' model files found in {models_dir}")
    print(f"--> Successfully loaded {len(models)} models, config, and norm_stats.")
    return models, config, norm_stats


def apply_normalization_to_input(flat_signal: np.ndarray, norm_stats: Dict, num_plds: int) -> np.ndarray:
    """Normalizes a single flat input vector for the NN."""
    raw_signal_part = flat_signal[:num_plds * 2]
    other_features_part = flat_signal[num_plds * 2:]
    pcasl_norm = (raw_signal_part[:num_plds] - norm_stats['pcasl_mean']) / (np.array(norm_stats['pcasl_std']) + 1e-6)
    vsasl_norm = (raw_signal_part[num_plds:] - norm_stats['vsasl_mean']) / (np.array(norm_stats['vsasl_std']) + 1e-6)
    return np.concatenate([pcasl_norm, vsasl_norm, other_features_part])


def denormalize_predictions(cbf_pred_norm: np.ndarray, att_pred_norm: np.ndarray, norm_stats: Dict) -> Tuple[float, float]:
    """De-normalizes NN predictions back to physical units."""
    cbf_pred_denorm = cbf_pred_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att_pred_denorm = att_pred_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    return float(cbf_pred_denorm), float(att_pred_denorm)


def predict_nn(models: List[EnhancedASLNet], signal_flat: np.ndarray, plds_from_data: np.ndarray, config: Dict, norm_stats: Dict, device: torch.device) -> Tuple[float, float]:
    """Runs inference with the NN ensemble, including the critical resampling step."""
    model_plds_list = config.get('pld_values', [])
    num_model_plds = len(model_plds_list)
    num_data_plds = len(plds_from_data)

    resampled_signal = np.zeros(num_model_plds * 2)
    target_indices, source_indices = [], []
    for i, pld in enumerate(plds_from_data):
        try:
            target_idx = model_plds_list.index(int(pld))
            target_indices.append(target_idx)
            source_indices.append(i)
        except ValueError:
            pass
    
    source_indices, target_indices = np.array(source_indices), np.array(target_indices)
    if source_indices.size > 0:
        resampled_signal[target_indices] = signal_flat[source_indices]
        resampled_signal[target_indices + num_model_plds] = signal_flat[source_indices + num_data_plds]
    
    num_plds_for_features = num_model_plds
    engineered_features = engineer_signal_features(resampled_signal, num_plds_for_features)
    nn_input_unnorm = np.concatenate([resampled_signal, engineered_features])
    
    nn_input_norm = apply_normalization_to_input(nn_input_unnorm, norm_stats, num_plds_for_features)
    input_tensor = torch.FloatTensor(nn_input_unnorm).unsqueeze(0).to(device)

    cbf_preds_norm, att_preds_norm = [], []
    with torch.no_grad():
        for model in models:
            cbf_mean, att_mean, _, _, _, _ = model(input_tensor)
            cbf_preds_norm.append(cbf_mean.item())
            att_preds_norm.append(att_mean.item())
            
    ensemble_cbf_norm, ensemble_att_norm = np.mean(cbf_preds_norm), np.mean(att_preds_norm)
    return denormalize_predictions(ensemble_cbf_norm, ensemble_att_norm, norm_stats)


def fit_ls(signal_flat: np.ndarray, plds: np.ndarray, config: Dict) -> Tuple[float, float]:
    """Runs the conventional Least-Squares fit."""
    num_plds = len(plds)
    signal_reshaped = np.vstack([signal_flat[:num_plds], signal_flat[num_plds:]]).T
    pldti = np.column_stack([plds, plds])
    ls_params = {k: v for k, v in config.items() if k in ['T1_artery', 'T_tau', 'T2_factor', 'alpha_BS1', 'alpha_PCASL', 'alpha_VSASL']}
    init_guess = [50.0 / 6000.0, 1500.0]
    try:
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signal_reshaped, init_guess, **ls_params)
        return beta[0] * 6000.0, beta[1]
    except Exception:
        return np.nan, np.nan


def calculate_metrics(df_group: pd.DataFrame) -> pd.Series:
    """Calculates a set of performance metrics for a given group of data."""
    metrics = {}
    for method in ['ls', 'nn']:
        cbf_err = df_group[f'{method}_pred_cbf'] - df_group['true_cbf']
        att_err = df_group[f'{method}_pred_att'] - df_group['true_att']
        valid_mask = ~cbf_err.isna()
        if valid_mask.sum() == 0:
            metrics.update({f'{method.upper()} {_met}': np.nan for _met in ["CBF MAE", "ATT MAE", "CBF RMSE", "ATT RMSE", "CBF Bias", "ATT Bias"]})
            metrics[f'{method.upper()} Fit Success %'] = 0.0
        else:
            metrics[f'{method.upper()} CBF MAE'] = cbf_err[valid_mask].abs().mean()
            metrics[f'{method.upper()} ATT MAE'] = att_err[valid_mask].abs().mean()
            metrics[f'{method.upper()} CBF RMSE'] = np.sqrt((cbf_err[valid_mask]**2).mean())
            metrics[f'{method.upper()} ATT RMSE'] = np.sqrt((att_err[valid_mask]**2).mean())
            metrics[f'{method.upper()} CBF Bias'] = cbf_err[valid_mask].mean()
            metrics[f'{method.upper()} ATT Bias'] = att_err[valid_mask].mean()
            metrics[f'{method.upper()} Fit Success %'] = (valid_mask.sum() / len(df_group)) * 100
    return pd.Series(metrics)


def main():
    """Main execution function for the final benchmark script."""
    parser = argparse.ArgumentParser(
        description="Run a final, definitive benchmark of a trained NN against LS fitting on maximally realistic simulated data.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model_artifacts_dir", type=str, help="Path to the trained model's artifacts directory (e.g., final_training_run_v10).")
    args = parser.parse_args()
    
    artifacts_path = Path(args.model_artifacts_dir)
    try:
        models, config, norm_stats = load_artifacts(artifacts_path)
    except Exception as e:
        print(f"\n[FATAL ERROR] Could not load artifacts: {e}", file=sys.stderr)
        sys.exit(1)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models:
        model.to(device)
    
    # --- Correct, Realistic Data Generation Loop ---
    print("\n--- Generating Maximally Realistic Simulation Dataset ---")
    plds_np = np.array(config.get('pld_values', []))
    sim_params = ASLParameters(**{k: v for k, v in config.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=sim_params)
    physio_var = simulator.physio_var

    n_samples_per_condition = 125 # Results in 125 * 4 * 5 = 2500 total samples
    noise_levels = [3.0, 5.0, 8.0, 10.0, 15.0]
    conditions = ['healthy', 'stroke', 'tumor', 'elderly']
    
    condition_map = {
        'healthy': (physio_var.cbf_range, physio_var.att_range),
        'stroke': (physio_var.stroke_cbf_range, physio_var.stroke_att_range),
        'tumor': (physio_var.tumor_cbf_range, physio_var.tumor_att_range),
        'elderly': (physio_var.elderly_cbf_range, physio_var.elderly_att_range)
    }
    
    results_list = []
    total_iterations = n_samples_per_condition * len(conditions) * len(noise_levels)
    pbar = tqdm(total=total_iterations, desc="Generating & Evaluating Samples")
    
    for condition in conditions:
        for _ in range(n_samples_per_condition):
            cbf_r, att_r = condition_map[condition]
            true_cbf = np.random.uniform(*cbf_r)
            true_att = np.random.uniform(*att_r)
            true_t1_artery = np.random.uniform(*physio_var.t1_artery_range)
            
            perturbed_t_tau = sim_params.T_tau * (1 + np.random.uniform(*physio_var.t_tau_perturb_range))
            perturbed_alpha_pcasl = np.clip(sim_params.alpha_PCASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.1)
            perturbed_alpha_vsasl = np.clip(sim_params.alpha_VSASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.0)
            
            vsasl_clean = simulator._generate_vsasl_signal(plds_np, true_att, true_cbf, true_t1_artery, perturbed_alpha_vsasl)
            pcasl_clean = simulator._generate_pcasl_signal(plds_np, true_att, true_cbf, true_t1_artery, perturbed_t_tau, perturbed_alpha_pcasl)

            for snr in noise_levels:
                pcasl_noisy = simulator.add_realistic_noise(pcasl_clean, snr=snr)
                vsasl_noisy = simulator.add_realistic_noise(vsasl_clean, snr=snr)
                signal = np.concatenate([pcasl_noisy, vsasl_noisy])
                
                cbf_ls, att_ls = fit_ls(signal, plds_np, config)
                cbf_nn, att_nn = predict_nn(models, signal, plds_np, config, norm_stats, device)
                
                results_list.append({
                    'true_cbf': true_cbf, 'true_att': true_att,
                    'ls_pred_cbf': cbf_ls, 'ls_pred_att': att_ls,
                    'nn_pred_cbf': cbf_nn, 'nn_pred_att': att_nn,
                    'condition': condition, 'noise_level': snr
                })
                pbar.update(1)
    pbar.close()

    # --- Analysis and Reporting ---
    df_results = pd.DataFrame(results_list)
    pd.set_option('display.width', 120)
    pd.set_option('display.float_format', '{:.2f}'.format)
    
    print("\n\n" + "="*80)
    print("FINAL BENCHMARK RESULTS")
    print("="*80)
    
    print("\nTABLE 1: OVERALL PERFORMANCE SUMMARY")
    overall_summary = calculate_metrics(df_results).to_frame(name='Overall').T
    print(overall_summary)
    
    print("\nTABLE 2: PERFORMANCE STRATIFIED BY ARTERIAL TRANSIT TIME (ATT)")
    att_bins = [0, 1500, 2500, np.inf]
    att_labels = ["Short (<1.5s)", "Medium (1.5-2.5s)", "Long (>2.5s)"]
    df_results['att_bin'] = pd.cut(df_results['true_att'], bins=att_bins, labels=att_labels, right=False)
    att_summary = df_results.groupby('att_bin', observed=False).apply(calculate_metrics)
    print(att_summary)
    
    print("\nTABLE 3: PERFORMANCE STRATIFIED BY PHYSIOLOGICAL CONDITION")
    condition_summary = df_results.groupby('condition').apply(calculate_metrics)
    print(condition_summary)
    
    print("\nTABLE 4: PERFORMANCE STRATIFIED BY TEMPORAL SNR (tSNR)")
    noise_summary = df_results.groupby('noise_level').apply(calculate_metrics)
    print(noise_summary)
    
    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)

if __name__ == '__main__':
    main()