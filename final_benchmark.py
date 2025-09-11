# final_benchmark.py
#
# A definitive, publication-ready benchmarking script to compare the trained
# EnhancedASLNet against the conventional Least-Squares (LS) fitting method.
# THIS VERSION USES A ROBUST, GRID-SEARCH-INITIALIZED LS BASELINE.

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import json
import argparse
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# --- Import from your project codebase ---
from enhanced_asl_network import EnhancedASLNet
from enhanced_simulation import RealisticASLSimulator, ASLParameters, PhysiologicalVariation
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
# --- MODIFIED: Import the robust initializer and other utils ---
from utils import engineer_signal_features, get_grid_search_initial_guess
from predict_on_invivo import apply_normalization_vectorized, denormalize_predictions # Re-use utility functions

# --- Configuration ---
NUM_SAMPLES = 5000  # Number of unique voxels to simulate for the benchmark
NOISE_LEVELS = [3.0, 5.0, 8.0, 10.0, 15.0] # tSNR levels to test
CONDITIONS = ['healthy', 'stroke', 'tumor', 'elderly']

def load_artifacts(model_results_root: Path) -> tuple:
    """Robustly loads the model ensemble, final config, and norm stats."""
    print(f"--> Loading artifacts from: {model_results_root}")
    try:
        with open(model_results_root / 'research_config.json', 'r') as f:
            config = json.load(f)
        with open(model_results_root / 'norm_stats.json', 'r') as f:
            norm_stats = json.load(f)

        models = []
        models_dir = model_results_root / 'trained_models'
        num_plds = len(config['pld_values'])
        base_input_size = num_plds * 2 + 4

        for model_path in models_dir.glob('ensemble_model_*.pt'):
            model = EnhancedASLNet(input_size=base_input_size, **config)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            models.append(model)

        if not models:
            raise FileNotFoundError("No models found in trained_models folder.")
        print(f"--> Successfully loaded {len(models)} models, config, and norm_stats.")
        return models, config, norm_stats
    except Exception as e:
        print(f"[FATAL ERROR] Could not load artifacts: {e}. Exiting.")
        sys.exit(1)

def calculate_metrics(df_group):
    """Calculates a comprehensive set of metrics for a given group of results."""
    metrics = {}
    
    # --- Define Clinical Tolerances --- #
    CBF_TOLERANCE = 5.0   # mL/100g/min
    ATT_TOLERANCE = 150.0 # ms

    for model in ['ls', 'nn']:
        for param in ['cbf', 'att']:
            true_col = f'true_{param}'
            pred_col = f'{model}_{param}_pred'
            
            # Filter out failed fits for LS
            valid_preds = df_group.dropna(subset=[pred_col])
            if valid_preds.empty:
                metrics[f'{model.upper()} {param.upper()} MAE'] = np.nan
                metrics[f'{model.upper()} {param.upper()} MedAE'] = np.nan
                metrics[f'{model.upper()} {param.upper()} RMSE'] = np.nan
                metrics[f'{model.upper()} {param.upper()} Bias'] = np.nan
                metrics[f'{model.upper()} {param.upper()} SD'] = np.nan
                metrics[f'{model.upper()} {param.upper()} Acc %'] = np.nan
                continue

            errors = valid_preds[pred_col] - valid_preds[true_col]
            metrics[f'{model.upper()} {param.upper()} MAE'] = np.mean(np.abs(errors))
            metrics[f'{model.upper()} {param.upper()} MedAE'] = np.median(np.abs(errors))
            metrics[f'{model.upper()} {param.upper()} RMSE'] = np.sqrt(np.mean(errors**2))
            metrics[f'{model.upper()} {param.upper()} Bias'] = np.mean(errors)
            metrics[f'{model.upper()} {param.upper()} SD'] = np.std(valid_preds[pred_col])
            
            # --- Calculate Accuracy within Tolerance --- #
            tolerance = CBF_TOLERANCE if param == 'cbf' else ATT_TOLERANCE
            correct_predictions = np.abs(errors) <= tolerance
            accuracy_percent = (np.sum(correct_predictions) / len(valid_preds)) * 100
            metrics[f'{model.upper()} {param.upper()} Acc %'] = accuracy_percent

    # Calculate CoV (Precision)
    for model in ['ls', 'nn']:
        for param in ['cbf', 'att']:
            mean_val = df_group[f'{model}_{param}_pred'].mean()
            sd_val = metrics.get(f'{model.upper()} {param.upper()} SD', np.nan)
            metrics[f'{model.upper()} {param.upper()} CoV'] = (sd_val / mean_val) if mean_val != 0 else np.nan

    metrics['Fit Success %'] = (df_group['ls_fit_success'].sum() / len(df_group)) * 100
    return pd.Series(metrics)

def main(model_dir: Path, output_dir: Path):
    """Main execution function."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load all necessary trained artifacts
    models, config, norm_stats = load_artifacts(model_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for model in models:
        model.to(device)

    # 2. Setup the hyper-realistic simulation environment
    print("\n--- Generating Maximally Realistic Simulation Dataset ---")
    asl_params = ASLParameters(**{k: v for k, v in config.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params)
    physio_var = PhysiologicalVariation()
    plds = np.array(config['pld_values'])
    num_plds = len(plds)

    # 3. Main Generation and Evaluation Loop
    results = []
    # --- Pre-extract the ASL parameters dictionary for the LS fitter ---
    ls_params = {k:v for k,v in config.items() if k in ['T1_artery','T_tau','T2_factor','alpha_BS1','alpha_PCASL','alpha_VSASL']}
    
    for _ in tqdm(range(NUM_SAMPLES), desc="Generating & Evaluating Samples"):
        # Sample ground truth parameters from realistic distributions
        true_cbf = np.random.uniform(*physio_var.cbf_range)
        true_att = np.random.uniform(*physio_var.att_range)
        true_t1_artery = np.random.uniform(*physio_var.t1_artery_range)
        tsnr = np.random.choice(NOISE_LEVELS)
        condition = np.random.choice(CONDITIONS)

        # Generate clean signal with parameter perturbations
        perturbed_t_tau = simulator.params.T_tau * (1 + np.random.uniform(*physio_var.t_tau_perturb_range))
        perturbed_alpha_pcasl = np.clip(simulator.params.alpha_PCASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.1)
        perturbed_alpha_vsasl = np.clip(simulator.params.alpha_VSASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.0)
        
        vsasl_clean = simulator._generate_vsasl_signal(plds, true_att, true_cbf, true_t1_artery, perturbed_alpha_vsasl)
        pcasl_clean = simulator._generate_pcasl_signal(plds, true_att, true_cbf, true_t1_artery, perturbed_t_tau, perturbed_alpha_pcasl)

        # Apply the full, complex, realistic noise model
        pcasl_noisy = simulator.add_realistic_noise(pcasl_clean, snr=tsnr)
        vsasl_noisy = simulator.add_realistic_noise(vsasl_clean, snr=tsnr)
        noisy_signal = np.concatenate([pcasl_noisy, vsasl_noisy])

        # --- A) Evaluate with Conventional LS Fitting (ROBUST VERSION) ---
        ls_cbf_pred, ls_att_pred, ls_success = np.nan, np.nan, False
        try:
            pldti = np.column_stack([plds, plds])
            signal_reshaped = noisy_signal.reshape((len(plds), 2), order='F')
            
            # === MODIFICATION: Replace hard-coded guess with robust initializer ===
            init_guess = get_grid_search_initial_guess(noisy_signal, plds, ls_params)
            # ======================================================================

            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signal_reshaped, init_guess, **ls_params)
            ls_cbf_pred = beta[0] * 6000.0
            ls_att_pred = beta[1]
            ls_success = True
        except Exception:
            pass # Keep results as NaN on failure

        # --- B) Evaluate with our Neural Network ---
        eng_feats = engineer_signal_features(noisy_signal, num_plds)
        nn_input = np.concatenate([noisy_signal, eng_feats])
        norm_input = apply_normalization_vectorized(nn_input.reshape(1, -1), norm_stats, num_plds)
        input_tensor = torch.FloatTensor(norm_input).to(device)
        
        with torch.no_grad():
            all_cbf_means, all_att_means = [], []
            for model in models:
                cbf_mean_norm, att_mean_norm, _, _, _, _ = model(input_tensor)
                all_cbf_means.append(cbf_mean_norm.cpu().numpy())
                all_att_means.append(att_mean_norm.cpu().numpy())
        
        ensemble_cbf_norm = np.mean(all_cbf_means)
        ensemble_att_norm = np.mean(all_att_means)
        nn_cbf_pred, nn_att_pred = denormalize_predictions(ensemble_cbf_norm, ensemble_att_norm, norm_stats)
        
        results.append({
            'true_cbf': true_cbf, 'true_att': true_att, 'tsnr': tsnr, 'condition': condition,
            'ls_cbf_pred': ls_cbf_pred, 'ls_att_pred': ls_att_pred, 'ls_fit_success': ls_success,
            'nn_cbf_pred': nn_cbf_pred, 'nn_att_pred': nn_att_pred,
        })

    # 4. Analyze and Present Results
    df_results = pd.DataFrame(results)
    df_results.to_csv(output_dir / 'full_benchmark_results_robust_ls.csv', index=False)

    print("\n\n" + "="*80)
    print("FINAL BENCHMARK RESULTS (vs. ROBUST LS BASELINE)")
    print("="*80 + "\n")

    # --- Table 1: Overall ---
    overall_summary = calculate_metrics(df_results)
    print("TABLE 1: OVERALL PERFORMANCE SUMMARY")
    print(overall_summary.to_frame().T.to_string())
    overall_summary.to_frame().T.to_csv(output_dir / 'summary_overall_robust_ls.csv')

    # --- Table 2: Stratified by ATT ---
    att_bins = pd.cut(df_results['true_att'], [0, 1500, 2500, 4000], labels=['Short (<1.5s)', 'Medium (1.5-2.5s)', 'Long (>2.5s)'])
    att_summary = df_results.groupby(att_bins, observed=False).apply(calculate_metrics)
    print("\nTABLE 2: PERFORMANCE STRATIFIED BY ARTERIAL TRANSIT TIME (ATT)")
    print(att_summary.to_string())
    att_summary.to_csv(output_dir / 'summary_by_att_robust_ls.csv')
    
    # --- Table 3: Stratified by Condition ---
    condition_summary = df_results.groupby('condition').apply(calculate_metrics)
    print("\nTABLE 3: PERFORMANCE STRATIFIED BY PHYSIOLOGICAL CONDITION")
    print(condition_summary.to_string())
    condition_summary.to_csv(output_dir / 'summary_by_condition_robust_ls.csv')

    # --- Table 4: Stratified by Noise ---
    noise_summary = df_results.groupby('tsnr').apply(calculate_metrics)
    print("\nTABLE 4: PERFORMANCE STRATIFIED BY TEMPORAL SNR (tSNR)")
    print(noise_summary.to_string())
    noise_summary.to_csv(output_dir / 'summary_by_tsnr_robust_ls.csv')

    # --- Figure 1: Publication-Ready Plots (vs. ATT) ---
    print("\n--- Generating Publication-Ready Plots ---")
    df_sorted = df_results.sort_values('true_att').dropna()
    window_size = len(df_sorted) // 20 # Rolling window for smoothing

    fig, axes = plt.subplots(3, 2, figsize=(16, 20), sharex=True)
    sns.set_style("whitegrid")
    
    plot_params = {
        'ls': {'label': 'Conventional LS (Robust Init.)', 'color': 'orangered', 'linestyle': '--'},
        'nn': {'label': 'Our Method (NN)', 'color': 'royalblue', 'linestyle': '-'}
    }

    # Row 1: Bias
    for model in ['ls', 'nn']:
        # CBF Bias
        bias_cbf = (df_sorted[f'{model}_cbf_pred'] - df_sorted['true_cbf']).rolling(window_size, center=True).mean()
        axes[0, 0].plot(df_sorted['true_att'], bias_cbf, **plot_params[model])
        # ATT Bias
        bias_att = (df_sorted[f'{model}_att_pred'] - df_sorted['true_att']).rolling(window_size, center=True).mean()
        axes[0, 1].plot(df_sorted['true_att'], bias_att, **plot_params[model])
    
    axes[0, 0].set_title('CBF Accuracy (Bias)', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Bias (mL/100g/min)')
    axes[0, 0].axhline(0, color='k', linestyle=':', alpha=0.7)
    axes[0, 1].set_title('ATT Accuracy (Bias)', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylabel('Bias (ms)')
    axes[0, 1].axhline(0, color='k', linestyle=':', alpha=0.7)

    # Row 2: CoV (Precision)
    for model in ['ls', 'nn']:
        # CBF CoV
        cov_cbf = (df_sorted[f'{model}_cbf_pred'].rolling(window_size).std() / df_sorted[f'{model}_cbf_pred'].rolling(window_size).mean()) * 100
        axes[1, 0].plot(df_sorted['true_att'], cov_cbf, **plot_params[model])
        # ATT CoV
        cov_att = (df_sorted[f'{model}_att_pred'].rolling(window_size).std() / df_sorted[f'{model}_att_pred'].rolling(window_size).mean()) * 100
        axes[1, 1].plot(df_sorted['true_att'], cov_att, **plot_params[model])

    axes[1, 0].set_title('CBF Precision (Coefficient of Variation)', fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel('CoV (%)')
    axes[1, 0].set_ylim(bottom=0)
    axes[1, 1].set_title('ATT Precision (Coefficient of Variation)', fontsize=14, fontweight='bold')
    axes[1, 1].set_ylabel('CoV (%)')
    axes[1, 1].set_ylim(bottom=0)

    # Row 3: nRMSE (Overall Performance)
    for model in ['ls', 'nn']:
        # CBF nRMSE
        rmse_cbf = ((df_sorted[f'{model}_cbf_pred'] - df_sorted['true_cbf'])**2).rolling(window_size).mean().apply(np.sqrt)
        nrmse_cbf = (rmse_cbf / df_sorted['true_cbf'].rolling(window_size).mean()) * 100
        axes[2, 0].plot(df_sorted['true_att'], nrmse_cbf, **plot_params[model])
        # ATT nRMSE
        rmse_att = ((df_sorted[f'{model}_att_pred'] - df_sorted['true_att'])**2).rolling(window_size).mean().apply(np.sqrt)
        nrmse_att = (rmse_att / df_sorted['true_att'].rolling(window_size).mean()) * 100
        axes[2, 1].plot(df_sorted['true_att'], nrmse_att, **plot_params[model])

    axes[2, 0].set_title('CBF Overall Error (nRMSE)', fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel('nRMSE (%)')
    axes[2, 0].set_xlabel('True Arterial Transit Time (ms)')
    axes[2, 0].set_ylim(bottom=0)
    axes[2, 1].set_title('ATT Overall Error (nRMSE)', fontsize=14, fontweight='bold')
    axes[2, 1].set_ylabel('nRMSE (%)')
    axes[2, 1].set_xlabel('True Arterial Transit Time (ms)')
    axes[2, 1].set_ylim(bottom=0)

    for ax in axes.flat:
        ax.legend()
        ax.set_xlim(500, 4000)

    fig.suptitle('Performance Comparison on Realistic Simulated Data', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig_path = output_dir / 'figure_performance_vs_att_robust_ls.png'
    plt.savefig(fig_path, dpi=300)
    print(f"--> Plots saved to {fig_path}")

    print("\n" + "="*80)
    print("BENCHMARK COMPLETE")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the definitive benchmark for the ASL Multiverse project.")
    parser.add_argument("model_artifacts_dir", type=str, help="Path to the directory containing the final trained model artifacts (e.g., 'final_training_run_v12').")
    parser.add_argument("output_dir", type=str, nargs='?', default=None, help="Directory to save the benchmark results. If not provided, a 'final_benchmark_results' directory will be created.")
    args = parser.parse_args()

    if args.output_dir:
        output_path = Path(args.output_dir)
    else:
        output_path = Path('./final_benchmark_results')
        
    main(Path(args.model_artifacts_dir), output_path)