# FILE: diagnose_simulation_and_model.py
# FINAL CORRECTED VERSION (V5 Architecture)
# This version is updated to use the robust V5 preprocessing pipeline, ensuring
# compatibility with the new model architecture and providing a fair comparison.

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
import warnings

# --- Suppress RuntimeWarning for mean of empty slice ---
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# --- Import from project codebase ---
try:
    from asl_simulation import ASLSimulator, ASLParameters
    from enhanced_simulation import PhysiologicalVariation
    from enhanced_asl_network import DisentangledASLNet
    from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
    from utils import get_grid_search_initial_guess, engineer_signal_features
except ImportError as e:
    print(f"FATAL: Could not import necessary project files. Error: {e}")
    print("Please run this script from the root directory of the 'adikondepudi-asl-multiverse' project.")
    sys.exit(1)

# --- Configuration ---
NUM_SIMS_PER_DATAPOINT = 500 # Number of unique realizations per ground truth point

# ==============================================================================
# V5-SPECIFIC HELPER FUNCTIONS
# ==============================================================================

def preprocess_for_v5_model(raw_signal_curves: np.ndarray, norm_stats: dict, num_plds: int) -> np.ndarray:
    """
    Applies the V5 preprocessing pipeline to a single sample.
    This exactly mirrors the logic in the updated ASLInMemoryDataset.
    """
    # 1. Perform per-instance normalization on raw (noisy) curves to get shape vector
    mu = np.mean(raw_signal_curves)
    sigma = np.std(raw_signal_curves)
    shape_vector = (raw_signal_curves - mu) / (sigma + 1e-6)

    # 2. Calculate engineered features from the scale-invariant shape vector
    eng_features = engineer_signal_features(shape_vector, num_plds)

    # 3. Assemble all 6 scalar features (mu, sigma, plus TTP/COM)
    scalar_features_unnorm = np.array([mu, sigma, *eng_features])

    # 4. Standardize the scalar features using pre-computed stats
    s_mean = np.array(norm_stats['scalar_features_mean'])
    s_std = np.array(norm_stats['scalar_features_std']) + 1e-6
    scalar_features_norm = (scalar_features_unnorm - s_mean) / s_std

    # 5. Concatenate final input vector and reshape for batch processing
    final_input = np.concatenate([shape_vector, scalar_features_norm])
    return final_input.reshape(1, -1).astype(np.float32)

def denormalize_predictions(cbf_norm: float, att_norm: float, norm_stats: dict) -> tuple:
    """Denormalizes CBF and ATT predictions."""
    cbf = cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att = att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    return cbf, att

# ==============================================================================
# LOADING & PREDICTION FUNCTIONS
# ==============================================================================

def load_artifacts(model_results_root: Path) -> tuple:
    """Robustly loads the model ensemble, config, and norm stats."""
    print(f"--> Loading artifacts from: {model_results_root}")
    try:
        with open(model_results_root / 'research_config.json', 'r') as f:
            config = json.load(f)
        with open(model_results_root / 'norm_stats.json', 'r') as f:
            norm_stats = json.load(f)

        models = []
        models_dir = model_results_root / 'trained_models'
        num_plds = len(config['pld_values'])
        
        model_class = DisentangledASLNet
        # V5 FIX: Input size is num_plds*2 (shape vector) + 6 (scalar features)
        base_input_size = num_plds * 2 + 6

        for model_path in models_dir.glob('ensemble_model_*.pt'):
            model = model_class(mode='regression', input_size=base_input_size, **config)
            model.to(dtype=torch.bfloat16)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            models.append(model)

        if not models:
            raise FileNotFoundError("No models found in trained_models folder.")
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for model in models: model.to(device)

        print(f"--> Successfully loaded {len(models)} models, config, and norm_stats to device '{device}'.")
        return models, config, norm_stats, device
    except Exception as e:
        print(f"[FATAL ERROR] Could not load artifacts: {e}. Exiting.")
        sys.exit(1)

def predict_nn_single_voxel(noisy_signal: np.ndarray, models: list, config: dict, norm_stats: dict, device: torch.device) -> tuple:
    """Runs a single voxel through the NN pipeline using V5 preprocessing."""
    num_plds = len(config['pld_values'])
    
    # V5 FIX: Use the new self-contained preprocessing logic
    norm_input = preprocess_for_v5_model(noisy_signal, norm_stats, num_plds)
    input_tensor = torch.from_numpy(norm_input).to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        cbf_means = [model(input_tensor)[0].cpu().float().numpy() for model in models]
        att_means = [model(input_tensor)[1].cpu().float().numpy() for model in models]
    
    nn_cbf_pred_norm = np.mean([item.item() for item in cbf_means])
    nn_att_pred_norm = np.mean([item.item() for item in att_means])

    nn_cbf_pred, nn_att_pred = denormalize_predictions(nn_cbf_pred_norm, nn_att_pred_norm, norm_stats)
    return nn_cbf_pred, nn_att_pred

def predict_ls_single_voxel(noisy_signal: np.ndarray, plds: np.ndarray, ls_params: dict) -> tuple:
    """Runs a single voxel through the robust LS pipeline."""
    try:
        init_guess = get_grid_search_initial_guess(noisy_signal, plds, ls_params)
        beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
            np.column_stack([plds, plds]), 
            noisy_signal.reshape((len(plds), 2), order='F'), 
            init_guess, 
            **ls_params
        )
        ls_cbf_pred, ls_att_pred = beta[0] * 6000.0, beta[1]
    except Exception:
        ls_cbf_pred, ls_att_pred = np.nan, np.nan
    return ls_cbf_pred, ls_att_pred

# ==============================================================================
# PHASE 1: Build and Validate the 'Ground Truth' Simulation Engine
# ==============================================================================
# This function remains unchanged as it is meant to test the ideal kinetic model.

def phase1_validate_simulation(simulator: ASLSimulator, plds: np.ndarray, output_dir: Path):
    """Generates and plots noiseless 'Golden Signals' to validate the kinetic model."""
    print("\n--- PHASE 1: Validating Simulation Engine (Generating 'Golden Signals') ---")
    
    scenarios = {
        "Healthy": {'cbf': 60.0, 'att': 1200.0},
        "Long_ATT": {'cbf': 30.0, 'att': 3500.0},
    }

    for name, params in scenarios.items():
        pcasl_clean = simulator._generate_pcasl_signal(plds, params['att'], params['cbf'], simulator.params.T1_artery, simulator.params.T_tau, simulator.params.alpha_PCASL)
        vsasl_clean = simulator._generate_vsasl_signal(plds, params['att'], params['cbf'], simulator.params.T1_artery, simulator.params.alpha_VSASL)

        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(plds, pcasl_clean, 'o-', label=f'PCASL Signal', color='royalblue')
        ax.plot(plds, vsasl_clean, 's--', label=f'VSASL Signal', color='darkorange')
        ax.set_title(f'Noiseless "Golden Signal" - {name.replace("_"," ")} Scenario\n(Ground Truth: CBF={params["cbf"]}, ATT={params["att"]}ms)', fontsize=14)
        ax.set_xlabel('Post-Labeling Delay (PLD) / Inversion Time (TI) [ms]', fontsize=12)
        ax.set_ylabel('ASL Difference Signal (a.u.)', fontsize=12)
        ax.legend()
        ax.axhline(0, color='black', linestyle=':', linewidth=0.5)
        
        fig_path = output_dir / f"A_golden_signal_{name.lower()}.png"
        plt.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  -> Saved Golden Signal plot to: {fig_path}")
    print("✅ Phase 1 Complete.")

# ==============================================================================
# PHASE 2: Systematic Evaluation on the Trusted Ground Truth (CORRECTED)
# ==============================================================================

def run_full_scenario(scenario_params: dict, simulator: ASLSimulator, plds: np.ndarray, nn_args: dict, ls_args: dict, output_dir: Path, num_sims: int):
    """
    Manages the simulation loop for a given scenario with realistic parameter uncertainty.
    This is the corrected function that prevents the "inverse crime".
    """
    scenario_name = scenario_params['name']
    print(f"\n--- PHASE 2: Running FAIR Diagnostic Scenario: {scenario_name} ---")
    print("-> Applying realistic physiological parameter variations to every simulated signal.")

    results = []
    
    physio_var = PhysiologicalVariation()
    base_params = simulator.params

    param_iterator = scenario_params['iterator']
    for gt_cbf, gt_att in tqdm(param_iterator, desc=f"Simulating {scenario_name}"):
        
        for _ in range(num_sims):
            
            true_t1_artery = np.random.uniform(*physio_var.t1_artery_range)
            perturbed_t_tau = base_params.T_tau * (1 + np.random.uniform(*physio_var.t_tau_perturb_range))
            perturbed_alpha_pcasl = np.clip(base_params.alpha_PCASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.1)
            perturbed_alpha_vsasl = np.clip(base_params.alpha_VSASL * (1 + np.random.uniform(*physio_var.alpha_perturb_range)), 0.1, 1.0)

            data_dict = simulator.generate_synthetic_data(
                plds,
                att_values=np.array([gt_att]),
                n_noise=1,
                tsnr=scenario_params['tsnr'],
                cbf_val=gt_cbf,
                t1_artery_val=true_t1_artery,
                t_tau_val=perturbed_t_tau,
                alpha_pcasl_val=perturbed_alpha_pcasl,
                alpha_vsasl_val=perturbed_alpha_vsasl
            )
            
            signals = data_dict['MULTIVERSE'][0, 0, :, :]
            noisy_signal = np.concatenate([signals[:, 0], signals[:, 1]])
            
            nn_cbf, nn_att = predict_nn_single_voxel(noisy_signal, **nn_args)
            ls_cbf, ls_att = predict_ls_single_voxel(noisy_signal, plds, **ls_args)

            results.append({
                'true_cbf': gt_cbf, 'true_att': gt_att,
                'nn_cbf_pred': nn_cbf, 'nn_att_pred': nn_att,
                'ls_cbf_pred': ls_cbf, 'ls_att_pred': ls_att,
            })

    df_results = pd.DataFrame(results)
    csv_path = output_dir / f"B_raw_results_{scenario_name}.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"  -> Saved raw simulation results to: {csv_path}")
    return csv_path

# ==============================================================================
# PHASE 3: Creating the Definitive Diagnostic Report
# ==============================================================================
# This function remains unchanged as it correctly processes the results CSV.

def phase3_generate_report(csv_path: Path, scenario_params: dict, output_dir: Path):
    """Analyzes a results CSV and generates the final report plots."""
    scenario_name = scenario_params['name']
    print(f"\n--- PHASE 3: Generating Report for Scenario: {scenario_name} ---")
    df = pd.read_csv(csv_path)

    x_param = scenario_params.get('x_axis_param', 'true_att')
    fixed_param_str = scenario_params.get('fixed_param_str', '')

    df['nn_cbf_err'] = df['nn_cbf_pred'] - df['true_cbf']
    df['ls_cbf_err'] = df['ls_cbf_pred'] - df['true_cbf']
    df['nn_att_err'] = df['nn_att_pred'] - df['true_att']
    df['ls_att_err'] = df['ls_att_pred'] - df['true_att']

    if scenario_params.get('bin_data', False):
        num_bins = 15
        df['bin'] = pd.cut(df[x_param], bins=num_bins)
        summary = df.groupby('bin', observed=True).agg({
            x_param: 'mean',
            'nn_cbf_err': ['mean', 'std'], 'ls_cbf_err': ['mean', 'std'],
            'nn_att_err': ['mean', 'std'], 'ls_att_err': ['mean', 'std'],
            'nn_cbf_pred': ['mean', 'std'], 'ls_cbf_pred': ['mean', 'std'],
            'nn_att_pred': ['mean', 'std'], 'ls_att_pred': ['mean', 'std']
        }).reset_index()
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        group_col = f"{x_param}_mean"
    else:
        summary = df.groupby(x_param).agg({
            'nn_cbf_err': ['mean', 'std'], 'ls_cbf_err': ['mean', 'std'],
            'nn_att_err': ['mean', 'std'], 'ls_att_err': ['mean', 'std'],
            'nn_cbf_pred': ['mean', 'std'], 'ls_cbf_pred': ['mean', 'std'],
            'nn_att_pred': ['mean', 'std'], 'ls_att_pred': ['mean', 'std']
        }).reset_index()
        summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
        group_col = x_param

    summary['nn_cbf_cov'] = summary['nn_cbf_pred_std'] / summary['nn_cbf_pred_mean']
    summary['ls_cbf_cov'] = summary['ls_cbf_pred_std'] / summary['ls_cbf_pred_mean']
    summary['nn_att_cov'] = summary['nn_att_pred_std'] / summary['nn_att_pred_mean']
    summary['ls_att_cov'] = summary['ls_att_pred_std'] / summary['ls_att_pred_mean']
    
    metrics = {
        'Bias (Prediction - Ground Truth)': {'cbf': ('nn_cbf_err_mean', 'ls_cbf_err_mean'), 'att': ('nn_att_err_mean', 'ls_att_err_mean')},
        'Coefficient of Variation': {'cbf': ('nn_cbf_cov', 'ls_cbf_cov'), 'att': ('nn_att_cov', 'ls_att_cov')}
    }
    
    x_label = f"Ground Truth {x_param.split('_')[-1].upper()}"

    for metric_name, params in metrics.items():
        for param_unit, (nn_col, ls_col) in params.items():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(summary[group_col], summary[nn_col], 'o-', label='NN Model', color='crimson')
            ax.plot(summary[group_col], summary[ls_col], 'x--', label='LS Baseline', color='darkgray')
            
            title = f"{param_unit.upper()} {metric_name} vs. {x_label}\n({fixed_param_str})"
            ax.set_title(title, fontsize=14)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            
            if 'Bias' in metric_name:
                ax.axhline(0, color='black', linestyle=':', linewidth=1.0, label='Perfect Accuracy')

            ax.legend()
            
            plot_name = f"C_{scenario_name}_{param_unit}_{metric_name.split(' ')[0].lower()}.png"
            fig_path = output_dir / plot_name
            plt.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"  -> Saved report plot to: {fig_path}")

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="The Definitive Diagnostic Plan: Validating the Simulation & Evaluating the Model.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model_artifacts_dir", type=str, help="Path to the trained model artifacts directory.")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where all plots and CSVs will be saved.")
    args = parser.parse_args()

    model_dir = Path(args.model_artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80); print("      STARTING DEFINITIVE DIAGNOSTIC PIPELINE (FAIR COMPARISON)"); print("="*80)
    
    models, config, norm_stats, device = load_artifacts(model_dir)
    plds = np.array(config['pld_values'])
    asl_params = ASLParameters(**{k: v for k, v in config.items() if k in ASLParameters.__annotations__})
    simulator = ASLSimulator(params=asl_params)
    
    ls_params_dict = {k:v for k,v in config.items() if k in ['T1_artery','T_tau','T2_factor','alpha_BS1','alpha_PCASL','alpha_VSASL']}
    
    nn_args = {'models': models, 'config': config, 'norm_stats': norm_stats, 'device': device}
    ls_args = {'ls_params': ls_params_dict}

    phase1_validate_simulation(simulator, plds, output_dir)
    
    att_sweep = np.linspace(500, 4000, 20)
    cbf_sweep = np.linspace(20, 120, 20)
    total_sims_random_scenarios = 20 * NUM_SIMS_PER_DATAPOINT
    
    scenarios_to_run = [
        {
            'name': 'A_FixedCBF_VaryingATT', 'tsnr': 10, 'x_axis_param': 'true_att', 
            'fixed_param_str': 'Ground Truth CBF = 60', 'bin_data': False,
            'iterator': [(60.0, att) for att in att_sweep]
        },
        {
            'name': 'B_FixedATT_VaryingCBF', 'tsnr': 10, 'x_axis_param': 'true_cbf',
            'fixed_param_str': 'Ground Truth ATT = 1500ms', 'bin_data': False,
            'iterator': [(cbf, 1500.0) for cbf in cbf_sweep]
        },
        {
            'name': 'C_VaryingBoth_StandardNoise', 'tsnr': 10, 'x_axis_param': 'true_att',
            'fixed_param_str': 'tSNR = 10, CBF sampled from [20,100]', 'bin_data': True,
            'iterator': zip(np.random.uniform(20, 100, total_sims_random_scenarios), np.random.uniform(500, 4000, total_sims_random_scenarios))
        },
        {
            'name': 'D_VaryingBoth_HighNoise', 'tsnr': 3, 'x_axis_param': 'true_att',
            'fixed_param_str': 'tSNR = 3, CBF sampled from [20,100]', 'bin_data': True,
            'iterator': zip(np.random.uniform(20, 100, total_sims_random_scenarios), np.random.uniform(500, 4000, total_sims_random_scenarios))
        }
    ]

    all_csvs = {}
    for scenario in scenarios_to_run:
        num_sims_for_this_run = 1 if scenario.get('bin_data', False) else NUM_SIMS_PER_DATAPOINT

        csv_path = run_full_scenario(
            scenario, simulator, plds, nn_args, ls_args, output_dir,
            num_sims=num_sims_for_this_run
        )
        all_csvs[scenario['name']] = {'path': csv_path, 'params': scenario}

    for scenario_name, data in all_csvs.items():
        phase3_generate_report(data['path'], data['params'], output_dir)
        
    print("\n" + "="*80); print("      DIAGNOSTIC PIPELINE FINISHED SUCCESSFULLY"); print(f"      All results saved in: {output_dir.resolve()}"); print("="*80)

if __name__ == "__main__":
    main()