# FILE: diagnose_simulation_and_model.py
# Corrected version, removes global variable bug.

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
# Ensure this script is run from the root of the `adikondepudi-asl-multiverse` directory
try:
    from asl_simulation import ASLSimulator, ASLParameters
    from enhanced_asl_network import DisentangledASLNet
    from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
    from utils import get_grid_search_initial_guess, engineer_signal_features
    # Re-using the robust normalization function from the prediction script
    from predict_on_invivo import apply_normalization_disentangled, denormalize_predictions
except ImportError as e:
    print(f"FATAL: Could not import necessary project files. Error: {e}")
    print("Please run this script from the root directory of the 'adikondepudi-asl-multiverse' project.")
    sys.exit(1)

# --- Configuration ---
NUM_SIMS_PER_DATAPOINT = 500 # Number of noise realizations per ground truth point for fixed-param scenarios

# ==============================================================================
# HELPER FUNCTIONS (LOADING & PREDICTION)
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
        
        # This diagnostic is specifically for the DisentangledASLNet
        model_class = DisentangledASLNet
        base_input_size = num_plds * 2 + 4 + 1 # shape + eng + amp

        for model_path in models_dir.glob('ensemble_model_*.pt'):
            model = model_class(input_size=base_input_size, **config)
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
    """Runs a single voxel through the NN pipeline."""
    num_plds = len(config['pld_values'])
    eng_feats = engineer_signal_features(noisy_signal, num_plds)
    nn_input_unnorm = np.concatenate([noisy_signal, eng_feats]).reshape(1, -1)
    
    norm_input = apply_normalization_disentangled(nn_input_unnorm, norm_stats, num_plds)
    input_tensor = torch.from_numpy(norm_input).to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        cbf_means = [model(input_tensor)[0].cpu().float().numpy() for model in models]
        att_means = [model(input_tensor)[1].cpu().float().numpy() for model in models]
    
    nn_cbf_pred_norm = np.mean([item.item() for item in cbf_means])
    nn_att_pred_norm = np.mean([item.item() for item in att_means])

    nn_cbf_pred, nn_att_pred, _, _ = denormalize_predictions(nn_cbf_pred_norm, nn_att_pred_norm, None, None, norm_stats)
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
# PHASE 2: Systematic Evaluation on the Trusted Ground Truth
# ==============================================================================

def run_full_scenario(scenario_params: dict, simulator: ASLSimulator, plds: np.ndarray, nn_args: dict, ls_args: dict, output_dir: Path, num_sims: int):
    """Manages the simulation loop for a given scenario and saves results."""
    scenario_name = scenario_params['name']
    print(f"\n--- PHASE 2: Running Diagnostic Scenario: {scenario_name} ---")
    
    # Use explicit comment as requested by the diagnostic plan
    print("-> Using simple Gaussian noise via ASLSimulator.generate_synthetic_data to ensure a clear, controlled comparison.")

    results = []
    
    # The main loop iterates over the ground truth values for this scenario
    param_iterator = scenario_params['iterator']
    for gt_cbf, gt_att in tqdm(param_iterator, desc=f"Simulating {scenario_name}"):
        
        # For each ground truth point, run N simulations with different noise
        data_dict = simulator.generate_synthetic_data(
            plds,
            att_values=np.array([gt_att]),
            n_noise=num_sims, # USE THE PASSED ARGUMENT
            tsnr=scenario_params['tsnr'],
            cbf_val=gt_cbf
        )
        
        # `data_dict['MULTIVERSE']` has shape (n_noise, n_att, n_plds, 2)
        # We have n_att=1, so we squeeze it.
        signals = data_dict['MULTIVERSE'][:, 0, :, :] # Shape: (n_noise, n_plds, 2)
        
        for i in range(num_sims): # USE THE PASSED ARGUMENT
            # Concatenate PCASL and VSASL parts for one noise realization
            pcasl_noisy = signals[i, :, 0]
            vsasl_noisy = signals[i, :, 1]
            noisy_signal = np.concatenate([pcasl_noisy, vsasl_noisy])
            
            # Get predictions from both models
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

def phase3_generate_report(csv_path: Path, scenario_params: dict, output_dir: Path):
    """Analyzes a results CSV and generates the final report plots."""
    scenario_name = scenario_params['name']
    print(f"\n--- PHASE 3: Generating Report for Scenario: {scenario_name} ---")
    df = pd.read_csv(csv_path)

    # Determine the varying parameter for the x-axis and grouping
    x_param = scenario_params.get('x_axis_param', 'true_att')
    fixed_param_str = scenario_params.get('fixed_param_str', '')

    # Calculate errors
    df['nn_cbf_err'] = df['nn_cbf_pred'] - df['true_cbf']
    df['ls_cbf_err'] = df['ls_cbf_pred'] - df['true_cbf']
    df['nn_att_err'] = df['nn_att_pred'] - df['true_att']
    df['ls_att_err'] = df['ls_att_pred'] - df['true_att']

    # Bin data if the x-axis parameter is continuous from random sampling
    if scenario_params.get('bin_data', False):
        num_bins = 15
        df['bin'] = pd.cut(df[x_param], bins=num_bins)
        # Use the midpoint of the bin for plotting
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

    # Calculate CoV
    summary['nn_cbf_cov'] = summary['nn_cbf_pred_std'] / summary['nn_cbf_pred_mean']
    summary['ls_cbf_cov'] = summary['ls_cbf_pred_std'] / summary['ls_cbf_pred_mean']
    summary['nn_att_cov'] = summary['nn_att_pred_std'] / summary['nn_att_pred_mean']
    summary['ls_att_cov'] = summary['ls_att_pred_std'] / summary['ls_att_pred_mean']
    
    # Plotting
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
    parser.add_argument("model_artifacts_dir", type=str, help="Path to the trained model artifacts directory (e.g., a fine-tuning run folder).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where all plots and CSVs will be saved.")
    args = parser.parse_args()

    model_dir = Path(args.model_artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("      STARTING DEFINITIVE DIAGNOSTIC PIPELINE")
    print("="*80)
    
    # --- Load all necessary artifacts once ---
    models, config, norm_stats, device = load_artifacts(model_dir)
    plds = np.array(config['pld_values'])
    asl_params = ASLParameters(**{k: v for k, v in config.items() if k in ASLParameters.__annotations__})
    simulator = ASLSimulator(params=asl_params)
    ls_params_dict = {k:v for k,v in config.items() if k in ['T1_artery','T_tau','T2_factor','alpha_BS1','alpha_PCASL','alpha_VSASL']}
    
    nn_args = {'models': models, 'config': config, 'norm_stats': norm_stats, 'device': device}
    ls_args = {'ls_params': ls_params_dict}

    # --- Phase 1 ---
    phase1_validate_simulation(simulator, plds, output_dir)
    
    # --- Phase 2 & 3 ---
    # Define the iterators for ground truth values for each scenario
    att_sweep = np.linspace(500, 4000, 20)
    cbf_sweep = np.linspace(20, 120, 20)
    
    # For scenarios C and D, we generate a large pool of random samples for robust binning later.
    total_sims_random_scenarios = 20 * NUM_SIMS_PER_DATAPOINT # ~10k total simulations for smooth plots
    
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
        # Determine the number of simulations for this specific scenario.
        # Scenarios C and D have unique ground truth for each run, so n_sims is 1.
        # Scenarios A and B have fixed ground truth points, so we run many noise realizations.
        num_sims_for_this_run = 1 if scenario.get('bin_data', False) else NUM_SIMS_PER_DATAPOINT

        csv_path = run_full_scenario(
            scenario, simulator, plds, nn_args, ls_args, output_dir,
            num_sims=num_sims_for_this_run
        )
        all_csvs[scenario['name']] = {'path': csv_path, 'params': scenario}

    for scenario_name, data in all_csvs.items():
        phase3_generate_report(data['path'], data['params'], output_dir)
        
    print("\n="*80)
    print("      DIAGNOSTIC PIPELINE FINISHED SUCCESSFULLY")
    print(f"      All results saved in: {output_dir.resolve()}")
    print("="*80)

if __name__ == "__main__":
    main()