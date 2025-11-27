# FILE: diagnose_simulation_and_model.py

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
from scipy.optimize import least_squares

# --- Suppress RuntimeWarning for mean of empty slice ---
warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Import from project codebase ---
try:
    from asl_simulation import ASLSimulator, ASLParameters, _generate_pcasl_signal_jit, _generate_vsasl_signal_jit
    from enhanced_simulation import PhysiologicalVariation
    from enhanced_asl_network import DisentangledASLNet
    from utils import engineer_signal_features, process_signals_cpu
except ImportError as e:
    print(f"FATAL: Could not import necessary project files. Error: {e}")
    print("Please run this script from the root directory of the 'adikondepudi-asl-multiverse' project.")
    sys.exit(1)

# --- Configuration ---
NUM_SIMS_PER_DATAPOINT = 100 # Reduced slightly for speed, increase for final paper plots

# ==============================================================================
# 1. STANDALONE LEAST SQUARES SOLVER (Replaces missing multiverse_functions)
# ==============================================================================

def kinetic_model_residuals(params_to_fit, plds, observed_signal, fixed_params):
    """
    Residual function for Scipy Least Squares.
    params_to_fit: [cbf, att]
    """
    cbf_val = params_to_fit[0] # ml/g/s
    att_val = params_to_fit[1] # ms
    
    # Unpack fixed physics parameters
    t1_a = fixed_params['T1_artery']
    t_tau = fixed_params['T_tau']
    alpha_pcasl = fixed_params['alpha_PCASL']
    alpha_vsasl = fixed_params['alpha_VSASL']
    t2_factor = fixed_params.get('T2_factor', 1.0)
    t_sat_vs = fixed_params.get('T_sat_vs', 2000.0) # Default if not in config

    # Generate Model Signals using the JIT functions
    # Note: JIT functions expect specific argument order
    model_pcasl = _generate_pcasl_signal_jit(
        plds, att_val, cbf_val, t1_a, t_tau, alpha_pcasl, t2_factor
    )
    
    model_vsasl = _generate_vsasl_signal_jit(
        plds, att_val, cbf_val, t1_a, alpha_vsasl, t2_factor, t_sat_vs
    )
    
    # Concatenate to match observed vector [PCASL..., VSASL...]
    model_signal = np.concatenate([model_pcasl, model_vsasl])
    
    # Return residuals (Model - Observed)
    # Handling NaNs just in case
    res = model_signal - observed_signal
    return np.nan_to_num(res)

def predict_ls_single_voxel(noisy_signal: np.ndarray, plds: np.ndarray, ls_params: dict) -> tuple:
    """
    Runs a robust Non-Linear Least Squares (NLLS) fit using Scipy.
    """
    # 1. Prepare constraints and initial guess
    # CBF in ml/g/s (0 to 0.025 which is 150 ml/100g/min), ATT in ms (0 to 6000)
    lower_bounds = [0.0, 0.0]
    upper_bounds = [0.025, 6000.0] 
    
    # Simple initial guess: CBF=60ml/100g/min, ATT=1500ms
    x0 = [60.0/6000.0, 1500.0]

    try:
        # 2. Run Optimization
        result = least_squares(
            kinetic_model_residuals,
            x0,
            bounds=(lower_bounds, upper_bounds),
            args=(plds, noisy_signal, ls_params),
            method='trf',
            ftol=1e-3, xtol=1e-3, gtol=1e-3 # Looser tolerances for speed in diagnostics
        )
        
        if result.success:
            cbf_ml_100g_min = result.x[0] * 6000.0
            att_ms = result.x[1]
            return cbf_ml_100g_min, att_ms
        else:
            return np.nan, np.nan
    except Exception:
        return np.nan, np.nan

# ==============================================================================
# 2. V5 NN PREPROCESSING & HELPERS
# ==============================================================================



def denormalize_predictions(cbf_norm: float, att_norm: float, norm_stats: dict) -> tuple:
    """Denormalizes CBF and ATT predictions."""
    cbf = cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att = att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    return cbf, att

def load_artifacts(model_results_root: Path) -> tuple:
    """Robustly loads the model ensemble."""
    print(f"--> Loading artifacts from: {model_results_root}")
    try:
        with open(model_results_root / 'research_config.json', 'r') as f:
            config = json.load(f)
        with open(model_results_root / 'norm_stats.json', 'r') as f:
            norm_stats = json.load(f)

        models = []
        models_dir = model_results_root / 'trained_models'
        num_plds = len(config['pld_values'])
        
        # V5 Input size: Shape (2*PLDs) + Scalars (8)
        base_input_size = num_plds * 2 + 8

        # Calculate dynamic number of scalar features (stats + T1)
        num_scalar_features_dynamic = len(norm_stats['scalar_features_mean']) + 1

        for model_path in sorted(models_dir.glob('ensemble_model_*.pt')):
            model = DisentangledASLNet(mode='regression', input_size=base_input_size, num_scalar_features=num_scalar_features_dynamic, **config)
            model.to(dtype=torch.bfloat16)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            models.append(model)

        if not models: raise FileNotFoundError("No models found.")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for model in models: model.to(device)

        print(f"--> Loaded {len(models)} models.")
        return models, config, norm_stats, device
    except Exception as e:
        print(f"[FATAL ERROR] Artifact load failed: {e}")
        sys.exit(1)

def predict_nn_single_voxel(noisy_signal: np.ndarray, models: list, config: dict, norm_stats: dict, device: torch.device, t1_val: float) -> tuple:
    """Runs NN inference."""
    num_plds = len(config['pld_values'])
    
    eng_feats = engineer_signal_features(noisy_signal, num_plds)
    nn_input_unnorm = np.concatenate([noisy_signal, eng_feats]).reshape(1, -1)
    
    t1_values = np.array([[t1_val]], dtype=np.float32)
    norm_input = process_signals_cpu(nn_input_unnorm, norm_stats, num_plds, t1_values=t1_values)
    input_tensor = torch.from_numpy(norm_input).to(device, dtype=torch.bfloat16)

    with torch.no_grad():
        # NN returns (cbf, att, logvar_cbf, logvar_att, ...)
        cbf_outs = [m(input_tensor)[0].cpu().float().item() for m in models]
        att_outs = [m(input_tensor)[1].cpu().float().item() for m in models]
    
    # Ensemble averaging in normalized space
    cbf_pred, att_pred = denormalize_predictions(np.mean(cbf_outs), np.mean(att_outs), norm_stats)
    return cbf_pred, att_pred

# ==============================================================================
# 3. PHASES & REPORTING
# ==============================================================================

def run_full_scenario(scenario_params: dict, simulator: ASLSimulator, plds: np.ndarray, nn_args: dict, ls_args: dict, output_dir: Path, num_sims: int):
    """Runs the simulation loop."""
    scenario_name = scenario_params['name']
    print(f"\n--- Running Scenario: {scenario_name} ---")
    results = []
    physio_var = PhysiologicalVariation()
    base_params = simulator.params

    param_iterator = list(scenario_params['iterator'])
    
    for gt_cbf, gt_att in tqdm(param_iterator, desc="Simulating"):
        for _ in range(num_sims):
            # Perturb physics slightly for realism ("Inverse Crime" prevention)
            # But keep within reason so LS has a fighting chance
            true_t1 = np.random.uniform(*physio_var.t1_artery_range)
            p_tau = base_params.T_tau # * np.random.uniform(0.95, 1.05) # Minor perturbation
            
            # Generate Data
            data_dict = simulator.generate_synthetic_data(
                plds, att_values=np.array([gt_att]), n_noise=1, tsnr=scenario_params['tsnr'],
                cbf_val=gt_cbf, t1_artery_val=true_t1, t_tau_val=p_tau
            )
            
            # Extract signal (batch 0, att_idx 0) -> Shape (plds*2,)
            signals = data_dict['MULTIVERSE'][0, 0, :, :]
            noisy_signal = np.concatenate([signals[:, 0], signals[:, 1]])
            
            # Predictions
            nn_cbf, nn_att = predict_nn_single_voxel(noisy_signal, t1_val=true_t1, **nn_args)
            ls_cbf, ls_att = predict_ls_single_voxel(noisy_signal, plds, **ls_args)

            results.append({
                'true_cbf': gt_cbf, 'true_att': gt_att,
                'nn_cbf_pred': nn_cbf, 'nn_att_pred': nn_att,
                'ls_cbf_pred': ls_cbf, 'ls_att_pred': ls_att,
            })

    df = pd.DataFrame(results)
    csv_path = output_dir / f"Results_{scenario_name}.csv"
    df.to_csv(csv_path, index=False)
    return csv_path

def generate_report(csv_path: Path, scenario_params: dict, output_dir: Path):
    """Generates comparison plots."""
    df = pd.read_csv(csv_path)
    name = scenario_params['name']
    x_param = scenario_params['x_axis_param']
    
    # Calculate Absolute Errors
    df['nn_cbf_abs_err'] = np.abs(df['nn_cbf_pred'] - df['true_cbf'])
    df['ls_cbf_abs_err'] = np.abs(df['ls_cbf_pred'] - df['true_cbf'])
    df['nn_att_abs_err'] = np.abs(df['nn_att_pred'] - df['true_att'])
    df['ls_att_abs_err'] = np.abs(df['ls_att_pred'] - df['true_att'])

    # Remove LS outliers (failed fits) for cleaner plots
    df_clean = df[df['ls_att_pred'] < 8000].copy()

    # Binning
    bins = np.linspace(df[x_param].min(), df[x_param].max(), 15)
    df_clean['bin'] = pd.cut(df_clean[x_param], bins)
    
    summary = df_clean.groupby('bin', observed=True).agg({
        x_param: 'mean',
        'nn_cbf_abs_err': 'mean', 'ls_cbf_abs_err': 'mean',
        'nn_att_abs_err': 'mean', 'ls_att_abs_err': 'mean'
    }).reset_index()

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # CBF Error Plot
    ax = axes[0]
    ax.plot(summary[x_param], summary['nn_cbf_abs_err'], 'o-', color='crimson', label='Neural Net', linewidth=2)
    ax.plot(summary[x_param], summary['ls_cbf_abs_err'], 's--', color='gray', label='Least Squares', linewidth=2)
    ax.set_xlabel(f'Ground Truth {x_param.upper()}')
    ax.set_ylabel('Mean Absolute Error (CBF)')
    ax.set_title(f'CBF Accuracy: {name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ATT Error Plot
    ax = axes[1]
    ax.plot(summary[x_param], summary['nn_att_abs_err'], 'o-', color='crimson', label='Neural Net', linewidth=2)
    ax.plot(summary[x_param], summary['ls_att_abs_err'], 's--', color='gray', label='Least Squares', linewidth=2)
    ax.set_xlabel(f'Ground Truth {x_param.upper()}')
    ax.set_ylabel('Mean Absolute Error (ATT)')
    ax.set_title(f'ATT Accuracy: {name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"Plot_{name}.png", dpi=150)
    plt.close()
    print(f"  -> Plot saved: {output_dir / f'Plot_{name}.png'}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_artifacts_dir", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()

    model_dir = Path(args.model_artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load System
    models, config, norm_stats, device = load_artifacts(model_dir)
    plds = np.array(config['pld_values'])
    
    # Simulator with base parameters
    sim_params = ASLParameters(**{k: v for k, v in config.items() if k in ASLParameters.__annotations__})
    simulator = ASLSimulator(params=sim_params)
    
    # Params for LS solver (need to extract relevant physics consts)
    ls_params_dict = {
        'T1_artery': sim_params.T1_artery, 'T_tau': sim_params.T_tau,
        'alpha_PCASL': sim_params.alpha_PCASL, 'alpha_VSASL': sim_params.alpha_VSASL,
        'T2_factor': sim_params.T2_factor, 'T_sat_vs': sim_params.T_sat_vs
    }
    
    nn_args = {'models': models, 'config': config, 'norm_stats': norm_stats, 'device': device}
    ls_args = {'ls_params': ls_params_dict}

    # 2. Define Scenarios
    scenarios = [
        {
            'name': 'A_VaryingATT_FixedCBF',
            'tsnr': 10.0, 'x_axis_param': 'true_att',
            'iterator': [(60.0, att) for att in np.linspace(500, 4000, 20)]
        },
        {
            'name': 'B_VaryingCBF_FixedATT',
            'tsnr': 10.0, 'x_axis_param': 'true_cbf',
            'iterator': [(cbf, 1500.0) for cbf in np.linspace(10, 100, 20)]
        },
        {
             # More challenging noise
            'name': 'C_HighNoise_VaryingATT',
            'tsnr': 5.0, 'x_axis_param': 'true_att',
            'iterator': [(60.0, att) for att in np.linspace(500, 4000, 20)]
        }
    ]

    # 3. Run Loop
    print("--- Starting Diagnostics ---")
    for sc in scenarios:
        csv = run_full_scenario(sc, simulator, plds, nn_args, ls_args, output_dir, NUM_SIMS_PER_DATAPOINT)
        generate_report(csv, sc, output_dir)

    print("\nDone! Results in:", output_dir)

if __name__ == "__main__":
    main()