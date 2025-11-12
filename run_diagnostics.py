# FILE: run_diagnostics.py
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
import os

# --- Suppress minor warnings for cleaner output ---
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")

# --- Import from project codebase ---
try:
    from asl_simulation import ASLParameters
    from enhanced_simulation import RealisticASLSimulator, PhysiologicalVariation
    from enhanced_asl_network import DisentangledASLNet, CustomLoss
    from asl_trainer import ASLInMemoryDataset
except ImportError as e:
    print(f"FATAL: Could not import necessary project files. Error: {e}")
    print("Please run this script from the root directory of the 'adikondepudi-asl-multiverse' project.")
    sys.exit(1)

# --- Global Configuration ---
SAMPLES_FOR_DIST_ANALYSIS = 1_000_000
SAMPLES_FOR_AMBIGUITY_ANALYSIS = 1000
GRID_RESOLUTION = 30  # Controls density of CBF/ATT grid for heatmaps

# ==============================================================================
# V6-SPECIFIC HELPER FUNCTIONS
# ==============================================================================

def denormalize_predictions(cbf_norm: float, att_norm: float, norm_stats: dict) -> tuple:
    """Denormalizes CBF and ATT predictions using stats from training."""
    cbf = cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    att = att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    return cbf, att

def load_artifacts(model_artifacts_root: Path) -> tuple:
    """Robustly loads the model ensemble, config, and norm stats for V6 architecture."""
    print(f"--> Loading artifacts from: {model_artifacts_root}")
    try:
        with open(model_artifacts_root / 'research_config.json', 'r') as f:
            config = json.load(f)
        with open(model_artifacts_root / 'norm_stats.json', 'r') as f:
            norm_stats = json.load(f)

        models = []
        models_dir = model_artifacts_root / 'trained_models'
        if not models_dir.exists():
            raise FileNotFoundError(f"Directory not found: {models_dir}")
        
        # Determine input size from norm_stats for robustness
        num_plds = len(config['pld_values'])
        num_scalar_features = len(norm_stats['scalar_features_mean'])
        base_input_size = num_plds * 2 + num_scalar_features

        for model_path in sorted(list(models_dir.glob('ensemble_model_*.pt'))):
            model = DisentangledASLNet(mode='regression', input_size=base_input_size, **config)
            model.to(dtype=torch.bfloat16)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            models.append(model)

        if not models:
            raise FileNotFoundError("No models found in 'trained_models' folder.")
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for model in models: model.to(device)

        print(f"--> Successfully loaded {len(models)} models, config, and norm_stats to device '{device}'.")
        return models, config, norm_stats, device
    except Exception as e:
        print(f"[FATAL ERROR] Could not load artifacts: {e}. Check paths and file integrity. Exiting.")
        sys.exit(1)

# ==============================================================================
# PHASE 1: DATA DISTRIBUTION & INTRINSIC AMBIGUITY
# ==============================================================================

def diagnostic_1_1_data_distribution(offline_dataset_path: str, output_dir: Path):
    """Analyzes and plots the ground truth parameter distribution from the offline dataset."""
    print("\n--- DIAGNOSTIC 1.1: Analyzing Training Data Distribution ---")
    dataset_path = Path(offline_dataset_path)
    if not dataset_path.exists():
        print(f"  [WARNING] Offline dataset path not found: {dataset_path}. Skipping Diagnostic 1.1.")
        return

    chunk_files = sorted(list(dataset_path.glob('dataset_chunk_*.npz')))
    if not chunk_files:
        print(f"  [WARNING] No dataset chunks found in {dataset_path}. Skipping Diagnostic 1.1.")
        return

    all_params = []
    num_loaded_samples = 0
    
    pbar = tqdm(chunk_files, desc="Loading dataset parameters")
    for chunk_file in pbar:
        try:
            data = np.load(chunk_file)
            params = data['params']
            all_params.append(params)
            num_loaded_samples += len(params)
            if num_loaded_samples >= SAMPLES_FOR_DIST_ANALYSIS:
                break
        except Exception as e:
            print(f"  [WARNING] Could not load {chunk_file}: {e}")
            continue

    if not all_params:
        print("  [ERROR] Failed to load any parameter data. Aborting Diagnostic 1.1.")
        return
        
    params_arr = np.concatenate(all_params, axis=0)
    df = pd.DataFrame(params_arr, columns=['cbf', 'att'])
    
    csv_path = output_dir / "1_1_data_distribution.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> Saved {len(df)} ground truth samples to {csv_path}")

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 8))
    hb = ax.hexbin(df['att'], df['cbf'], gridsize=50, cmap='viridis', mincnt=1)
    fig.colorbar(hb, ax=ax, label='Sample Count per Bin')
    ax.set_title('Training Data Distribution of Ground Truth Parameters', fontsize=16)
    ax.set_xlabel('Arterial Transit Time (ATT) [ms]', fontsize=12)
    ax.set_ylabel('Cerebral Blood Flow (CBF) [ml/100g/min]', fontsize=12)
    
    fig_path = output_dir / "1_1_data_distribution.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved distribution plot to: {fig_path}")
    print("✅ Diagnostic 1.1 Complete.")

def diagnostic_1_2_signal_ambiguity(simulator: RealisticASLSimulator, plds_np: np.ndarray, output_dir: Path):
    """Quantifies signal ambiguity around physical model transition points."""
    print("\n--- DIAGNOSTIC 1.2: Quantifying Signal Ambiguity ---")
    
    # Define key transition points from the simulation model
    T_TAU = simulator.params.T_tau
    T_SAT_VS = simulator.params.T_sat_vs

    scenarios = {
        "around_T_tau": (T_TAU - 50, T_TAU + 50),
        "around_T_sat_vs": (T_SAT_VS - 50, T_SAT_VS + 50)
    }

    results = []
    for name, (att_A, att_B) in scenarios.items():
        print(f"  -> Simulating ambiguity between ATT={att_A}ms and ATT={att_B}ms...")
        
        # Generate clean templates
        clean_A_pcasl = simulator._generate_pcasl_signal(plds_np, att_A, 60.0, simulator.params.T1_artery, simulator.params.T_tau, simulator.params.alpha_PCASL)
        clean_A_vsasl = simulator._generate_vsasl_signal(plds_np, att_A, 60.0, simulator.params.T1_artery, simulator.params.alpha_VSASL)
        template_A = np.concatenate([clean_A_pcasl, clean_A_vsasl])

        clean_B_pcasl = simulator._generate_pcasl_signal(plds_np, att_B, 60.0, simulator.params.T1_artery, simulator.params.T_tau, simulator.params.alpha_PCASL)
        clean_B_vsasl = simulator._generate_vsasl_signal(plds_np, att_B, 60.0, simulator.params.T1_artery, simulator.params.alpha_VSASL)
        template_B = np.concatenate([clean_B_pcasl, clean_B_vsasl])
        
        # Generate noisy signals for both ground truths
        noisy_signals_A = simulator.generate_synthetic_data(plds_np, np.array([att_A]), n_noise=SAMPLES_FOR_AMBIGUITY_ANALYSIS, tsnr=5.0, cbf_val=60.0)['MULTIVERSE']
        noisy_signals_B = simulator.generate_synthetic_data(plds_np, np.array([att_B]), n_noise=SAMPLES_FOR_AMBIGUITY_ANALYSIS, tsnr=5.0, cbf_val=60.0)['MULTIVERSE']

        for i in range(SAMPLES_FOR_AMBIGUITY_ANALYSIS):
            noisy_A = np.concatenate([noisy_signals_A[i, 0, :, 0], noisy_signals_A[i, 0, :, 1]])
            noisy_B = np.concatenate([noisy_signals_B[i, 0, :, 0], noisy_signals_B[i, 0, :, 1]])
            
            # For a signal generated at A, how far is it from template A and B?
            results.append({'scenario': name, 'gt_att': att_A, 'mse_vs_A': np.mean((noisy_A - template_A)**2), 'mse_vs_B': np.mean((noisy_A - template_B)**2)})
            # For a signal generated at B, how far is it from template A and B?
            results.append({'scenario': name, 'gt_att': att_B, 'mse_vs_A': np.mean((noisy_B - template_A)**2), 'mse_vs_B': np.mean((noisy_B - template_B)**2)})
    
    df = pd.DataFrame(results)
    csv_path = output_dir / "1_2_signal_ambiguity_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"  -> Saved MSE results to: {csv_path}")

    # Plotting
    for name, (att_A, att_B) in scenarios.items():
        sub_df = df[df['scenario'] == name]
        fig, ax = plt.subplots(figsize=(12, 6))
        
        sns.kdeplot(data=sub_df[sub_df['gt_att'] == att_A], x='mse_vs_A', fill=True, alpha=0.5, label=f'Noisy signals from {att_A}ms vs. clean {att_A}ms', ax=ax)
        sns.kdeplot(data=sub_df[sub_df['gt_att'] == att_A], x='mse_vs_B', fill=True, alpha=0.5, label=f'Noisy signals from {att_A}ms vs. clean {att_B}ms', ax=ax)
        
        ax.set_title(f'Signal Ambiguity for ATT near {name.replace("_", " ")}\n(Noisy signals generated from ATT={att_A}ms)', fontsize=14)
        ax.set_xlabel('Mean Squared Error (MSE) to Clean Templates', fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.legend()
        fig_path = output_dir / f"1_2_signal_ambiguity_{name}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> Saved ambiguity plot for {name} to: {fig_path}")
        
    print("✅ Diagnostic 1.2 Complete.")

# ==============================================================================
# PHASE 2 & 3: MODEL BEHAVIOR & LOSS LANDSCAPE
# ==============================================================================
# These two are combined as they require iterating over the same grid of data.

def diagnostic_2_and_3_model_and_loss(models, config, norm_stats, device, simulator, output_dir):
    """Probes MoE behavior and visualizes the loss landscape over a parameter grid."""
    print("\n--- DIAGNOSTIC 2 & 3: Probing MoE Specialization & Loss Landscape ---")

    # --- Setup ---
    plds_np = np.array(config['pld_values'])
    physio_var = PhysiologicalVariation()
    att_grid = np.linspace(physio_var.att_range[0], physio_var.att_range[1], GRID_RESOLUTION)
    cbf_grid = np.linspace(physio_var.cbf_range[0], physio_var.cbf_range[1], GRID_RESOLUTION)
    
    # We only need the first model of the ensemble for this analysis
    model = models[0]
    
    # Prepare loss function for Stage 2
    loss_fn = CustomLoss(training_stage=2, w_cbf=1.0, w_att=1.0)
    
    # --- Data structures for capturing results ---
    gating_weights = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION, config['moe']['num_experts']))
    per_sample_loss = np.zeros((GRID_RESOLUTION, GRID_RESOLUTION))
    
    # --- Setup forward hook to capture gating weights ---
    captured_weights = []
    def hook_fn(module, input, output):
        captured_weights.append(output.detach().cpu())
    
    # MoE head is inside the 'head' attribute of the main model
    gating_network = model.head.gating_network
    handle = gating_network.register_forward_hook(hook_fn)

    # --- Iterate over the parameter grid ---
    pbar = tqdm(total=GRID_RESOLUTION * GRID_RESOLUTION, desc="Analyzing model over parameter grid")
    for i, cbf in enumerate(cbf_grid):
        for j, att in enumerate(att_grid):
            # Generate one noisy sample for this grid point
            data_dict = simulator.generate_synthetic_data(plds_np, np.array([att]), n_noise=1, tsnr=10.0, cbf_val=cbf)
            
            # --- Prepare input for model ---
            # This follows the exact preprocessing logic from ASLInMemoryDataset
            val_dataset = ASLInMemoryDataset(data_dir=None, norm_stats=norm_stats, stage=2)
            raw_curves = np.concatenate([data_dict['MULTIVERSE'][0, 0, :, 0], data_dict['MULTIVERSE'][0, 0, :, 1]])
            eng_features = val_dataset.engineer_signal_features(raw_curves.reshape(1,-1), len(plds_np))
            input_for_processing = np.concatenate([raw_curves, eng_features.flatten()]).reshape(1,-1)
            processed_input = val_dataset._process_signals(input_for_processing)
            input_tensor = torch.from_numpy(processed_input.astype(np.float32)).to(device, dtype=torch.bfloat16)

            # --- Prepare target for loss function ---
            target_unnorm = np.array([[cbf, att]])
            target_norm = val_dataset._normalize_params(target_unnorm)
            target_tensor = torch.from_numpy(target_norm.astype(np.float32)).to(device)

            # --- Run inference ---
            captured_weights.clear()
            with torch.no_grad():
                model_outputs = model(input_tensor)
                _, loss_comps = loss_fn(model_outputs, target_tensor, global_epoch=100)
            
            # --- Store results ---
            gating_weights[i, j, :] = captured_weights[0].numpy().flatten()
            per_sample_loss[i, j] = loss_comps['unreduced_loss'].mean().item()
            pbar.update(1)
    
    pbar.close()
    handle.remove() # IMPORTANT: Clean up the hook

    # --- Save raw data to CSV ---
    grid_data = []
    for i in range(GRID_RESOLUTION):
        for j in range(GRID_RESOLUTION):
            row = {'cbf_gt': cbf_grid[i], 'att_gt': att_grid[j], 'loss': per_sample_loss[i, j]}
            for k in range(config['moe']['num_experts']):
                row[f'expert_{k}_weight'] = gating_weights[i, j, k]
            grid_data.append(row)
    df_grid = pd.DataFrame(grid_data)
    csv_path = output_dir / "2_3_grid_analysis_results.csv"
    df_grid.to_csv(csv_path, index=False)
    print(f"  -> Saved grid analysis results to: {csv_path}")

    # --- Plotting ---
    # Diagnostic 2.1: MoE Gating Weights
    num_experts = config['moe']['num_experts']
    fig, axes = plt.subplots(1, num_experts, figsize=(5 * num_experts, 5), sharey=True)
    fig.suptitle('MoE Gating Network Specialization', fontsize=16)
    for k in range(num_experts):
        ax = axes[k]
        im = ax.imshow(gating_weights[:, :, k].T, origin='lower', aspect='auto', cmap='viridis',
                       extent=[cbf_grid[0], cbf_grid[-1], att_grid[0], att_grid[-1]])
        ax.set_title(f'Expert {k} Weight')
        ax.set_xlabel('Ground Truth CBF')
        if k == 0: ax.set_ylabel('Ground Truth ATT')
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
    fig_path = output_dir / "2_1_moe_specialization.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved MoE specialization plot to: {fig_path}")

    # Diagnostic 3.1: Loss Landscape
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(per_sample_loss.T, origin='lower', aspect='auto', cmap='magma',
                   extent=[cbf_grid[0], cbf_grid[-1], att_grid[0], att_grid[-1]])
    ax.axhline(config['T_tau'], color='cyan', linestyle='--', label=f'T_tau ({config["T_tau"]} ms)')
    ax.axhline(simulator.params.T_sat_vs, color='lime', linestyle='--', label=f'T_sat_vs ({simulator.params.T_sat_vs} ms)')
    ax.set_title('Per-Sample Loss Landscape', fontsize=16)
    ax.set_xlabel('Ground Truth CBF')
    ax.set_ylabel('Ground Truth ATT')
    ax.legend()
    fig.colorbar(im, ax=ax, label='NLL Loss')
    fig_path = output_dir / "3_1_loss_landscape.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> Saved loss landscape plot to: {fig_path}")
    
    print("✅ Diagnostics 2 & 3 Complete.")

# ==============================================================================
# MAIN ORCHESTRATOR
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Diagnostic Script for the ASL Multiverse Project.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("model_artifacts_dir", type=str, help="Path to the trained model artifacts directory (e.g., a stage 2 output folder).")
    parser.add_argument("output_dir", type=str, help="Path to the output directory where all plots and CSVs will be saved.")
    args = parser.parse_args()

    model_dir = Path(args.model_artifacts_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80); print("      STARTING COMPREHENSIVE DIAGNOSTIC PIPELINE"); print("="*80)
    
    # --- Load all necessary components ---
    models, config, norm_stats, device = load_artifacts(model_dir)
    plds_np = np.array(config['pld_values'])
    asl_params_sim = ASLParameters(**{k:v for k,v in config.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params_sim)

    # --- Run Diagnostics Sequentially ---
    diagnostic_1_1_data_distribution(config.get('offline_dataset_path'), output_dir)
    diagnostic_1_2_signal_ambiguity(simulator, plds_np, output_dir)
    diagnostic_2_and_3_model_and_loss(models, config, norm_stats, device, simulator, output_dir)
        
    print("\n" + "="*80)
    print("      DIAGNOSTIC PIPELINE FINISHED SUCCESSFULLY")
    print(f"      All results saved in: {output_dir.resolve()}")
    print("="*80)

if __name__ == "__main__":
    main()