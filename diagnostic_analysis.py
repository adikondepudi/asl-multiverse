# diagnostic_analysis.py

import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import inspect

# --- Configuration: SET THIS PATH ---
RESULTS_DIR = "./comprehensive_results/asl_research_20250616_175302/"  # <-- Edit this line!
# --- End Configuration ---

# Import your project's custom modules
try:
    from enhanced_asl_network import EnhancedASLNet
    from enhanced_simulation import RealisticASLSimulator, ASLParameters
    from utils import engineer_signal_features
    from pcasl_functions import fun_PCASL_1comp_vect_pep
    from vsasl_functions import fun_VSASL_1comp_vect_pep
    from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
    from comparison_framework import apply_normalization_to_input_flat, denormalize_predictions
except ImportError as e:
    print(f"Error: Could not import necessary project modules. Make sure you are running this script from the project's root directory. Details: {e}")
    exit()

# --- Utility Functions ---

def load_analysis_artifacts(results_dir: str):
    """Loads the model, config, and norm_stats for analysis, handling Optuna-tuned architectures."""
    print("--- Loading Artifacts ---")

    results_dir_path = Path(results_dir)
    if not results_dir_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # --- FIX: Robust config loading for Optuna compatibility ---
    # 1. Load the initial base config file
    config_path = results_dir_path / 'research_config.json'
    if not config_path.exists(): raise FileNotFoundError(f"Initial config file not found: {config_path}")
    with open(config_path, 'r') as f: config = json.load(f)
    print(f"Loaded base config from: {config_path}")

    # 2. Check for final results and update config with Optuna's best params if they exist
    final_results_path = results_dir_path / 'final_research_results.json'
    if final_results_path.exists():
        with open(final_results_path, 'r') as f: final_results = json.load(f)
        if 'optuna_best_params' in final_results and final_results['optuna_best_params']:
            print("Found Optuna results. Updating config with optimized architecture parameters.")
            best_params = final_results['optuna_best_params']
            config['hidden_sizes'] = [
                best_params.get('hidden_size_1'), best_params.get('hidden_size_2'), best_params.get('hidden_size_3')
            ]
            config['dropout_rate'] = best_params.get('dropout_rate')
            print(f"Using optimized hidden_sizes: {config['hidden_sizes']}")

    # 3. Load normalization stats
    norm_stats_path = results_dir_path / 'norm_stats.json'
    if not norm_stats_path.exists(): raise FileNotFoundError(f"Norm stats not found: {norm_stats_path}")
    with open(norm_stats_path, 'r') as f: norm_stats = json.load(f)
    print(f"Loaded norm_stats from: {norm_stats_path}")

    # 4. Load the first model from the ensemble
    model_path = results_dir_path / 'trained_models' / 'ensemble_model_0.pt'
    if not model_path.exists(): raise FileNotFoundError(f"Model file not found: {model_path}")

    # Re-create the model with the correct architecture from the (potentially updated) config
    plds_np = np.array(config.get('pld_values', range(500, 3001, 500)))
    num_plds = len(plds_np)
    base_input_size_nn = num_plds * 2 + 4

    model = EnhancedASLNet(input_size=base_input_size_nn, **config)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"Successfully loaded model from: {model_path}")
    
    return model, config, norm_stats

# --- Analysis Functions ---

def analyze_healthy_adult_failure(model, config, norm_stats):
    """Generates data for a specific case to showcase NN advantage and plots it."""
    print("\n--- Running Part 1: Showcasing NN Advantage in a Challenging Case ---")
    
    true_cbf, true_att, snr = 54.286, 2444.444, 5.0
    print(f"Analyzing specific case: True CBF = {true_cbf:.1f}, True ATT = {true_att:.1f}, SNR = {snr}")
    
    asl_params_dict = {k: v for k, v in config.items() if k in ASLParameters.__annotations__}
    simulator = RealisticASLSimulator(params=ASLParameters(**asl_params_dict))
    plds = np.array(config['pld_values'])
    num_plds = len(plds)
    
    np.random.seed(101) 
    data_dict = simulator.generate_synthetic_data(plds, np.array([true_att]), n_noise=1, tsnr=snr, cbf_val=true_cbf)
    noisy_pcasl = data_dict['MULTIVERSE'][0, 0, :, 0]
    noisy_vsasl = data_dict['MULTIVERSE'][0, 0, :, 1]
    raw_signal_vector = np.concatenate([noisy_pcasl, noisy_vsasl])

    # --- FIX: Implement full, correct NN input pipeline: features -> normalization ---
    engineered_feats = engineer_signal_features(raw_signal_vector.reshape(1, -1), num_plds)
    nn_input_unnormalized = np.concatenate([raw_signal_vector, engineered_feats.flatten()])
    nn_input_normalized = apply_normalization_to_input_flat(nn_input_unnormalized, norm_stats, num_plds, has_m0=False)
    
    input_tensor = torch.FloatTensor(nn_input_normalized).unsqueeze(0)
    with torch.no_grad():
        cbf_pred_norm, att_pred_norm, _, _, _, _ = model(input_tensor)
    
    pred_cbf, pred_att, _, _ = denormalize_predictions(
        cbf_pred_norm.numpy(), att_pred_norm.numpy(), None, None, norm_stats
    )
    pred_cbf, pred_att = pred_cbf.item(), pred_att.item()
    
    # Generate signals for plotting
    pcasl_kwargs = {k: v for k, v in config.items() if k in ['T1_artery', 'T_tau', 'T2_factor', 'alpha_BS1', 'alpha_PCASL']}
    vsasl_kwargs = {k: v for k, v in config.items() if k in ['T1_artery', 'T2_factor', 'alpha_BS1', 'alpha_VSASL']}
    clean_pcasl = fun_PCASL_1comp_vect_pep([true_cbf / 6000.0, true_att], plds, **pcasl_kwargs)
    clean_vsasl = fun_VSASL_1comp_vect_pep([true_cbf / 6000.0, true_att], plds, **vsasl_kwargs)
    pred_pcasl = fun_PCASL_1comp_vect_pep([pred_cbf / 6000.0, pred_att], plds, **pcasl_kwargs)
    pred_vsasl = fun_VSASL_1comp_vect_pep([pred_cbf / 6000.0, pred_att], plds, **vsasl_kwargs)

    # Run and process the conventional LS fit
    noisy_multiverse = np.column_stack((noisy_pcasl, noisy_vsasl))
    pldti = np.column_stack([plds, plds])
    try:
        beta_ls, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, noisy_multiverse, [50.0/6000.0, 1500.0], **asl_params_dict)
        ls_pred_cbf, ls_pred_att = beta_ls[0] * 6000.0, beta_ls[1]
        ls_pcasl = fun_PCASL_1comp_vect_pep([ls_pred_cbf / 6000.0, ls_pred_att], plds, **pcasl_kwargs)
        ls_vsasl = fun_VSASL_1comp_vect_pep([ls_pred_cbf / 6000.0, ls_pred_att], plds, **vsasl_kwargs)
    except Exception as e:
        print(f"  - Conventional LS fit failed for the showcase case: {e}")
        ls_pred_cbf, ls_pred_att, ls_pcasl, ls_vsasl = np.nan, np.nan, np.full_like(plds, np.nan), np.full_like(plds, np.nan)

    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"NN vs. LS on a Challenging Case (True ATT = {true_att:.0f} ms)\n"
                 f"  True Params: CBF={true_cbf:.1f}, ATT={true_att:.0f}\n"
                 f"  NN   Params: CBF={pred_cbf:.1f} (Δ {pred_cbf-true_cbf:.1f}), ATT={pred_att:.0f} (Δ {pred_att-true_att:.0f})\n"
                 f"  LS   Params: CBF={ls_pred_cbf:.1f} (Δ {ls_pred_cbf-true_cbf:.1f}), ATT={ls_pred_att:.0f} (Δ {ls_pred_att-true_att:.0f})",
                 fontsize=13)
    ax1.plot(plds, noisy_pcasl, 'ko', alpha=0.7, label='Noisy Input'); ax1.plot(plds, clean_pcasl, 'g-', lw=3, label='Ground Truth Signal')
    ax1.plot(plds, pred_pcasl, 'r--', lw=2.5, label='NN Reconstructed Signal'); ax1.plot(plds, ls_pcasl, 'm:', lw=2.5, label='LS Reconstructed Signal')
    ax1.set_title('PCASL Signal'); ax1.set_xlabel('PLD (ms)'); ax1.set_ylabel('Signal'); ax1.legend(); ax1.grid(True, alpha=0.5)
    ax2.plot(plds, noisy_vsasl, 'ko', alpha=0.7, label='Noisy Input'); ax2.plot(plds, clean_vsasl, 'g-', lw=3, label='Ground Truth Signal')
    ax2.plot(plds, pred_vsasl, 'r--', lw=2.5, label='NN Reconstructed Signal'); ax2.plot(plds, ls_vsasl, 'm:', lw=2.5, label='LS Reconstructed Signal')
    ax2.set_title('VSASL Signal'); ax2.set_xlabel('PLD (ms)'); ax2.legend(); ax2.grid(True, alpha=0.5)
    plt.tight_layout(rect=[0, 0, 1, 0.85]); plt.show()
    

def analyze_training_data_distribution(config):
    """Generates the full training dataset and plots its 2D distribution."""
    print("\n--- Running Part 2: Analyzing Training Data Distribution ---")
    
    asl_params = ASLParameters(**{k:v for k,v in config.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params)
    plds = np.array(config['pld_values'])
    
    num_subjects_for_dist_plot = 10000 
    print(f"Generating balanced dataset with {num_subjects_for_dist_plot} subjects... (This may take a minute or two)")
    
    # --- FIX: Use generate_diverse_dataset as generate_balanced_dataset was removed ---
    training_data = simulator.generate_diverse_dataset(
        plds=plds, 
        n_subjects=num_subjects_for_dist_plot,
        conditions=['healthy', 'stroke', 'tumor', 'elderly'],
        noise_levels=config.get('training_noise_levels_stage1', [3.0, 5.0, 10.0, 15.0])
    )
    
    if 'parameters' not in training_data or training_data['parameters'].shape[0] == 0:
        print("Could not generate training data for distribution analysis. Skipping.")
        return

    true_atts, true_cbfs = training_data['parameters'][:, 1], training_data['parameters'][:, 0]
    
    print("Plotting 2D histogram of the training data...")
    plt.figure(figsize=(10, 8))
    plt.hist2d(true_atts, true_cbfs, bins=(50, 50), cmap='viridis', cmin=1)
    plt.colorbar(label='Number of Samples in Bin')
    plt.xlabel('True Arterial Transit Time (ATT) [ms]'); plt.ylabel('True Cerebral Blood Flow (CBF) [ml/100g/min]')
    plt.title(f'2D Distribution of Training Dataset ({len(true_cbfs)} total samples)')
    plt.grid(True, alpha=0.2); plt.show()


# --- Main Execution Block ---

if __name__ == "__main__":
    np.random.seed(42)
    try:
        model, config, norm_stats = load_analysis_artifacts(RESULTS_DIR)
        analyze_healthy_adult_failure(model, config, norm_stats)
        analyze_training_data_distribution(config)
    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found. Please check your paths.\nDetails: {e}\nIs the `RESULTS_DIR` variable set correctly?")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()