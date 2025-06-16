import torch
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import inspect

# --- Configuration: SET THIS PATH ---
RESULTS_DIR = "./comprehensive_results/asl_research_20250616_130042/"  # <-- Edit this line!
# --- End Configuration ---

# Import your project's custom modules
try:
    from enhanced_asl_network import EnhancedASLNet
    from enhanced_simulation import RealisticASLSimulator, ASLParameters
    from main import engineer_signal_features  # We need the exact feature engineering
    from pcasl_functions import fun_PCASL_1comp_vect_pep
    from vsasl_functions import fun_VSASL_1comp_vect_pep
except ImportError as e:
    print(f"Error: Could not import necessary project modules. Make sure you are running this script from the project's root directory. Details: {e}")
    exit()

# --- Utility Functions ---

# In diagnostic_analysis.py

def load_analysis_artifacts(results_dir: str):
    """Loads the model, config, and norm_stats for analysis.
    
    Args:
        results_dir: Path to results directory as string
    """
    print("--- Loading Artifacts ---")

    # Ensure the directory exists
    if not os.path.exists(results_dir):
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    # 1. Load the initial config file
    config_path = os.path.join(results_dir, 'research_config.json')
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Initial config file not found: {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)
    print(f"Loaded initial config from: {config_path}")

    # 2. Load the final results to get the *actual* parameters used for training
    final_results_path = os.path.join(results_dir, 'final_research_results.json')
    if os.path.exists(final_results_path):
        with open(final_results_path, 'r') as f:
            final_results = json.load(f)
        
        # Check for Optuna's best params and update the config with them
        if 'optuna_best_params' in final_results and final_results['optuna_best_params']:
            print("Found Optuna results. Updating config with optimized parameters.")
            best_params = final_results['optuna_best_params']
            
            # This logic needs to match how main.py updates the config
            if all(k in best_params for k in ['hidden_size_1', 'hidden_size_2', 'hidden_size_3']):
                 config['hidden_sizes'] = [
                    best_params['hidden_size_1'], best_params['hidden_size_2'], best_params['hidden_size_3']
                 ]
            if 'dropout_rate' in best_params:
                config['dropout_rate'] = best_params['dropout_rate']
            
            print(f"Using optimized hidden_sizes: {config['hidden_sizes']}")
        
        # More robust: use the config saved within the final results if it exists
        elif 'config' in final_results:
             print("Loading config from final_research_results.json for accuracy.")
             config = final_results['config']

    # Load the normalization stats
    norm_stats_path = os.path.join(results_dir, 'norm_stats.json')
    if not os.path.exists(norm_stats_path):
        raise FileNotFoundError(f"Normalization stats file not found: {norm_stats_path}")
    
    with open(norm_stats_path, 'r') as f:
        norm_stats = json.load(f)
    print(f"Loaded norm_stats from: {norm_stats_path}")

    # Load the first model from the ensemble
    model_path = os.path.join(results_dir, 'trained_models', 'ensemble_model_0.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Re-create the model with the correct architecture from the now-updated config file
    plds_np = np.array(config.get('pld_values', range(500, 3001, 500)))
    num_plds = len(plds_np)
    # The input size is num_plds * 2 (raw signals) + 4 (engineered features)
    base_input_size_nn = num_plds * 2 + 4

    # Filter kwargs to match EnhancedASLNet signature
    model_param_keys = inspect.signature(EnhancedASLNet).parameters.keys()
    filtered_kwargs = {k: v for k, v in config.items() if k in model_param_keys}
    
    # Add n_plds to the config for the model if it's missing
    if 'n_plds' not in filtered_kwargs:
        filtered_kwargs['n_plds'] = num_plds

    model = EnhancedASLNet(input_size=base_input_size_nn, **filtered_kwargs)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    print(f"Successfully loaded model from: {model_path}")
    
    return model, config, norm_stats

# --- Analysis Functions ---

def analyze_healthy_adult_failure(model, config, norm_stats):
    """Generates data, finds the worst predictions, and plots them."""
    print("\n--- Running Part 1: Interrogating 'Healthy Adult' Failure ---")
    
    # Setup simulator with parameters from the config
    asl_params_dict = {
        'T1_artery': config.get('T1_artery', 1850.0),
        'T_tau': config.get('T_tau', 1800.0),
        'alpha_PCASL': config.get('alpha_PCASL', 0.85),
        'alpha_VSASL': config.get('alpha_VSASL', 0.56),
        'alpha_BS1': config.get('alpha_BS1', 1.0),
        'T2_factor': config.get('T2_factor', 1.0)
    }
    asl_params = ASLParameters(**{k: v for k, v in asl_params_dict.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params)
    plds = np.array(config['pld_values'])
    
    # Create specific parameter dictionaries for each function to avoid the TypeError
    pcasl_kwargs = {
        'T1_artery': asl_params_dict['T1_artery'],
        'T_tau': asl_params_dict['T_tau'],
        'T2_factor': asl_params_dict['T2_factor'],
        'alpha_BS1': asl_params_dict['alpha_BS1'],
        'alpha_PCASL': asl_params_dict['alpha_PCASL']
    }
    vsasl_kwargs = {
        'T1_artery': asl_params_dict['T1_artery'],
        'T2_factor': asl_params_dict['T2_factor'],
        'alpha_BS1': asl_params_dict['alpha_BS1'],
        'alpha_VSASL': asl_params_dict['alpha_VSASL']
    }
    
    # Generate the "Healthy Adult" clinical scenario data
    scenario_params = config['clinical_scenario_definitions']['healthy_adult']
    n_subjects = 50  # Generate a decent number of samples to find bad ones
    true_cbfs = np.random.uniform(*scenario_params['cbf_range'], n_subjects)
    true_atts = np.random.uniform(*scenario_params['att_range'], n_subjects)
    snr = scenario_params['snr']
    
    results = []
    print(f"Generating {n_subjects} healthy adult samples and making predictions...")
    
    for i in range(n_subjects):
        # 1. Generate one noisy sample
        data_dict = simulator.generate_synthetic_data(
            plds, np.array([true_atts[i]]), n_noise=1, tsnr=snr, cbf_val=true_cbfs[i]
        )
        noisy_pcasl = data_dict['PCASL'][0, 0, :]
        noisy_vsasl = data_dict['VSASL'][0, 0, :]
        raw_signal_vector = np.concatenate([noisy_pcasl, noisy_vsasl])

        # 2. Apply the *exact* same feature pipeline as in training
        engineered_feats = engineer_signal_features(raw_signal_vector.reshape(1, -1), len(plds))
        full_input_vector = np.concatenate([raw_signal_vector, engineered_feats.flatten()])
        
        # 3. Predict with the model
        input_tensor = torch.FloatTensor(full_input_vector).unsqueeze(0)
        with torch.no_grad():
            pred_cbf_norm, pred_att_norm, _, _ = model(input_tensor)
        
        # 4. Denormalize the prediction
        pred_cbf = pred_cbf_norm.item() * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
        pred_att = pred_att_norm.item() * norm_stats['y_std_att'] + norm_stats['y_mean_att']
        
        # 5. Store everything we need for plotting
        att_error = abs(pred_att - true_atts[i])
        results.append({
            'true_cbf': true_cbfs[i], 'true_att': true_atts[i],
            'pred_cbf': pred_cbf, 'pred_att': pred_att,
            'att_error': att_error,
            'noisy_pcasl': noisy_pcasl, 'noisy_vsasl': noisy_vsasl
        })

    # Sort results to find the worst offenders
    results.sort(key=lambda x: x['att_error'], reverse=True)
    
    print(f"\nPlotting the {min(10, len(results))} worst ATT predictions...")
    for i, res in enumerate(results[:10]):
        # Generate the clean signal from TRUE parameters
        true_cbf_cgs = res['true_cbf'] / 6000.0
        clean_pcasl = fun_PCASL_1comp_vect_pep([true_cbf_cgs, res['true_att']], plds, **pcasl_kwargs)
        clean_vsasl = fun_VSASL_1comp_vect_pep([true_cbf_cgs, res['true_att']], plds, **vsasl_kwargs)

        # Generate the signal from PREDICTED parameters
        pred_cbf_cgs = res['pred_cbf'] / 6000.0
        pred_pcasl = fun_PCASL_1comp_vect_pep([pred_cbf_cgs, res['pred_att']], plds, **pcasl_kwargs)
        pred_vsasl = fun_VSASL_1comp_vect_pep([pred_cbf_cgs, res['pred_att']], plds, **vsasl_kwargs)

        # Create the plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Case #{i+1} | ATT Error: {res['att_error']:.0f} ms\n"
                     f"True: CBF={res['true_cbf']:.1f}, ATT={res['true_att']:.0f} | "
                     f"Pred: CBF={res['pred_cbf']:.1f}, ATT={res['pred_att']:.0f}",
                     fontsize=14)
        
        # PCASL Plot
        ax1.plot(plds, res['noisy_pcasl'], 'ko', label='Noisy Input')
        ax1.plot(plds, clean_pcasl, 'g-', lw=2, label='Ground Truth Signal')
        ax1.plot(plds, pred_pcasl, 'r--', lw=2, label='NN Predicted Signal')
        ax1.set_title('PCASL Signal')
        ax1.set_xlabel('PLD (ms)')
        ax1.set_ylabel('Signal')
        ax1.legend()
        ax1.grid(True, alpha=0.5)

        # VSASL Plot
        ax2.plot(plds, res['noisy_vsasl'], 'ko', label='Noisy Input')
        ax2.plot(plds, clean_vsasl, 'g-', lw=2, label='Ground Truth Signal')
        ax2.plot(plds, pred_vsasl, 'r--', lw=2, label='NN Predicted Signal')
        ax2.set_title('VSASL Signal')
        ax2.set_xlabel('PLD (ms)')
        ax2.legend()
        ax2.grid(True, alpha=0.5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()


def analyze_training_data_distribution(config):
    """Generates the full training dataset and plots its 2D distribution."""
    print("\n--- Running Part 2: Analyzing Training Data Distribution ---")
    
    # Setup simulator
    asl_params = ASLParameters(
        T1_artery=config.get('T1_artery', 1850.0), 
        T_tau=config.get('T_tau', 1800.0)
    )
    simulator = RealisticASLSimulator(params=asl_params)
    plds = np.array(config['pld_values'])
    
    # To see the distribution for the *full* 10k run, we override the config value locally.
    # The config from the file reflects the run being analyzed (which might be a debug run).
    # Here, we want to know what the *ideal* full dataset looks like.
    num_subjects_for_dist_plot = 10000 
    print(f"Generating balanced dataset with {num_subjects_for_dist_plot} subjects... (This may take a minute or two)")
    training_data = simulator.generate_balanced_dataset(
        plds=plds, 
        total_subjects=num_subjects_for_dist_plot,
        noise_levels=config.get('training_noise_levels', [3.0, 5.0, 10.0, 15.0])
    )
    
    if 'parameters' not in training_data or training_data['parameters'].shape[0] == 0:
        print("Could not generate training data for distribution analysis. Skipping.")
        return

    true_atts = training_data['parameters'][:, 1]
    true_cbfs = training_data['parameters'][:, 0]
    
    print("Plotting 2D histogram of the training data...")
    plt.figure(figsize=(10, 8))
    
    # Use hist2d to show density
    plt.hist2d(true_atts, true_cbfs, bins=(50, 50), cmap='viridis', cmin=1)
    
    plt.colorbar(label='Number of Samples in Bin')
    plt.xlabel('True Arterial Transit Time (ATT) [ms]')
    plt.ylabel('True Cerebral Blood Flow (CBF) [ml/100g/min]')
    plt.title(f'2D Distribution of Training Dataset ({num_subjects_for_dist_plot} subjects)')
    plt.grid(True, alpha=0.2)
    plt.show()


# --- Main Execution Block ---

if __name__ == "__main__":
    # Set a fixed seed for all random operations in this script for reproducibility
    np.random.seed(42)

    try:
        # Load all the necessary components from the specified results directory
        model, config, norm_stats = load_analysis_artifacts(RESULTS_DIR)

        # Run the failure analysis and plotting
        analyze_healthy_adult_failure(model, config, norm_stats)
        
        # Run the training data distribution analysis
        analyze_training_data_distribution(config)

    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found. Please check your paths.")
        print(f"Details: {e}")
        print(f"Is the `RESULTS_DIR` variable at the top of the script set correctly to a valid results folder?")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()