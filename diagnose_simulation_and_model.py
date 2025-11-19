import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import argparse
import sys
import scipy.optimize
from tqdm import tqdm
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# --- Project Imports ---
try:
    from asl_simulation import ASLParameters, _generate_pcasl_signal_jit, _generate_vsasl_signal_jit
    from enhanced_simulation import RealisticASLSimulator
    from enhanced_asl_network import DisentangledASLNet
    from asl_trainer import EnhancedASLTrainer
    from utils import engineer_signal_features_torch
except ImportError as e:
    print(f"Error importing project modules: {e}")
    sys.exit(1)

# --- Configuration ---
LS_INITIAL_GUESS = [60.0, 1500.0] # [CBF, ATT] - The "trap" for LS
LS_BOUNDS = ([0, 0], [150, 4000]) # Bounds for LS fitting

def standard_kinetic_model(p, plds, params):
    """
    Standard 2-compartment kinetic model function for Least Squares fitting.
    Matches the simulator physics exactly.
    """
    cbf_val = p[0] / 6000.0 # Convert to ml/g/s
    att_val = p[1]
    
    # Unpack physics constants
    t1_a = params.T1_artery
    t_tau = params.T_tau
    alpha_pc = params.alpha_PCASL
    alpha_vs = params.alpha_VSASL
    
    # Generate expected signals
    pcasl_sig = _generate_pcasl_signal_jit(plds, att_val, cbf_val, t1_a, t_tau, alpha_pc, 1.0)
    vsasl_sig = _generate_vsasl_signal_jit(plds, att_val, cbf_val, t1_a, alpha_vs, 1.0, 2000.0)
    
    return np.concatenate([pcasl_sig, vsasl_sig])

def fit_least_squares(noisy_signals, plds, asl_params):
    """
    Runs Scipy Least Squares for a batch of signals.
    Uses randomized initialization to prevent artificial 'perfect' fits.
    """
    preds = []
    stuck_count = 0
    
    for sig in noisy_signals:
        # Define residual function: (Model - Data)
        def residual(p):
            return standard_kinetic_model(p, plds, asl_params) - sig
        
        # Randomize initial guess to simulate clinical uncertainty
        # CBF: 40-80, ATT: 1000-2000 (Centered around standard 60/1500)
        current_guess = [np.random.uniform(40, 80), np.random.uniform(1000, 2000)]

        try:
            res = scipy.optimize.least_squares(
                residual, 
                x0=current_guess, 
                bounds=LS_BOUNDS, 
                method='trf',
                ftol=1e-3, # Loose tolerance for speed
                xtol=1e-3
            )
            pred = res.x
        except:
            pred = current_guess
            
        # Check if solver got "stuck" at initial guess (common failure mode)
        # We check if it's close to the *specific* random guess used for this iteration
        if np.allclose(pred, current_guess, atol=0.1):
            stuck_count += 1
            
        preds.append(pred)
    
    return np.array(preds), stuck_count

def load_model_and_config(run_dir):
    run_path = Path(run_dir)
    with open(run_path / "research_config.json", 'r') as f:
        config = json.load(f)
    with open(run_path / "norm_stats.json", 'r') as f:
        norm_stats = json.load(f)
        
    # Determine input size
    num_plds = len(config['pld_values'])
    input_size = num_plds * 2 + 8 # 8 scalar features
    
    model = DisentangledASLNet(mode='regression', input_size=input_size, **config)
    
    # Load ensemble
    models = []
    model_files = sorted(list((run_path / "trained_models").glob("ensemble_model_*.pt")))
    if not model_files:
        print("No ensemble models found! Checking for single model...")
        # Fallback logic if needed
        
    for mf in model_files:
        m = DisentangledASLNet(mode='regression', input_size=input_size, **config)
        m.load_state_dict(torch.load(mf, map_location='cpu'))
        m.eval()
        models.append(m)
        
    return models, config, norm_stats

def predict_nn_ensemble(models, signals, norm_stats, device):
    """
    Run inference using the Neural Network Ensemble.
    Handles preprocessing internally.
    """
    # 1. Preprocessing (Move to Torch GPU)
    signals_tensor = torch.from_numpy(signals).float().to(device)
    num_plds = signals.shape[1] // 2
    
    # Feature Engineering
    eng_features = engineer_signal_features_torch(signals_tensor, num_plds)
    
    # Normalization
    pcasl = signals_tensor[:, :num_plds]
    vsasl = signals_tensor[:, num_plds:]
    
    p_mu = pcasl.mean(dim=1, keepdim=True)
    p_std = pcasl.std(dim=1, keepdim=True) + 1e-6
    v_mu = vsasl.mean(dim=1, keepdim=True)
    v_std = vsasl.std(dim=1, keepdim=True) + 1e-6
    
    p_shape = (pcasl - p_mu) / p_std
    v_shape = (vsasl - v_mu) / v_std
    
    scalars = torch.cat([p_mu, p_std, v_mu, v_std, eng_features], dim=1)
    
    # Normalize scalars using stored stats
    s_mean = torch.tensor(norm_stats['scalar_features_mean'], device=device).float()
    s_std = torch.tensor(norm_stats['scalar_features_std'], device=device).float()
    scalars_norm = (scalars - s_mean) / s_std
    
    model_input = torch.cat([p_shape, v_shape, scalars_norm], dim=1)

    model_input = torch.clamp(model_input, -15.0, 15.0)
    
    # 2. Inference
    preds_cbf = []
    preds_att = []
    
    with torch.no_grad():
        for model in models:
            model = model.to(device)
            out = model(model_input)
            # out = (cbf_mean, att_mean, cbf_logvar, att_logvar)
            preds_cbf.append(out[0].cpu().numpy())
            preds_att.append(out[1].cpu().numpy())
            
    # 3. Ensemble Averaging
    avg_cbf_norm = np.mean(preds_cbf, axis=0)
    avg_att_norm = np.mean(preds_att, axis=0)
    
    # 4. Denormalization
    pred_cbf = avg_cbf_norm * norm_stats['y_std_cbf'] + norm_stats['y_mean_cbf']
    pred_att = avg_att_norm * norm_stats['y_std_att'] + norm_stats['y_mean_att']
    
    return np.stack([pred_cbf, pred_att], axis=1)

def run_experiment(name, param_grid, simulator, models, norm_stats, config, output_dir, snr_level=5.0):
    print(f"\n>>> Running Experiment: {name} (SNR={snr_level})")
    
    results = []
    plds = np.array(config['pld_values'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Prepare Simulator Params for LS
    asl_params = ASLParameters(**{k:v for k,v in config.items() if k in ASLParameters.__annotations__})

    for params in tqdm(param_grid):
        true_cbf = params['cbf']
        true_att = params['att']
        
        # 1. Generate Clean Signal
        # Note: Using simulator to generate clean, then adding simple Gaussian noise 
        # to match the "Simple Gaussian" training paradigm perfectly.
        vsasl_clean = simulator._generate_vsasl_signal(plds, true_att, true_cbf, simulator.params.T1_artery, simulator.params.alpha_VSASL)
        pcasl_clean = simulator._generate_pcasl_signal(plds, true_att, true_cbf, simulator.params.T1_artery, simulator.params.T_tau, simulator.params.alpha_PCASL)
        clean_sig = np.concatenate([pcasl_clean, vsasl_clean])
        
        # 2. Add Noise (Batch of 100 repetitions for stats)
        n_reps = 100
        clean_batch = np.tile(clean_sig, (n_reps, 1))
        
        # Calculate noise sigma based on reference signal
        ref_sig = simulator._compute_reference_signal()
        noise_std = ref_sig / snr_level
        
        # Apply TR Scaling
        scalings = simulator.compute_tr_noise_scaling(plds)
        scale_vec = np.concatenate([np.full(len(plds), scalings['PCASL']), np.full(len(plds), scalings['VSASL'])])
        
        noise = np.random.randn(*clean_batch.shape) * noise_std * scale_vec
        noisy_batch = clean_batch + noise
        
        # 3. Neural Network Prediction
        nn_preds = predict_nn_ensemble(models, noisy_batch, norm_stats, device)
        
        # 4. Least Squares Prediction
        ls_preds, stuck_count = fit_least_squares(noisy_batch, plds, asl_params)
        
        # 5. Metrics Calculation
        # NN Metrics
        nn_cbf_err = nn_preds[:, 0] - true_cbf
        nn_att_err = nn_preds[:, 1] - true_att
        
        # LS Metrics
        ls_cbf_err = ls_preds[:, 0] - true_cbf
        ls_att_err = ls_preds[:, 1] - true_att
        
        results.append({
            'True_CBF': true_cbf,
            'True_ATT': true_att,
            
            # Neural Net Stats
            'NN_CBF_Bias': np.mean(nn_cbf_err),
            'NN_CBF_MAE': np.mean(np.abs(nn_cbf_err)),
            'NN_CBF_SD': np.std(nn_cbf_err),
            'NN_ATT_Bias': np.mean(nn_att_err),
            'NN_ATT_MAE': np.mean(np.abs(nn_att_err)),
            'NN_ATT_SD': np.std(nn_att_err),
            
            # Least Squares Stats
            'LS_CBF_Bias': np.mean(ls_cbf_err),
            'LS_CBF_MAE': np.mean(np.abs(ls_cbf_err)),
            'LS_CBF_SD': np.std(ls_cbf_err),
            'LS_ATT_Bias': np.mean(ls_att_err),
            'LS_ATT_MAE': np.mean(np.abs(ls_att_err)),
            'LS_ATT_SD': np.std(ls_att_err),
            'LS_Stuck_Pct': (stuck_count / n_reps) * 100
        })

    df = pd.DataFrame(results)
    df.to_csv(output_dir / f"results_{name}.csv", index=False)
    return df

def plot_results(df, x_axis, name, output_dir):
    """Generates detailed Bias/Precision plots."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot Styling
    sns.set_style("whitegrid")
    colors = {'NN': '#E63946', 'LS': '#457B9D'} # Red vs Blue
    
    # --- CBF Plots ---
    # Bias (Accuracy)
    ax = axes[0,0]
    ax.plot(df[x_axis], df['NN_CBF_Bias'], label='Neural Net Bias', color=colors['NN'], marker='o', linewidth=2)
    ax.plot(df[x_axis], df['LS_CBF_Bias'], label='Least Squares Bias', color=colors['LS'], marker='s', linestyle='--', linewidth=2)
    ax.fill_between(df[x_axis], 
                    df['NN_CBF_Bias'] - df['NN_CBF_SD'], 
                    df['NN_CBF_Bias'] + df['NN_CBF_SD'], 
                    color=colors['NN'], alpha=0.15, label='NN Precision (SD)')
    ax.set_title(f"CBF Accuracy (Bias & Precision) vs {x_axis}")
    ax.set_ylabel("CBF Error (ml/100g/min)")
    ax.legend()
    
    # MAE (Overall Error)
    ax = axes[1,0]
    ax.plot(df[x_axis], df['NN_CBF_MAE'], label='Neural Net MAE', color=colors['NN'], marker='o')
    ax.plot(df[x_axis], df['LS_CBF_MAE'], label='Least Squares MAE', color=colors['LS'], marker='s', linestyle='--')
    ax.set_title(f"CBF Mean Absolute Error vs {x_axis}")
    ax.set_ylabel("MAE")
    ax.set_xlabel(x_axis)
    
    # --- ATT Plots ---
    # Bias
    ax = axes[0,1]
    ax.plot(df[x_axis], df['NN_ATT_Bias'], label='Neural Net Bias', color=colors['NN'], marker='o', linewidth=2)
    ax.plot(df[x_axis], df['LS_ATT_Bias'], label='Least Squares Bias', color=colors['LS'], marker='s', linestyle='--', linewidth=2)
    ax.fill_between(df[x_axis], 
                    df['NN_ATT_Bias'] - df['NN_ATT_SD'], 
                    df['NN_ATT_Bias'] + df['NN_ATT_SD'], 
                    color=colors['NN'], alpha=0.15, label='NN Precision (SD)')
    ax.set_title(f"ATT Accuracy (Bias & Precision) vs {x_axis}")
    ax.set_ylabel("ATT Error (ms)")
    ax.legend()
    
    # MAE
    ax = axes[1,1]
    ax.plot(df[x_axis], df['NN_ATT_MAE'], label='Neural Net MAE', color=colors['NN'], marker='o')
    ax.plot(df[x_axis], df['LS_ATT_MAE'], label='Least Squares MAE', color=colors['LS'], marker='s', linestyle='--')
    
    # Add LS Stuck annotation if relevant
    stuck_avg = df['LS_Stuck_Pct'].mean()
    if stuck_avg > 5:
        ax.annotate(f"LS Stuck Rate: {stuck_avg:.1f}%", xy=(0.05, 0.9), xycoords='axes fraction', color='red', fontweight='bold')
        
    ax.set_title(f"ATT Mean Absolute Error vs {x_axis}")
    ax.set_ylabel("MAE")
    ax.set_xlabel(x_axis)
    
    plt.tight_layout()
    plt.savefig(output_dir / f"Plot_{name}.png", dpi=150)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=str, help="Path to the training run directory")
    parser.add_argument("--output_dir", type=str, default="diagnostics_output")
    args = parser.parse_args()
    
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    print(f"Loading artifacts from {args.run_dir}...")
    models, config, norm_stats = load_model_and_config(args.run_dir)
    
    sim_params = ASLParameters(**{k:v for k,v in config.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=sim_params)
    
    # --- Scenario 1: Varying ATT (Fixed CBF) ---
    # Matches "First Figure" request
    att_range = np.linspace(500, 4000, 15)
    grid_A = [{'att': a, 'cbf': 60.0} for a in att_range]
    df_A = run_experiment("A_VaryingATT_FixedCBF", grid_A, simulator, models, norm_stats, config, output_path, snr_level=5.0)
    plot_results(df_A, "True_ATT", "A_VaryingATT_FixedCBF", output_path)
    
    # --- Scenario 2: Varying CBF (Fixed ATT) ---
    # Matches "Second Figure" request
    cbf_range = np.linspace(10, 100, 10)
    grid_B = [{'att': 1500.0, 'cbf': c} for c in cbf_range]
    df_B = run_experiment("B_VaryingCBF_FixedATT", grid_B, simulator, models, norm_stats, config, output_path, snr_level=5.0)
    plot_results(df_B, "True_CBF", "B_VaryingCBF_FixedATT", output_path)
    
    # --- Scenario 3: High Noise Varying ATT ---
    # Matches "Fourth Figure" request
    df_C = run_experiment("C_HighNoise_VaryingATT", grid_A, simulator, models, norm_stats, config, output_path, snr_level=2.0) # SNR dropped to 2.0
    plot_results(df_C, "True_ATT", "C_HighNoise_VaryingATT", output_path)
    
    # --- Scenario 4: The PI's "Third Figure" (Global Robustness) ---
    # "Group all different CBF values... plot as function of ATT"
    # We create a random grid of (CBF, ATT) pairs, then bin by ATT
    print("\n>>> Running Experiment: D_Global_Robustness (Random Grid)")
    n_samples = 500
    random_grid = []
    for _ in range(n_samples):
        random_grid.append({
            'cbf': np.random.uniform(20, 100),
            'att': np.random.uniform(500, 4000)
        })
    
    df_D_raw = run_experiment("D_Global_Raw", random_grid, simulator, models, norm_stats, config, output_path)
    
    # Binning for plotting
    df_D_raw['ATT_Bin'] = pd.cut(df_D_raw['True_ATT'], bins=np.linspace(500, 4000, 8), labels=np.linspace(750, 3750, 7))
    df_D_grouped = df_D_raw.groupby('ATT_Bin').mean().reset_index()
    # Need to handle numeric conversion for plotting
    df_D_grouped['True_ATT'] = df_D_grouped['ATT_Bin'].astype(float)
    
    plot_results(df_D_grouped, "True_ATT", "D_Global_Robustness", output_path)

    print("\n===========================================")
    print("       DIAGNOSTIC SUMMARY TABLES           ")
    print("===========================================")
    
    print("\n1. VARYING ATT (Fixed CBF=60) - AVERAGE PERFORMANCE")
    print(df_A[['True_ATT', 'NN_CBF_MAE', 'LS_CBF_MAE', 'NN_ATT_MAE', 'LS_ATT_MAE']].round(2).to_string(index=False))
    
    print("\n2. LS STUCK PERCENTAGE (Did LS fail to converge?)")
    stuck_A = df_A['LS_Stuck_Pct'].mean()
    stuck_C = df_C['LS_Stuck_Pct'].mean()
    print(f"Scenario A (Normal Noise): {stuck_A:.1f}% of samples stuck")
    print(f"Scenario C (High Noise):   {stuck_C:.1f}% of samples stuck")
    
    print(f"\nDone! Detailed CSVs and plots saved to {output_path}")

if __name__ == "__main__":
    main()