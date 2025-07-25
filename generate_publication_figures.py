# generate_publication_figures.py

import os
import json
import inspect
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

# --- Configuration: SET THIS PATH ---
RESULTS_DIR = "./comprehensive_results/asl_research_20250616_175302/"  # <-- Edit this line!
# --- End Configuration ---

# Import your project's custom modules for the diagnostic plot
try:
    from enhanced_asl_network import EnhancedASLNet
    from enhanced_simulation import RealisticASLSimulator, ASLParameters
    from pcasl_functions import fun_PCASL_1comp_vect_pep
    from vsasl_functions import fun_VSASL_1comp_vect_pep
    from utils import engineer_signal_features
    from comparison_framework import apply_normalization_to_input_flat, denormalize_predictions
except ImportError as e:
    print(f"Error: Could not import necessary project modules. Make sure you are running this script from the project's root directory. Details: {e}")
    exit()


def load_data(results_dir_str: str) -> (pd.DataFrame, dict):
    """Loads the necessary CSV and JSON files for plotting."""
    print("--- Loading data artifacts ---")
    results_dir = os.path.join(results_dir_str)
    benchmark_csv_path = os.path.join(results_dir, 'benchmark_performance_summary.csv')
    final_results_json_path = os.path.join(results_dir, 'final_research_results.json')

    if not os.path.exists(benchmark_csv_path): raise FileNotFoundError(f"Benchmark CSV not found at: {benchmark_csv_path}")
    if not os.path.exists(final_results_json_path): raise FileNotFoundError(f"Final results JSON not found at: {final_results_json_path}")
        
    df_benchmark = pd.read_csv(benchmark_csv_path)
    print(f"Loaded benchmark data from: {benchmark_csv_path}")
    with open(final_results_json_path, 'r') as f: final_results = json.load(f)
    print(f"Loaded clinical and config data from: {final_results_json_path}")

    return df_benchmark, final_results


def plot_performance_metrics(df: pd.DataFrame, output_dir: str):
    """
    Generate Figure 1: A 2x2 grid comparing the Neural Network and Least-Squares
    on key precision (CoV) and accuracy (nRMSE) metrics.
    """
    print("Generating Figure 1: Core Performance Metrics...")
    df_plot = df[df['method'].isin(['NN', 'LS'])].copy()
    method_order = ['LS', 'NN']; palette = {'LS': '#4c72b0', 'NN': '#dd8452'}
    
    sns.set_theme(style="whitegrid", context="talk")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), sharex=True)
    fig.suptitle('Performance Comparison: Neural Network vs. Least-Squares Fitting', fontsize=24, weight='bold')

    metrics_to_plot = [
        ('cbf_cov', 'CBF Precision (CoV %)', axes[0, 0]),
        ('cbf_nrmse_perc', 'CBF Accuracy (nRMSE %)', axes[0, 1]),
        ('att_cov', 'ATT Precision (CoV %)', axes[1, 0]),
        ('att_nrmse_perc', 'ATT Accuracy (nRMSE %)', axes[1, 1])
    ]
    for metric_key, title, ax in metrics_to_plot:
        sns.barplot(data=df_plot, x='att_range_name', y=metric_key, hue='method',
                    hue_order=method_order, palette=palette, ax=ax, errorbar=None)
        ax.set_title(title, fontsize=18, weight='medium'); ax.set_ylabel(''); ax.set_xlabel('')
        ax.tick_params(axis='x', labelsize=14); ax.tick_params(axis='y', labelsize=14); ax.get_legend().remove()

    axes[1, 0].set_xlabel('ATT Range', fontsize=16); axes[1, 1].set_xlabel('ATT Range', fontsize=16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', fontsize=16, bbox_to_anchor=(0.95, 0.95), title='Method')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig_path = os.path.join(output_dir, 'figure1_performance_summary.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close(fig)
    print(f"Saved Figure 1 to: {fig_path}")


def plot_clinical_scenarios(clinical_data: dict, output_dir: str):
    """
    Generate Figure 2: A grouped bar chart showing RMSE in simulated clinical scenarios.
    """
    print("Generating Figure 2: Clinical Scenario Validation...")
    plot_data = []
    for scenario, methods in clinical_data.items():
        for method, metrics in methods.items():
            if not metrics: continue
            scenario_name_map = {'healthy_adult': 'Healthy', 'elderly_patient': 'Elderly', 'stroke_patient': 'Stroke', 'tumor_patient': 'Tumor'}
            plot_data.append({'Scenario': scenario_name_map.get(scenario, scenario), 'Method': method, 'Metric': 'CBF RMSE', 'Value': metrics.get('cbf_rmse', np.nan)})
            plot_data.append({'Scenario': scenario_name_map.get(scenario, scenario), 'Method': 'ATT RMSE', 'Value': metrics.get('att_rmse', np.nan)})
    df_plot = pd.DataFrame(plot_data)
    method_order = ['LS (1-repeat)', 'LS (4-repeat)', 'NN (1-repeat)', 'NN (4-repeat)']
    
    sns.set_theme(style="whitegrid", context="talk")
    g = sns.catplot(data=df_plot, kind='bar', x='Scenario', y='Value', hue='Method', col='Metric',
                    hue_order=[m for m in method_order if m in df_plot['Method'].unique()],
                    sharey=False, height=6, aspect=1.2, palette='viridis', legend_out=True)
    g.fig.suptitle('Performance in Simulated Clinical Scenarios', fontsize=22, weight='bold', y=1.03)
    g.set_axis_labels("", "Root Mean Square Error (RMSE)"); g.set_titles("{col_name}", size=18)
    g.despine(left=True); g.legend.set_title("Method"); plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(output_dir, 'figure2_clinical_scenarios.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close()
    print(f"Saved Figure 2 to: {fig_path}")


def plot_diagnostic_example(results_dir_str: str, output_dir: str):
    """
    Generate Figure 3: A diagnostic plot for a single challenging case,
    showing how the NN's prediction reconstructs the signal.
    """
    print("Generating Figure 3: Diagnostic Signal Reconstruction...")
    try:
        # --- FIX: Robustly load config, handling potential Optuna updates ---
        results_dir = os.path.join(results_dir_str)
        with open(os.path.join(results_dir, 'research_config.json'), 'r') as f: config = json.load(f)
        final_results_path = os.path.join(results_dir, 'final_research_results.json')
        if os.path.exists(final_results_path):
            with open(final_results_path, 'r') as f: final_results = json.load(f)
            if 'optuna_best_params' in final_results and final_results['optuna_best_params']:
                best_params = final_results['optuna_best_params']
                config['hidden_sizes'] = [best_params.get(h) for h in ['hidden_size_1', 'hidden_size_2', 'hidden_size_3']]
                config['dropout_rate'] = best_params.get('dropout_rate')
        
        with open(os.path.join(results_dir, 'norm_stats.json'), 'r') as f: norm_stats = json.load(f)
        model_path = os.path.join(results_dir, 'trained_models', 'ensemble_model_0.pt')
        num_plds = len(config.get('pld_values', [])); input_size = num_plds * 2 + 4
        
        model = EnhancedASLNet(input_size=input_size, **config)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))); model.eval()
    except Exception as e:
        print(f"Could not load model for diagnostic plot. Skipping. Error: {e}"); return

    asl_params = ASLParameters(**{k: v for k, v in config.items() if k in ASLParameters.__annotations__})
    simulator = RealisticASLSimulator(params=asl_params)
    plds = np.array(config['pld_values'])
    true_cbf, true_att, tsnr = 40.0, 3200.0, 5.0
    
    data_dict = simulator.generate_synthetic_data(plds, np.array([true_att]), n_noise=1, tsnr=tsnr, cbf_val=true_cbf)
    noisy_pcasl = data_dict['MULTIVERSE'][0, 0, :, 0]; noisy_vsasl = data_dict['MULTIVERSE'][0, 0, :, 1]
    raw_signal = np.concatenate([noisy_pcasl, noisy_vsasl])

    # --- FIX: Correctly normalize input signal before feeding to the model ---
    engineered_feats = engineer_signal_features(raw_signal.reshape(1, -1), len(plds))
    nn_input_unnorm = np.concatenate([raw_signal, engineered_feats.flatten()])
    nn_input_norm = apply_normalization_to_input_flat(nn_input_unnorm, norm_stats, num_plds, has_m0=False)
    input_tensor = torch.FloatTensor(nn_input_norm).unsqueeze(0)
    
    with torch.no_grad():
        cbf_pred_norm, att_pred_norm, _, _, _, _ = model(input_tensor) # Correctly unpack 6 return values
    
    # --- FIX: Correctly denormalize the model's output ---
    pred_cbf, pred_att, _, _ = denormalize_predictions(
        cbf_pred_norm.numpy(), att_pred_norm.numpy(), None, None, norm_stats
    )
    pred_cbf, pred_att = pred_cbf.item(), pred_att.item()
    
    pcasl_kwargs = {k: v for k, v in config.items() if k in ['T1_artery', 'T_tau', 'T2_factor', 'alpha_BS1', 'alpha_PCASL']}
    vsasl_kwargs = {k: v for k, v in config.items() if k in ['T1_artery', 'T2_factor', 'alpha_BS1', 'alpha_VSASL']}
    clean_pcasl = fun_PCASL_1comp_vect_pep([true_cbf / 6000.0, true_att], plds, **pcasl_kwargs)
    clean_vsasl = fun_VSASL_1comp_vect_pep([true_cbf / 6000.0, true_att], plds, **vsasl_kwargs)
    pred_pcasl = fun_PCASL_1comp_vect_pep([pred_cbf / 6000.0, pred_att], plds, **pcasl_kwargs)
    pred_vsasl = fun_VSASL_1comp_vect_pep([pred_cbf / 6000.0, pred_att], plds, **vsasl_kwargs)
    
    sns.set_theme(style="ticks", context="talk"); fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f"Diagnostic Example: Long ATT (True ATT = {true_att:.0f} ms)", fontsize=22, weight='bold')
    ax1.plot(plds, noisy_pcasl, 'o', color='gray', label='Noisy Input Data', alpha=0.8)
    ax1.plot(plds, clean_pcasl, '-', color='#1f77b4', lw=3, label='Ground Truth Signal')
    ax1.plot(plds, pred_pcasl, '--', color='#ff7f0e', lw=3, label='NN Reconstructed Signal')
    ax1.set_title(f'PCASL Signal\nPred: CBF={pred_cbf:.1f}, ATT={pred_att:.0f}', fontsize=16); ax1.set_xlabel('PLD (ms)', fontsize=14); ax1.set_ylabel('Signal (a.u.)', fontsize=14); ax1.grid(True, linestyle='--', alpha=0.6)
    ax2.plot(plds, noisy_vsasl, 'o', color='gray', label='Noisy Input Data', alpha=0.8)
    ax2.plot(plds, clean_vsasl, '-', color='#1f77b4', lw=3, label='Ground Truth Signal')
    ax2.plot(plds, pred_vsasl, '--', color='#ff7f0e', lw=3, label='NN Reconstructed Signal')
    ax2.set_title(f'VSASL Signal\nTrue: CBF={true_cbf:.1f}, ATT={true_att:.0f}', fontsize=16); ax2.set_xlabel('Inflow Time (ms)', fontsize=14)
    handles, labels = ax1.get_legend_handles_labels(); fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.94), ncol=3, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.9])
    fig_path = os.path.join(output_dir, 'figure3_diagnostic_plot.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight'); plt.close(fig); print(f"Saved Figure 3 to: {fig_path}")

if __name__ == "__main__":
    try:
        df_benchmark, final_results = load_data(RESULTS_DIR)
        output_figure_dir = os.path.join(RESULTS_DIR, "publication_figures"); os.makedirs(output_figure_dir, exist_ok=True)
        plot_performance_metrics(df_benchmark, output_figure_dir)
        plot_clinical_scenarios(final_results.get('clinical_validation_results', {}), output_figure_dir)
        plot_diagnostic_example(RESULTS_DIR, output_figure_dir)
        print("\n--- Publication figure generation complete! ---"); print(f"Figures saved in: {output_figure_dir}")
    except FileNotFoundError as e:
        print(f"\nERROR: A required file was not found. Please check your paths.\nDetails: {e}\nIs `RESULTS_DIR` set correctly?")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}"); import traceback; traceback.print_exc()