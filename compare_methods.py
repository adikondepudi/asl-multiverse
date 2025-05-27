import numpy as np
import matplotlib.pyplot as plt
from vsasl_functions import fun_VSASL_1comp_vect_pep, fit_VSASL_vectInit_pep
from pcasl_functions import fun_PCASL_1comp_vect_pep, fit_PCASL_vectInit_pep
from multiverse_functions import (fun_PCVSASL_misMatchPLD_vect_pep,
                                fit_PCVSASL_misMatchPLD_vectInit_pep)

def run_monte_carlo_simulation(n_iterations=1000):
    """Run Monte Carlo simulations for all three methods"""
    # Common parameters
    CBF = 60  # ml/100g/min
    cbf = CBF/6000  # Convert to ml/g/s
    T1_artery = 1850
    T2_factor = 1
    alpha_BS1 = 1
    alpha_PCASL = 0.85
    alpha_VSASL = 0.56
    T_tau = 1800
    
    # Test different ATT values
    att_values = np.array([800, 1600, 2400])
    PLDs = np.arange(500, 3001, 500)
    PLDTI = np.column_stack((PLDs, PLDs))  # Matched PLDs/TIs for MULTIVERSE
    
    # Storage for results
    results = {
        'PCASL': {att: {'cbf': [], 'att': []} for att in att_values},
        'VSASL': {att: {'cbf': [], 'att': []} for att in att_values},
        'MULTIVERSE': {att: {'cbf': [], 'att': []} for att in att_values}
    }
    
    # Run simulations
    for true_att in att_values:
        print(f"\nSimulating ATT = {true_att} ms")
        
        # Generate clean signals
        clean_pcasl = fun_PCASL_1comp_vect_pep([cbf, true_att], PLDs, T1_artery,
                                              T_tau, T2_factor, alpha_BS1, alpha_PCASL)
        clean_vsasl = fun_VSASL_1comp_vect_pep([cbf, true_att], PLDs, T1_artery,
                                              T2_factor, alpha_BS1, alpha_VSASL)
        clean_multi = fun_PCVSASL_misMatchPLD_vect_pep([cbf, true_att], PLDTI,
                                                      T1_artery, T_tau, T2_factor,
                                                      alpha_BS1, alpha_PCASL, alpha_VSASL)
        
        # Add noise and fit for each iteration
        for i in range(n_iterations):
            if i % 100 == 0:
                print(f"Iteration {i}/{n_iterations}")
                
            # Add noise
            noise_level = 0.0002
            #TODO noise level should be different btwn PCASL/VSASL and MULTIVERSE
            noisy_pcasl = clean_pcasl + np.random.normal(0, noise_level, clean_pcasl.shape)
            noisy_vsasl = clean_vsasl + np.random.normal(0, noise_level, clean_vsasl.shape)
            noisy_multi = clean_multi + np.random.normal(0, noise_level, clean_multi.shape)
            
            # Initial guesses
            init = [50/6000, 1500]
            
            # Fit each method
            try:
                # PCASL
                beta_pcasl, _, _, _ = fit_PCASL_vectInit_pep(PLDs, noisy_pcasl, init,
                    T1_artery, T_tau, T2_factor, alpha_BS1, alpha_PCASL)
                results['PCASL'][true_att]['cbf'].append(beta_pcasl[0]*6000)
                results['PCASL'][true_att]['att'].append(beta_pcasl[1])
                
                # VSASL
                beta_vsasl, _, _, _ = fit_VSASL_vectInit_pep(PLDs, noisy_vsasl, init,
                    T1_artery, T2_factor, alpha_BS1, alpha_VSASL)
                results['VSASL'][true_att]['cbf'].append(beta_vsasl[0]*6000)
                results['VSASL'][true_att]['att'].append(beta_vsasl[1])
                
                # MULTIVERSE
                beta_multi, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(PLDTI,
                    noisy_multi, init, T1_artery, T_tau, T2_factor, alpha_BS1,
                    alpha_PCASL, alpha_VSASL)
                results['MULTIVERSE'][true_att]['cbf'].append(beta_multi[0]*6000)
                results['MULTIVERSE'][true_att]['att'].append(beta_multi[1])
                
            except Exception as e:
                print(f"Fitting error at iteration {i}: {str(e)}")
                continue
    
    return results, att_values

def calculate_metrics(results, att_values, true_cbf=60):
    """Calculate performance metrics for each method"""
    metrics = {}
    
    for method in results.keys():
        metrics[method] = {'cbf': {}, 'att': {}}
        
        for att in att_values:
            # Get results for this ATT value
            cbf_estimates = np.array(results[method][att]['cbf'])
            att_estimates = np.array(results[method][att]['att'])
            
            # Filter out any outliers (optional)
            cbf_filtered = cbf_estimates[np.abs(cbf_estimates - true_cbf) < 100]
            att_filtered = att_estimates[np.abs(att_estimates - att) < 2000]
            
            # CBF metrics
            metrics[method]['cbf'][att] = {
                'bias': np.mean(cbf_filtered) - true_cbf,
                'cv': np.std(cbf_filtered) / np.mean(cbf_filtered) * 100,
                'rmse': np.sqrt(np.mean((cbf_filtered - true_cbf)**2)),
                'ci': np.percentile(cbf_filtered, [2.5, 97.5])
            }
            
            # ATT metrics
            metrics[method]['att'][att] = {
                'bias': np.mean(att_filtered) - att,
                'cv': np.std(att_filtered) / np.mean(att_filtered) * 100,
                'rmse': np.sqrt(np.mean((att_filtered - att)**2)),
                'ci': np.percentile(att_filtered, [2.5, 97.5])
            }
    
    return metrics

def plot_results(metrics, att_values):
    """Plot comparison of methods"""
    methods = list(metrics.keys())
    parameters = ['cbf', 'att']
    metric_names = ['bias', 'cv', 'rmse']
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for i, param in enumerate(parameters):
        for j, metric in enumerate(metric_names):
            ax = axes[i, j]
            
            x = np.arange(len(att_values))
            width = 0.25
            
            for k, method in enumerate(methods):
                values = [metrics[method][param][att][metric] for att in att_values]
                ax.bar(x + k*width, values, width, label=method)
            
            ax.set_xlabel('True ATT (ms)')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{param.upper()} {metric.upper()}')
            ax.set_xticks(x + width)
            ax.set_xticklabels(att_values)
            ax.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Run simulations
    print("Running Monte Carlo simulations...")
    results, att_values = run_monte_carlo_simulation(n_iterations=1000)
    
    # Calculate metrics
    print("\nCalculating performance metrics...")
    metrics = calculate_metrics(results, att_values)
    
    # Print summary
    print("\nPerformance Summary:")
    for method in metrics.keys():
        print(f"\n{method} Results:")
        for att in att_values:
            print(f"\nATT = {att} ms:")
            print("CBF Metrics:")
            for metric, value in metrics[method]['cbf'][att].items():
                if metric == 'ci':
                    print(f"  {metric.upper()}: [{value[0]:.1f}, {value[1]:.1f}]")
                else:
                    print(f"  {metric.upper()}: {value:.1f}")
            print("ATT Metrics:")
            for metric, value in metrics[method]['att'][att].items():
                if metric == 'ci':
                    print(f"  {metric.upper()}: [{value[0]:.1f}, {value[1]:.1f}]")
                else:
                    print(f"  {metric.upper()}: {value:.1f}")
    
    # Plot results
    plot_results(metrics, att_values)

if __name__ == "__main__":
    main()