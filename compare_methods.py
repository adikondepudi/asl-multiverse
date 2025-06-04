import numpy as np
import matplotlib.pyplot as plt # Kept for plt.show() for interactive testing
from vsasl_functions import fun_VSASL_1comp_vect_pep, fit_VSASL_vectInit_pep
from pcasl_functions import fun_PCASL_1comp_vect_pep, fit_PCASL_vectInit_pep
from multiverse_functions import (fun_PCVSASL_misMatchPLD_vect_pep,
                                fit_PCVSASL_misMatchPLD_vectInit_pep)

def run_monte_carlo_simulation(n_iterations=1000):
    CBF, cbf = 60, 60/6000
    T1_artery, T2_factor, alpha_BS1, alpha_PCASL, alpha_VSASL, T_tau = 1850, 1, 1, 0.85, 0.56, 1800
    att_values, PLDs = np.array([800, 1600, 2400]), np.arange(500, 3001, 500)
    PLDTI = np.column_stack((PLDs, PLDs))
    results = {'PCASL': {att: {'cbf': [], 'att': []} for att in att_values},
               'VSASL': {att: {'cbf': [], 'att': []} for att in att_values},
               'MULTIVERSE': {att: {'cbf': [], 'att': []} for att in att_values}}
    for true_att in att_values:
        print(f"\nSimulating ATT = {true_att} ms")
        clean_pcasl = fun_PCASL_1comp_vect_pep([cbf,true_att],PLDs,T1_artery,T_tau,T2_factor,alpha_BS1,alpha_PCASL)
        clean_vsasl = fun_VSASL_1comp_vect_pep([cbf,true_att],PLDs,T1_artery,T2_factor,alpha_BS1,alpha_VSASL)
        clean_multi = fun_PCVSASL_misMatchPLD_vect_pep([cbf,true_att],PLDTI,T1_artery,T_tau,T2_factor,alpha_BS1,alpha_PCASL,alpha_VSASL)
        for i in range(n_iterations):
            if i % 100 == 0: print(f"Iteration {i}/{n_iterations}")
            noise_level = 0.0002
            noisy_pcasl, noisy_vsasl, noisy_multi = clean_pcasl+np.random.normal(0,noise_level,clean_pcasl.shape), clean_vsasl+np.random.normal(0,noise_level,clean_vsasl.shape), clean_multi+np.random.normal(0,noise_level,clean_multi.shape)
            init = [50/6000, 1500]
            try:
                beta_p,_,_,_ = fit_PCASL_vectInit_pep(PLDs,noisy_pcasl,init,T1_artery,T_tau,T2_factor,alpha_BS1,alpha_PCASL)
                results['PCASL'][true_att]['cbf'].append(beta_p[0]*6000); results['PCASL'][true_att]['att'].append(beta_p[1])
                beta_v,_,_,_ = fit_VSASL_vectInit_pep(PLDs,noisy_vsasl,init,T1_artery,T2_factor,alpha_BS1,alpha_VSASL)
                results['VSASL'][true_att]['cbf'].append(beta_v[0]*6000); results['VSASL'][true_att]['att'].append(beta_v[1])
                beta_m,_,_,_ = fit_PCVSASL_misMatchPLD_vectInit_pep(PLDTI,noisy_multi,init,T1_artery,T_tau,T2_factor,alpha_BS1,alpha_PCASL,alpha_VSASL)
                results['MULTIVERSE'][true_att]['cbf'].append(beta_m[0]*6000); results['MULTIVERSE'][true_att]['att'].append(beta_m[1])
            except Exception as e: print(f"Fitting error at iter {i}: {e}"); continue
    return results, att_values

def calculate_metrics(results, att_values, true_cbf=60):
    metrics = {}
    for method in results.keys():
        metrics[method] = {'cbf': {}, 'att': {}}
        for att in att_values:
            cbf_est, att_est = np.array(results[method][att]['cbf']), np.array(results[method][att]['att'])
            cbf_filt, att_filt = cbf_est[np.abs(cbf_est-true_cbf)<100], att_est[np.abs(att_est-att)<2000]
            metrics[method]['cbf'][att] = {'bias':np.mean(cbf_filt)-true_cbf, 'cv':np.std(cbf_filt)/np.mean(cbf_filt)*100,
                                          'rmse':np.sqrt(np.mean((cbf_filt-true_cbf)**2)), 'ci':np.percentile(cbf_filt,[2.5,97.5])}
            metrics[method]['att'][att] = {'bias':np.mean(att_filt)-att, 'cv':np.std(att_filt)/np.mean(att_filt)*100,
                                          'rmse':np.sqrt(np.mean((att_filt-att)**2)), 'ci':np.percentile(att_filt,[2.5,97.5])}
    return metrics

def plot_results_interactive(metrics, att_values): # Renamed to indicate interactive
    methods, parameters, metric_names = list(metrics.keys()), ['cbf','att'], ['bias','cv','rmse']
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, param in enumerate(parameters):
        for j, metric in enumerate(metric_names):
            ax, x, width = axes[i,j], np.arange(len(att_values)), 0.25
            for k, method in enumerate(methods):
                values = [metrics[method][param][att][metric] for att in att_values]
                ax.bar(x + k*width, values, width, label=method)
            ax.set_xlabel('True ATT (ms)'); ax.set_ylabel(metric.upper()); ax.set_title(f'{param.upper()} {metric.upper()}')
            ax.set_xticks(x+width); ax.set_xticklabels(att_values); ax.legend()
    plt.tight_layout()
    plt.show() # Kept for interactive test run

def main():
    print("Running Monte Carlo simulations...")
    results, att_values = run_monte_carlo_simulation(n_iterations=1000) # n_iterations reduced for faster example
    print("\nCalculating performance metrics...")
    metrics = calculate_metrics(results, att_values)
    print("\nPerformance Summary:")
    for method in metrics.keys():
        print(f"\n{method} Results:")
        for att in att_values:
            print(f"\nATT = {att} ms:"); print("CBF Metrics:")
            for metric, value in metrics[method]['cbf'][att].items(): print(f"  {metric.upper()}: {f'[{value[0]:.1f}, {value[1]:.1f}]' if metric=='ci' else f'{value:.1f}'}")
            print("ATT Metrics:")
            for metric, value in metrics[method]['att'][att].items(): print(f"  {metric.upper()}: {f'[{value[0]:.1f}, {value[1]:.1f}]' if metric=='ci' else f'{value:.1f}'}")
    plot_results_interactive(metrics, att_values) # Changed to interactive plot

if __name__ == "__main__":
    main()
