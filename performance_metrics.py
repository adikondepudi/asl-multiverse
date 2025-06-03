# performance_metrics.py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class ProposalMetrics:
    """Metrics specifically mentioned in the research proposal"""
    normalized_bias_cbf: float
    normalized_bias_att: float
    coefficient_variation_cbf: float
    coefficient_variation_att: float
    normalized_rmse_cbf: float
    normalized_rmse_att: float
    snr_improvement: float
    scan_time_reduction: float

class ProposalEvaluator:
    """Evaluation framework specifically for proposal objectives"""
    
    def __init__(self):
        self.att_ranges = [
            (500, 1500, "Short ATT"),
            (1500, 2500, "Medium ATT"), 
            (2500, 4000, "Long ATT")
        ]
    
    def calculate_proposal_metrics(self, 
                                 true_cbf: np.ndarray,
                                 pred_cbf: np.ndarray,
                                 true_att: np.ndarray,
                                 pred_att: np.ndarray,
                                 method_name: str) -> ProposalMetrics:
        """Calculate metrics exactly as defined in proposal"""
        
        # Normalized bias (as percentage)
        nbias_cbf = np.mean((pred_cbf - true_cbf) / true_cbf) * 100
        nbias_att = np.mean((pred_att - true_att) / true_att) * 100
        
        # Coefficient of variation
        cov_cbf = np.std(pred_cbf) / np.mean(pred_cbf) * 100
        cov_att = np.std(pred_att) / np.mean(pred_att) * 100
        
        # Normalized RMSE
        nrmse_cbf = np.sqrt(np.mean((pred_cbf - true_cbf)**2)) / np.mean(true_cbf) * 100
        nrmse_att = np.sqrt(np.mean((pred_att - true_att)**2)) / np.mean(true_att) * 100
        
        # SNR improvement (to be calculated relative to baseline)
        snr_improvement = 0.0  # Placeholder
        
        # Scan time reduction (based on reduced averaging needs)
        scan_time_reduction = 0.0  # Placeholder
        
        return ProposalMetrics(
            normalized_bias_cbf=nbias_cbf,
            normalized_bias_att=nbias_att,
            coefficient_variation_cbf=cov_cbf,
            coefficient_variation_att=cov_att,
            normalized_rmse_cbf=nrmse_cbf,
            normalized_rmse_att=nrmse_att,
            snr_improvement=snr_improvement,
            scan_time_reduction=scan_time_reduction
        )
    
    def create_proposal_figure1(self, results_dict: Dict[str, Dict]) -> plt.Figure:
        """Recreate Figure 1 from proposal with neural network results"""
        
        fig, axes = plt.subplots(3, 2, figsize=(12, 15))
        
        # Define ATT values for x-axis
        att_values = np.arange(500, 4000, 100)
        
        metrics = ['normalized_bias', 'coefficient_variation', 'normalized_rmse']
        metric_labels = ['nBias (%)', 'CoV (%)', 'nRMSE (%)']
        
        for row, (metric, label) in enumerate(zip(metrics, metric_labels)):
            # CBF subplot
            ax_cbf = axes[row, 0]
            ax_cbf.plot(att_values, results_dict['PCASL'][f'{metric}_cbf'], 
                       'b-', label='PCASL', linewidth=2)
            ax_cbf.plot(att_values, results_dict['VSASL'][f'{metric}_cbf'], 
                       'g-', label='VSASL', linewidth=2)
            ax_cbf.plot(att_values, results_dict['MULTIVERSE_LS'][f'{metric}_cbf'], 
                       'r-', label='MULTIVERSE LS', linewidth=2)
            ax_cbf.plot(att_values, results_dict['MULTIVERSE_NN'][f'{metric}_cbf'], 
                       'r--', label='MULTIVERSE NN', linewidth=3)
            
            ax_cbf.set_xlabel('Arterial Transit Time (ms)')
            ax_cbf.set_ylabel(f'CBF {label}')
            ax_cbf.grid(True, alpha=0.3)
            ax_cbf.legend()
            
            # ATT subplot
            ax_att = axes[row, 1]
            ax_att.plot(att_values, results_dict['PCASL'][f'{metric}_att'], 
                       'b-', label='PCASL', linewidth=2)
            ax_att.plot(att_values, results_dict['VSASL'][f'{metric}_att'], 
                       'g-', label='VSASL', linewidth=2)
            ax_att.plot(att_values, results_dict['MULTIVERSE_LS'][f'{metric}_att'], 
                       'r-', label='MULTIVERSE LS', linewidth=2)
            ax_att.plot(att_values, results_dict['MULTIVERSE_NN'][f'{metric}_att'], 
                       'r--', label='MULTIVERSE NN', linewidth=3)
            
            ax_att.set_xlabel('Arterial Transit Time (ms)')
            ax_att.set_ylabel(f'ATT {label}')
            ax_att.grid(True, alpha=0.3)
            ax_att.legend()
        
        plt.tight_layout()
        return fig
    
    def benchmark_against_proposal_goals(self, neural_net_results: Dict, 
                                       baseline_results: Dict) -> Dict:
        """Benchmark neural network performance against proposal goals"""
        
        improvements = {}
        
        # Key proposal goals:
        # 1. Maintain accuracy while reducing scan time
        # 2. Improve SNR significantly
        # 3. Outperform conventional least-squares fitting
        
        for att_range_name in ["Short ATT", "Medium ATT", "Long ATT"]:
            nn_metrics = neural_net_results[att_range_name]
            baseline_metrics = baseline_results[att_range_name]
            
            improvements[att_range_name] = {
                'bias_improvement_cbf': baseline_metrics.normalized_bias_cbf - nn_metrics.normalized_bias_cbf,
                'bias_improvement_att': baseline_metrics.normalized_bias_att - nn_metrics.normalized_bias_att,
                'precision_improvement_cbf': baseline_metrics.coefficient_variation_cbf - nn_metrics.coefficient_variation_cbf,
                'precision_improvement_att': baseline_metrics.coefficient_variation_att - nn_metrics.coefficient_variation_att,
                'rmse_improvement_cbf': baseline_metrics.normalized_rmse_cbf - nn_metrics.normalized_rmse_cbf,
                'rmse_improvement_att': baseline_metrics.normalized_rmse_att - nn_metrics.normalized_rmse_att
            }
        
        return improvements

def run_proposal_validation():
    """Main function to run validation according to proposal objectives"""
    
    evaluator = ProposalEvaluator()
    
    # This would use your trained models and test data
    # Placeholder for actual implementation
    print("Running proposal validation...")
    print("1. Loading trained neural network models...")
    print("2. Loading high-quality validation datasets...")
    print("3. Comparing NN vs least-squares performance...")
    print("4. Generating proposal figures...")
    
    return evaluator

if __name__ == "__main__":
    run_proposal_validation()