import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import time
from scipy import stats
from dataclasses import dataclass
import json

from vsasl_functions import fit_VSASL_vectInit_pep
from pcasl_functions import fit_PCASL_vectInit_pep
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from enhanced_asl_network import EnhancedASLNet
from asl_trainer import EnhancedASLTrainer
from enhanced_simulation import RealisticASLSimulator

@dataclass
class ComparisonResult:
    """Container for comparison results"""
    method: str
    cbf_bias: float
    att_bias: float
    cbf_cov: float
    att_cov: float
    cbf_rmse: float
    att_rmse: float
    cbf_ci_width: float
    att_ci_width: float
    computation_time: float
    success_rate: float

class ComprehensiveComparison:
    """Compare neural network and least-squares fitting methods"""
    
    def __init__(self,
                 nn_model_path: Optional[str] = None,
                 output_dir: str = 'comparison_results'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize neural network model
        if nn_model_path:
            self.nn_model = self._load_nn_model(nn_model_path)
        else:
            self.nn_model = None
            
        # ASL parameters
        self.asl_params = {
            'T1_artery': 1850,
            'T2_factor': 1.0,
            'alpha_BS1': 1.0,
            'alpha_PCASL': 0.85,
            'alpha_VSASL': 0.56,
            'T_tau': 1800
        }
        
        self.results = []
        
    def _load_nn_model(self, model_path: str) -> torch.nn.Module:
        """Load trained neural network model"""
        # Create model instance
        model = EnhancedASLNet(
            input_size=12,  # 6 PLDs * 2 (PCASL + VSASL)
            hidden_sizes=[256, 128, 64],
            n_plds=6
        )
        
        # Load weights
        model.load_state_dict(torch.load(model_path))
        model.eval()
        
        return model
    
    def compare_methods(self,
                       test_data: Dict[str, np.ndarray],
                       true_params: np.ndarray,
                       plds: np.ndarray,
                       att_ranges: List[Tuple[float, float, str]]) -> pd.DataFrame:
        """Compare all methods on test data"""
        
        results = []
        
        for att_min, att_max, range_name in att_ranges:
            # Filter data by ATT range
            mask = (true_params[:, 1] >= att_min) & (true_params[:, 1] < att_max)
            range_data = {k: v[mask] for k, v in test_data.items()}
            range_params = true_params[mask]
            
            print(f"\nEvaluating {range_name} (n={mask.sum()})...")
            
            # 1. Least-squares fitting methods
            ls_results = self._evaluate_least_squares(
                range_data, range_params, plds, range_name)
            results.extend(ls_results)
            
            # 2. Neural network method
            if self.nn_model is not None:
                nn_results = self._evaluate_neural_network(
                    range_data, range_params, plds, range_name)
                results.extend(nn_results)
            
            # 3. Hybrid approach (NN for initialization, LS for refinement)
            hybrid_results = self._evaluate_hybrid(
                range_data, range_params, plds, range_name)
            results.extend(hybrid_results)
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv(self.output_dir / 'comparison_results.csv', index=False)
        
        return df
    
    def _evaluate_least_squares(self,
                              data: Dict[str, np.ndarray],
                              true_params: np.ndarray,
                              plds: np.ndarray,
                              range_name: str) -> List[ComparisonResult]:
        """Evaluate least-squares fitting methods"""
        
        results = []
        
        # MULTIVERSE fitting
        print("  Evaluating MULTIVERSE least-squares...")
        multiverse_result = self._fit_multiverse_ls(
            data['MULTIVERSE'], true_params, plds)
        multiverse_result.method = f"MULTIVERSE-LS ({range_name})"
        results.append(multiverse_result)
        
        # PCASL-only fitting
        print("  Evaluating PCASL least-squares...")
        pcasl_result = self._fit_pcasl_ls(
            data['PCASL'], true_params, plds)
        pcasl_result.method = f"PCASL-LS ({range_name})"
        results.append(pcasl_result)
        
        # VSASL-only fitting
        print("  Evaluating VSASL least-squares...")
        vsasl_result = self._fit_vsasl_ls(
            data['VSASL'], true_params, plds)
        vsasl_result.method = f"VSASL-LS ({range_name})"
        results.append(vsasl_result)
        
        return results
    
    def _fit_multiverse_ls(self,
                          signals: np.ndarray,
                          true_params: np.ndarray,
                          plds: np.ndarray) -> ComparisonResult:
        """Fit MULTIVERSE data using least-squares"""
        
        n_samples = signals.shape[0]
        cbf_estimates = []
        att_estimates = []
        ci_widths_cbf = []
        ci_widths_att = []
        fit_times = []
        successes = 0
        
        # Create PLDTI array
        pldti = np.column_stack([plds, plds])
        
        for i in range(n_samples):
            signal = signals[i]
            
            # Initial guess
            init = [50/6000, 1500]
            
            try:
                start_time = time.time()
                
                beta, conintval, rmse, df = fit_PCVSASL_misMatchPLD_vectInit_pep(
                    pldti, signal, init,
                    self.asl_params['T1_artery'],
                    self.asl_params['T_tau'],
                    self.asl_params['T2_factor'],
                    self.asl_params['alpha_BS1'],
                    self.asl_params['alpha_PCASL'],
                    self.asl_params['alpha_VSASL']
                )
                
                fit_time = time.time() - start_time
                
                cbf_estimates.append(beta[0] * 6000)
                att_estimates.append(beta[1])
                ci_widths_cbf.append((conintval[0,1] - conintval[0,0]) * 6000)
                ci_widths_att.append(conintval[1,1] - conintval[1,0])
                fit_times.append(fit_time)
                successes += 1
                
            except Exception as e:
                # Fitting failed
                cbf_estimates.append(np.nan)
                att_estimates.append(np.nan)
                ci_widths_cbf.append(np.nan)
                ci_widths_att.append(np.nan)
                fit_times.append(np.nan)
        
        # Calculate metrics
        cbf_estimates = np.array(cbf_estimates)
        att_estimates = np.array(att_estimates)
        
        valid_mask = ~np.isnan(cbf_estimates)
        
        return ComparisonResult(
            method="MULTIVERSE-LS",
            cbf_bias=np.nanmean(cbf_estimates - true_params[:, 0]),
            att_bias=np.nanmean(att_estimates - true_params[:, 1]),
            cbf_cov=np.nanstd(cbf_estimates) / np.nanmean(cbf_estimates) * 100,
            att_cov=np.nanstd(att_estimates) / np.nanmean(att_estimates) * 100,
            cbf_rmse=np.sqrt(np.nanmean((cbf_estimates - true_params[:, 0])**2)),
            att_rmse=np.sqrt(np.nanmean((att_estimates - true_params[:, 1])**2)),
            cbf_ci_width=np.nanmean(ci_widths_cbf),
            att_ci_width=np.nanmean(ci_widths_att),
            computation_time=np.nanmean(fit_times),
            success_rate=successes / n_samples * 100
        )
    
    def _fit_pcasl_ls(self,
                     signals: np.ndarray,
                     true_params: np.ndarray,
                     plds: np.ndarray) -> ComparisonResult:
        """Fit PCASL data using least-squares"""
        
        n_samples = signals.shape[0]
        cbf_estimates = []
        att_estimates = []
        fit_times = []
        successes = 0
        
        for i in range(n_samples):
            signal = signals[i]
            init = [50/6000, 1500]
            
            try:
                start_time = time.time()
                
                beta, conintval, rmse, df = fit_PCASL_vectInit_pep(
                    plds, signal, init,
                    self.asl_params['T1_artery'],
                    self.asl_params['T_tau'],
                    self.asl_params['T2_factor'],
                    self.asl_params['alpha_BS1'],
                    self.asl_params['alpha_PCASL']
                )
                
                fit_time = time.time() - start_time
                
                cbf_estimates.append(beta[0] * 6000)
                att_estimates.append(beta[1])
                fit_times.append(fit_time)
                successes += 1
                
            except:
                cbf_estimates.append(np.nan)
                att_estimates.append(np.nan)
                fit_times.append(np.nan)
        
        cbf_estimates = np.array(cbf_estimates)
        att_estimates = np.array(att_estimates)
        
        return ComparisonResult(
            method="PCASL-LS",
            cbf_bias=np.nanmean(cbf_estimates - true_params[:, 0]),
            att_bias=np.nanmean(att_estimates - true_params[:, 1]),
            cbf_cov=np.nanstd(cbf_estimates) / np.nanmean(cbf_estimates) * 100,
            att_cov=np.nanstd(att_estimates) / np.nanmean(att_estimates) * 100,
            cbf_rmse=np.sqrt(np.nanmean((cbf_estimates - true_params[:, 0])**2)),
            att_rmse=np.sqrt(np.nanmean((att_estimates - true_params[:, 1])**2)),
            cbf_ci_width=np.nan,  # Not computed for brevity
            att_ci_width=np.nan,
            computation_time=np.nanmean(fit_times),
            success_rate=successes / n_samples * 100
        )
    
    def _fit_vsasl_ls(self,
                     signals: np.ndarray,
                     true_params: np.ndarray,
                     plds: np.ndarray) -> ComparisonResult:
        """Fit VSASL data using least-squares"""
        
        n_samples = signals.shape[0]
        cbf_estimates = []
        att_estimates = []
        fit_times = []
        successes = 0
        
        for i in range(n_samples):
            signal = signals[i]
            init = [50/6000, 1500]
            
            try:
                start_time = time.time()
                
                beta, conintval, rmse, df = fit_VSASL_vectInit_pep(
                    plds, signal, init,
                    self.asl_params['T1_artery'],
                    self.asl_params['T2_factor'],
                    self.asl_params['alpha_BS1'],
                    self.asl_params['alpha_VSASL']
                )
                
                fit_time = time.time() - start_time
                
                cbf_estimates.append(beta[0] * 6000)
                att_estimates.append(beta[1])
                fit_times.append(fit_time)
                successes += 1
                
            except:
                cbf_estimates.append(np.nan)
                att_estimates.append(np.nan)
                fit_times.append(np.nan)
        
        cbf_estimates = np.array(cbf_estimates)
        att_estimates = np.array(att_estimates)
        
        return ComparisonResult(
            method="VSASL-LS",
            cbf_bias=np.nanmean(cbf_estimates - true_params[:, 0]),
            att_bias=np.nanmean(att_estimates - true_params[:, 1]),
            cbf_cov=np.nanstd(cbf_estimates) / np.nanmean(cbf_estimates) * 100,
            att_cov=np.nanstd(att_estimates) / np.nanmean(att_estimates) * 100,
            cbf_rmse=np.sqrt(np.nanmean((cbf_estimates - true_params[:, 0])**2)),
            att_rmse=np.sqrt(np.nanmean((att_estimates - true_params[:, 1])**2)),
            cbf_ci_width=np.nan,
            att_ci_width=np.nan,
            computation_time=np.nanmean(fit_times),
            success_rate=successes / n_samples * 100
        )
    
    def _evaluate_neural_network(self,
                               data: Dict[str, np.ndarray],
                               true_params: np.ndarray,
                               plds: np.ndarray,
                               range_name: str) -> List[ComparisonResult]:
        """Evaluate neural network method"""
        
        print("  Evaluating Neural Network...")
        
        # Prepare input data (concatenate PCASL and VSASL)
        nn_input = np.concatenate([data['PCASL'], data['VSASL']], axis=1)
        
        # Convert to tensor
        input_tensor = torch.FloatTensor(nn_input)
        
        # Run inference
        start_time = time.time()
        
        with torch.no_grad():
            cbf_pred, att_pred, cbf_log_var, att_log_var = self.nn_model(input_tensor)
        
        inference_time = (time.time() - start_time) / len(nn_input)
        
        # Convert to numpy
        cbf_estimates = cbf_pred.numpy().squeeze() * 6000
        att_estimates = att_pred.numpy().squeeze()
        cbf_uncertainty = np.exp(cbf_log_var.numpy().squeeze() / 2) * 6000
        att_uncertainty = np.exp(att_log_var.numpy().squeeze() / 2)
        
        # Calculate metrics
        result = ComparisonResult(
            method=f"Neural Network ({range_name})",
            cbf_bias=np.mean(cbf_estimates - true_params[:, 0]),
            att_bias=np.mean(att_estimates - true_params[:, 1]),
            cbf_cov=np.std(cbf_estimates) / np.mean(cbf_estimates) * 100,
            att_cov=np.std(att_estimates) / np.mean(att_estimates) * 100,
            cbf_rmse=np.sqrt(np.mean((cbf_estimates - true_params[:, 0])**2)),
            att_rmse=np.sqrt(np.mean((att_estimates - true_params[:, 1])**2)),
            cbf_ci_width=np.mean(cbf_uncertainty) * 1.96 * 2,  # 95% CI
            att_ci_width=np.mean(att_uncertainty) * 1.96 * 2,
            computation_time=inference_time,
            success_rate=100.0  # NN always produces output
        )
        
        return [result]
    
    def _evaluate_hybrid(self,
                        data: Dict[str, np.ndarray],
                        true_params: np.ndarray,
                        plds: np.ndarray,
                        range_name: str) -> List[ComparisonResult]:
        """Evaluate hybrid approach (NN initialization + LS refinement)"""
        
        print("  Evaluating Hybrid approach...")
        
        if self.nn_model is None:
            return []
        
        # Get NN predictions for initialization
        nn_input = np.concatenate([data['PCASL'], data['VSASL']], axis=1)
        input_tensor = torch.FloatTensor(nn_input)
        
        with torch.no_grad():
            cbf_init, att_init, _, _ = self.nn_model(input_tensor)
        
        cbf_init = cbf_init.numpy().squeeze()
        att_init = att_init.numpy().squeeze()
        
        # Use NN predictions as initialization for LS fitting
        n_samples = data['MULTIVERSE'].shape[0]
        cbf_estimates = []
        att_estimates = []
        fit_times = []
        successes = 0
        
        pldti = np.column_stack([plds, plds])
        
        for i in range(n_samples):
            signal = data['MULTIVERSE'][i]
            
            # NN-based initialization
            init = [cbf_init[i], att_init[i]]
            
            try:
                start_time = time.time()
                
                beta, conintval, rmse, df = fit_PCVSASL_misMatchPLD_vectInit_pep(
                    pldti, signal, init,
                    self.asl_params['T1_artery'],
                    self.asl_params['T_tau'],
                    self.asl_params['T2_factor'],
                    self.asl_params['alpha_BS1'],
                    self.asl_params['alpha_PCASL'],
                    self.asl_params['alpha_VSASL']
                )
                
                fit_time = time.time() - start_time
                
                cbf_estimates.append(beta[0] * 6000)
                att_estimates.append(beta[1])
                fit_times.append(fit_time)
                successes += 1
                
            except:
                # Fall back to NN estimate
                cbf_estimates.append(cbf_init[i] * 6000)
                att_estimates.append(att_init[i])
                fit_times.append(0)
                successes += 1
        
        cbf_estimates = np.array(cbf_estimates)
        att_estimates = np.array(att_estimates)
        
        result = ComparisonResult(
            method=f"Hybrid (NN+LS) ({range_name})",
            cbf_bias=np.mean(cbf_estimates - true_params[:, 0]),
            att_bias=np.mean(att_estimates - true_params[:, 1]),
            cbf_cov=np.std(cbf_estimates) / np.mean(cbf_estimates) * 100,
            att_cov=np.std(att_estimates) / np.mean(att_estimates) * 100,
            cbf_rmse=np.sqrt(np.mean((cbf_estimates - true_params[:, 0])**2)),
            att_rmse=np.sqrt(np.mean((att_estimates - true_params[:, 1])**2)),
            cbf_ci_width=np.nan,
            att_ci_width=np.nan,
            computation_time=np.mean(fit_times),
            success_rate=successes / n_samples * 100
        )
        
        return [result]
    
    def visualize_results(self, results_df: pd.DataFrame):
        """Create comprehensive visualization of comparison results"""
        
        # Set up style
        sns.set_style("whitegrid")
        
        # Create figure similar to proposal Figure 1
        fig, axes = plt.subplots(3, 2, figsize=(15, 18))
        
        # Extract ATT ranges from method names
        att_ranges = ['Short ATT', 'Medium ATT', 'Long ATT']
        colors = {'MULTIVERSE-LS': 'red', 'PCASL-LS': 'blue', 
                 'VSASL-LS': 'green', 'Neural Network': 'purple',
                 'Hybrid (NN+LS)': 'orange'}
        
        # Plot metrics
        metrics = [
            ('cbf_bias', 'att_bias', 'Bias', 'mL/100g/min', 'ms'),
            ('cbf_cov', 'att_cov', 'CoV', '%', '%'),
            ('cbf_rmse', 'att_rmse', 'RMSE', 'mL/100g/min', 'ms')
        ]
        
        for row, (cbf_metric, att_metric, metric_name, cbf_unit, att_unit) in enumerate(metrics):
            # CBF subplot
            ax_cbf = axes[row, 0]
            for method in results_df['method'].str.extract(r'(.*?)\s\(')[0].unique():
                if pd.isna(method):
                    continue
                method_data = results_df[results_df['method'].str.contains(method, na=False)]
                x_positions = [i for i, att in enumerate(att_ranges) 
                             if att in method_data['method'].values[0]]
                y_values = [method_data[method_data['method'].str.contains(att)][cbf_metric].values[0] 
                          for att in att_ranges if att in method_data['method'].str.cat()]
                
                if x_positions and y_values:
                    ax_cbf.plot(x_positions, y_values, 'o-', 
                              color=colors.get(method, 'black'),
                              label=method, linewidth=2, markersize=8)
            
            ax_cbf.set_ylabel(f'{metric_name} ({cbf_unit})')
            ax_cbf.set_title(f'CBF {metric_name}')
            ax_cbf.set_xticks(range(len(att_ranges)))
            ax_cbf.set_xticklabels(att_ranges)
            if row == 0:
                ax_cbf.legend()
            
            # ATT subplot
            ax_att = axes[row, 1]
            for method in results_df['method'].str.extract(r'(.*?)\s\(')[0].unique():
                if pd.isna(method):
                    continue
                method_data = results_df[results_df['method'].str.contains(method, na=False)]
                x_positions = [i for i, att in enumerate(att_ranges) 
                             if att in method_data['method'].values[0]]
                y_values = [method_data[method_data['method'].str.contains(att)][att_metric].values[0] 
                          for att in att_ranges if att in method_data['method'].str.cat()]
                
                if x_positions and y_values:
                    ax_att.plot(x_positions, y_values, 'o-', 
                              color=colors.get(method, 'black'),
                              label=method, linewidth=2, markersize=8)
            
            ax_att.set_ylabel(f'{metric_name} ({att_unit})')
            ax_att.set_title(f'ATT {metric_name}')
            ax_att.set_xticks(range(len(att_ranges)))
            ax_att.set_xticklabels(att_ranges)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'comparison_figure1_style.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Additional plots
        self._plot_computation_time(results_df)
        self._plot_success_rates(results_df)
        
    def _plot_computation_time(self, results_df: pd.DataFrame):
        """Plot computation time comparison"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Average computation time per method
        avg_times = results_df.groupby(
            results_df['method'].str.extract(r'(.*?)\s\(')[0]
        )['computation_time'].mean()
        
        bars = ax.bar(avg_times.index, avg_times.values * 1000)  # Convert to ms
        ax.set_ylabel('Computation Time (ms)')
        ax.set_title('Average Computation Time per Sample')
        ax.set_yscale('log')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'computation_time_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_success_rates(self, results_df: pd.DataFrame):
        """Plot fitting success rates"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Success rate per method and ATT range
        pivot_data = results_df.pivot_table(
            values='success_rate',
            index=results_df['method'].str.extract(r'\((.*?)\)')[0],
            columns=results_df['method'].str.extract(r'(.*?)\s\(')[0],
            aggfunc='first'
        )
        
        pivot_data.plot(kind='bar', ax=ax)
        ax.set_ylabel('Success Rate (%)')
        ax.set_xlabel('ATT Range')
        ax.set_title('Fitting Success Rates')
        ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rates.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()


def run_comprehensive_comparison():
    """Run the full comparison analysis"""
    
    print("Running comprehensive comparison of NN vs Least-Squares methods...")
    
    # Initialize simulator
    simulator = RealisticASLSimulator()
    plds = np.arange(500, 3001, 500)
    
    # Generate test data
    print("Generating test data...")
    test_data = simulator.generate_diverse_dataset(
        plds=plds,
        n_subjects=50,
        conditions=['healthy', 'stroke', 'elderly'],
        noise_levels=[5.0]
    )
    
    # Reshape data for analysis
    signals = {
        'PCASL': test_data['signals'][:, :6],
        'VSASL': test_data['signals'][:, 6:],
        'MULTIVERSE': test_data['signals'].reshape(-1, 6, 2)
    }
    
    # Initialize comparison framework
    comparator = ComprehensiveComparison()
    
    # Define ATT ranges
    att_ranges = [
        (500, 1500, "Short ATT"),
        (1500, 2500, "Medium ATT"),
        (2500, 4000, "Long ATT")
    ]
    
    # Run comparison
    results_df = comparator.compare_methods(
        signals,
        test_data['parameters'],
        plds,
        att_ranges
    )
    
    # Visualize results
    print("\nGenerating visualizations...")
    comparator.visualize_results(results_df)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("=" * 80)
    print(results_df.groupby(
        results_df['method'].str.extract(r'(.*?)\s\(')[0]
    )[['cbf_rmse', 'att_rmse', 'computation_time']].mean())
    
    return results_df


if __name__ == "__main__":
    # Run the comparison
    results = run_comprehensive_comparison()
    
    # Additional analysis
    print("\nDetailed Results:")
    print(results.to_string())