import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import time
from scipy import stats
from dataclasses import dataclass, asdict # Added asdict
import json

from vsasl_functions import fit_VSASL_vectInit_pep
from pcasl_functions import fit_PCASL_vectInit_pep
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from enhanced_asl_network import EnhancedASLNet
# from asl_trainer import EnhancedASLTrainer # Not directly used here
# from enhanced_simulation import RealisticASLSimulator # Not directly used here

# For logging within this file
import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


@dataclass
class ComparisonResult:
    """Container for comparison results"""
    method: str
    att_range_name: str # Added to distinguish results
    cbf_bias: float
    att_bias: float
    cbf_cov: float
    att_cov: float
    cbf_rmse: float
    att_rmse: float
    # Normalized metrics
    cbf_nbias_perc: float # Normalized bias as percentage
    att_nbias_perc: float
    cbf_nrmse_perc: float # Normalized RMSE as percentage
    att_nrmse_perc: float
    # CI and success
    cbf_ci_width: float
    att_ci_width: float
    computation_time: float # Per sample
    success_rate: float


class ComprehensiveComparison:
    """Compare neural network and least-squares fitting methods"""

    def __init__(self,
                 nn_model_path: Optional[str] = None,
                 output_dir: str = 'comparison_results',
                 # NN model config (should come from main config)
                 nn_input_size: int = 12,
                 nn_hidden_sizes: Optional[List[int]] = None,
                 nn_n_plds: int = 6
                 ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.nn_input_size = nn_input_size
        self.nn_hidden_sizes = nn_hidden_sizes if nn_hidden_sizes is not None else [256, 128, 64]
        self.nn_n_plds = nn_n_plds

        # Initialize neural network model
        if nn_model_path and Path(nn_model_path).exists():
            self.nn_model = self._load_nn_model(nn_model_path)
            logger.info(f"Loaded NN model from {nn_model_path}")
        elif nn_model_path:
            logger.warning(f"NN model path {nn_model_path} does not exist. NN evaluation will be skipped.")
            self.nn_model = None
        else:
            self.nn_model = None
            logger.info("No NN model path provided. NN evaluation will be skipped.")


        # ASL parameters (can be overridden if needed)
        self.asl_params = {
            'T1_artery': 1850.0,
            'T2_factor': 1.0,
            'alpha_BS1': 1.0,
            'alpha_PCASL': 0.85,
            'alpha_VSASL': 0.56,
            'T_tau': 1800.0
        }

        self.results_list = [] # Changed from self.results to avoid conflict

    def _load_nn_model(self, model_path: str) -> torch.nn.Module:
        """Load trained neural network model"""
        model = EnhancedASLNet(
            input_size=self.nn_input_size,
            hidden_sizes=self.nn_hidden_sizes,
            n_plds=self.nn_n_plds
        )
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Ensure loadable if not on CUDA
        model.eval()
        return model

    def _calculate_detailed_metrics(self, pred_cbf, true_cbf, pred_att, true_att):
        """Helper to calculate bias, cov, rmse, and normalized versions."""
        # Ensure inputs are numpy arrays
        pred_cbf, true_cbf, pred_att, true_att = map(np.asarray, [pred_cbf, true_cbf, pred_att, true_att])

        # Filter NaNs from predictions for metric calculation
        valid_mask_cbf = ~np.isnan(pred_cbf) & ~np.isnan(true_cbf)
        valid_mask_att = ~np.isnan(pred_att) & ~np.isnan(true_att)

        metrics = {
            'cbf_bias': np.nan, 'att_bias': np.nan,
            'cbf_cov': np.nan, 'att_cov': np.nan,
            'cbf_rmse': np.nan, 'att_rmse': np.nan,
            'cbf_nbias_perc': np.nan, 'att_nbias_perc': np.nan,
            'cbf_nrmse_perc': np.nan, 'att_nrmse_perc': np.nan,
        }

        if np.sum(valid_mask_cbf) > 0:
            vc_pred_cbf, vc_true_cbf = pred_cbf[valid_mask_cbf], true_cbf[valid_mask_cbf]
            metrics['cbf_bias'] = np.mean(vc_pred_cbf - vc_true_cbf)
            mean_true_cbf = np.mean(vc_true_cbf)
            if mean_true_cbf != 0:
                 metrics['cbf_nbias_perc'] = (metrics['cbf_bias'] / mean_true_cbf) * 100
            if np.mean(vc_pred_cbf) != 0:
                 metrics['cbf_cov'] = (np.std(vc_pred_cbf) / np.mean(vc_pred_cbf)) * 100
            metrics['cbf_rmse'] = np.sqrt(np.mean((vc_pred_cbf - vc_true_cbf)**2))
            if mean_true_cbf != 0:
                 metrics['cbf_nrmse_perc'] = (metrics['cbf_rmse'] / mean_true_cbf) * 100
        
        if np.sum(valid_mask_att) > 0:
            va_pred_att, va_true_att = pred_att[valid_mask_att], true_att[valid_mask_att]
            metrics['att_bias'] = np.mean(va_pred_att - va_true_att)
            mean_true_att = np.mean(va_true_att)
            if mean_true_att != 0:
                 metrics['att_nbias_perc'] = (metrics['att_bias'] / mean_true_att) * 100
            if np.mean(va_pred_att) != 0:
                metrics['att_cov'] = (np.std(va_pred_att) / np.mean(va_pred_att)) * 100
            metrics['att_rmse'] = np.sqrt(np.mean((va_pred_att - va_true_att)**2))
            if mean_true_att != 0:
                metrics['att_nrmse_perc'] = (metrics['att_rmse'] / mean_true_att) * 100

        return metrics


    def compare_methods(self,
                       test_data_dict: Dict[str, np.ndarray], # Keys: 'PCASL', 'VSASL', 'MULTIVERSE' (N, n_plds*2 or N, n_plds, 2)
                       true_params_arr: np.ndarray, # (N, 2) with [CBF_ml/100g/min, ATT_ms]
                       plds_arr: np.ndarray,
                       att_ranges_list: List[Tuple[float, float, str]]) -> pd.DataFrame:
        """Compare all methods on test data"""
        self.results_list = [] # Clear previous results

        for att_min, att_max, range_name_str in att_ranges_list:
            # Filter data by ATT range
            # true_params_arr[:, 1] is ATT
            mask = (true_params_arr[:, 1] >= att_min) & (true_params_arr[:, 1] < att_max)
            
            if not np.any(mask):
                logger.warning(f"No test data for ATT range {range_name_str}. Skipping.")
                continue

            range_true_params = true_params_arr[mask]
            
            # Filter signals for each method
            range_data_signals = {}
            # Assuming test_data_dict['MULTIVERSE'] is (N, n_plds, 2) for LS fitting and (N, n_plds*2) for NN
            # And test_data_dict['PCASL'] / ['VSASL'] are (N, n_plds)
            
            # For LS methods that take separate PCASL/VSASL or combined (N, n_plds, 2)
            range_data_signals['PCASL_LS'] = test_data_dict['PCASL'][mask]
            range_data_signals['VSASL_LS'] = test_data_dict['VSASL'][mask]
            range_data_signals['MULTIVERSE_LS'] = test_data_dict['MULTIVERSE_LS_FORMAT'][mask] # Expects (N, n_plds, 2)

            # For NN and Hybrid that take combined (N, n_plds*2)
            range_data_signals['NN_INPUT'] = test_data_dict['NN_INPUT_FORMAT'][mask] # Expects (N, n_plds*2)


            logger.info(f"\nEvaluating {range_name_str} (n={mask.sum()})...")

            # 1. Least-squares fitting methods
            ls_eval_results = self._evaluate_least_squares(
                range_data_signals, range_true_params, plds_arr, range_name_str)
            self.results_list.extend(ls_eval_results)

            # 2. Neural network method
            if self.nn_model is not None:
                nn_eval_results = self._evaluate_neural_network(
                    range_data_signals['NN_INPUT'], range_true_params, plds_arr, range_name_str)
                self.results_list.extend(nn_eval_results)
            else:
                logger.info("Skipping Neural Network evaluation as model is not loaded.")


            # 3. Hybrid approach (NN for initialization, LS for refinement)
            if self.nn_model is not None:
                hybrid_eval_results = self._evaluate_hybrid(
                    range_data_signals['MULTIVERSE_LS'], # Uses LS format for MULTIVERSE signal
                    range_data_signals['NN_INPUT'],      # Uses NN format for NN init
                    range_true_params, plds_arr, range_name_str)
                self.results_list.extend(hybrid_eval_results)
            else:
                logger.info("Skipping Hybrid evaluation as model is not loaded.")


        # Convert to DataFrame
        if not self.results_list:
            logger.warning("No results generated from comparison. Returning empty DataFrame.")
            return pd.DataFrame()

        # Convert list of dataclass objects to DataFrame
        df_data = [asdict(r) for r in self.results_list]
        df = pd.DataFrame(df_data)
        df.to_csv(self.output_dir / 'comparison_results_detailed.csv', index=False)
        return df

    def _evaluate_least_squares(self,
                              data_signals: Dict[str, np.ndarray], # Contains 'PCASL_LS', 'VSASL_LS', 'MULTIVERSE_LS'
                              true_params: np.ndarray,
                              plds: np.ndarray,
                              range_name: str) -> List[ComparisonResult]:
        """Evaluate least-squares fitting methods"""
        ls_results_list = []

        logger.info("  Evaluating MULTIVERSE least-squares...")
        multiverse_res = self._fit_multiverse_ls(
            data_signals['MULTIVERSE_LS'], true_params, plds, range_name)
        if multiverse_res: ls_results_list.append(multiverse_res)

        logger.info("  Evaluating PCASL least-squares...")
        pcasl_res = self._fit_pcasl_ls(
            data_signals['PCASL_LS'], true_params, plds, range_name)
        if pcasl_res: ls_results_list.append(pcasl_res)

        logger.info("  Evaluating VSASL least-squares...")
        vsasl_res = self._fit_vsasl_ls(
            data_signals['VSASL_LS'], true_params, plds, range_name)
        if vsasl_res: ls_results_list.append(vsasl_res)
        
        return ls_results_list

    def _fit_multiverse_ls(self,
                          signals_arr: np.ndarray, # Expected shape (N, n_plds, 2)
                          true_params_arr: np.ndarray,
                          plds_arr: np.ndarray,
                          range_name_str: str) -> Optional[ComparisonResult]:
        """Fit MULTIVERSE data using least-squares"""
        n_samples = signals_arr.shape[0]
        if n_samples == 0: return None

        cbf_estimates, att_estimates = [], []
        ci_widths_cbf, ci_widths_att = [], []
        fit_times = []
        successes = 0
        pldti = np.column_stack([plds_arr, plds_arr])

        for i in range(n_samples):
            signal_sample = signals_arr[i] # Shape (n_plds, 2)
            # Use true params for a good initial guess for LS, or a fixed robust one
            # init_cbf = true_params_arr[i, 0] / 6000.0
            # init_att = true_params_arr[i, 1]
            init_cbf, init_att = 50.0 / 6000.0, 1500.0 # Fixed robust init
            
            start_time = time.time()
            try:
                beta, conintval, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                    pldti, signal_sample, [init_cbf, init_att],
                    self.asl_params['T1_artery'], self.asl_params['T_tau'],
                    self.asl_params['T2_factor'], self.asl_params['alpha_BS1'],
                    self.asl_params['alpha_PCASL'], self.asl_params['alpha_VSASL']
                )
                fit_times.append(time.time() - start_time)
                cbf_estimates.append(beta[0] * 6000) # Convert to ml/100g/min
                att_estimates.append(beta[1])
                ci_widths_cbf.append((conintval[0,1] - conintval[0,0]) * 6000)
                ci_widths_att.append(conintval[1,1] - conintval[1,0])
                successes += 1
            except Exception:
                fit_times.append(time.time() - start_time) # Still record time for attempt
                cbf_estimates.append(np.nan); att_estimates.append(np.nan)
                ci_widths_cbf.append(np.nan); ci_widths_att.append(np.nan)
        
        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:,0], att_estimates, true_params_arr[:,1])
        return ComparisonResult(
            method="MULTIVERSE-LS", att_range_name=range_name_str,
            **metrics, # Unpack calculated metrics
            cbf_ci_width=np.nanmean(ci_widths_cbf), att_ci_width=np.nanmean(ci_widths_att),
            computation_time=np.nanmean(fit_times) if fit_times else np.nan,
            success_rate=(successes / n_samples * 100) if n_samples > 0 else 0
        )

    def _fit_pcasl_ls(self, signals_arr: np.ndarray, true_params_arr: np.ndarray, plds_arr: np.ndarray, range_name_str: str) -> Optional[ComparisonResult]:
        n_samples = signals_arr.shape[0]
        if n_samples == 0: return None
        cbf_estimates, att_estimates, fit_times = [], [], []
        successes = 0
        for i in range(n_samples):
            init_cbf, init_att = 50.0 / 6000.0, 1500.0
            start_time = time.time()
            try:
                beta, _, _, _ = fit_PCASL_vectInit_pep(
                    plds_arr, signals_arr[i], [init_cbf, init_att],
                    self.asl_params['T1_artery'], self.asl_params['T_tau'],
                    self.asl_params['T2_factor'], self.asl_params['alpha_BS1'], self.asl_params['alpha_PCASL']
                )
                fit_times.append(time.time() - start_time)
                cbf_estimates.append(beta[0] * 6000); att_estimates.append(beta[1])
                successes += 1
            except Exception:
                fit_times.append(time.time() - start_time)
                cbf_estimates.append(np.nan); att_estimates.append(np.nan)
        
        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:,0], att_estimates, true_params_arr[:,1])
        return ComparisonResult(
            method="PCASL-LS", att_range_name=range_name_str, **metrics,
            cbf_ci_width=np.nan, att_ci_width=np.nan, # Not typically calculated for individual LS
            computation_time=np.nanmean(fit_times) if fit_times else np.nan,
            success_rate=(successes / n_samples * 100) if n_samples > 0 else 0
        )

    def _fit_vsasl_ls(self, signals_arr: np.ndarray, true_params_arr: np.ndarray, plds_arr: np.ndarray, range_name_str: str) -> Optional[ComparisonResult]:
        n_samples = signals_arr.shape[0]
        if n_samples == 0: return None
        cbf_estimates, att_estimates, fit_times = [], [], []
        successes = 0
        for i in range(n_samples):
            init_cbf, init_att = 50.0 / 6000.0, 1500.0
            start_time = time.time()
            try:
                beta, _, _, _ = fit_VSASL_vectInit_pep(
                    plds_arr, signals_arr[i], [init_cbf, init_att],
                    self.asl_params['T1_artery'], self.asl_params['T2_factor'],
                    self.asl_params['alpha_BS1'], self.asl_params['alpha_VSASL']
                )
                fit_times.append(time.time() - start_time)
                cbf_estimates.append(beta[0] * 6000); att_estimates.append(beta[1])
                successes += 1
            except Exception:
                fit_times.append(time.time() - start_time)
                cbf_estimates.append(np.nan); att_estimates.append(np.nan)

        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:,0], att_estimates, true_params_arr[:,1])
        return ComparisonResult(
            method="VSASL-LS", att_range_name=range_name_str, **metrics,
            cbf_ci_width=np.nan, att_ci_width=np.nan,
            computation_time=np.nanmean(fit_times) if fit_times else np.nan,
            success_rate=(successes / n_samples * 100) if n_samples > 0 else 0
        )

    def _evaluate_neural_network(self,
                               nn_input_arr: np.ndarray, # Expected shape (N, n_plds*2)
                               true_params_arr: np.ndarray,
                               plds_arr: np.ndarray, # Unused here, but kept for signature consistency
                               range_name_str: str) -> List[ComparisonResult]:
        """Evaluate neural network method"""
        if self.nn_model is None or nn_input_arr.shape[0] == 0:
            return []
        logger.info("  Evaluating Neural Network...")
        input_tensor = torch.FloatTensor(nn_input_arr)
        start_time = time.time()
        with torch.no_grad():
            # NN output is CBF (ml/100g/min), ATT (ms)
            cbf_pred, att_pred, cbf_log_var, att_log_var = self.nn_model(input_tensor)
        inference_time_total = time.time() - start_time
        
        cbf_estimates = cbf_pred.numpy().squeeze() # Already in ml/100g/min
        att_estimates = att_pred.numpy().squeeze()
        # Uncertainty (std dev)
        cbf_uncertainty_std = np.exp(cbf_log_var.numpy().squeeze() / 2.0)
        att_uncertainty_std = np.exp(att_log_var.numpy().squeeze() / 2.0)

        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:,0], att_estimates, true_params_arr[:,1])
        return [ComparisonResult(
            method="Neural Network", att_range_name=range_name_str, **metrics,
            cbf_ci_width=np.mean(cbf_uncertainty_std) * 1.96 * 2, # 95% CI width from std
            att_ci_width=np.mean(att_uncertainty_std) * 1.96 * 2,
            computation_time=inference_time_total / len(nn_input_arr) if len(nn_input_arr) > 0 else np.nan,
            success_rate=100.0
        )]

    def _evaluate_hybrid(self,
                        multiverse_ls_signals: np.ndarray, # (N, n_plds, 2)
                        nn_input_signals: np.ndarray,      # (N, n_plds*2)
                        true_params_arr: np.ndarray,
                        plds_arr: np.ndarray,
                        range_name_str: str) -> List[ComparisonResult]:
        """Evaluate hybrid approach (NN initialization + LS refinement)"""
        if self.nn_model is None or nn_input_signals.shape[0] == 0:
            return []
        logger.info("  Evaluating Hybrid approach...")
        
        input_tensor = torch.FloatTensor(nn_input_signals)
        with torch.no_grad():
            # NN output is CBF (ml/100g/min), ATT (ms)
            cbf_init_nn, att_init_nn, _, _ = self.nn_model(input_tensor)
        
        # Convert NN output for LS init: CBF to ml/g/s
        cbf_init_ls = cbf_init_nn.numpy().squeeze() / 6000.0
        att_init_ls = att_init_nn.numpy().squeeze()
        
        n_samples = multiverse_ls_signals.shape[0]
        cbf_estimates, att_estimates, fit_times = [], [], []
        successes = 0
        pldti = np.column_stack([plds_arr, plds_arr])

        for i in range(n_samples):
            signal_sample_ls = multiverse_ls_signals[i] # For LS fitting
            init_for_ls = [cbf_init_ls[i], att_init_ls[i]]
            start_time = time.time()
            try:
                beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                    pldti, signal_sample_ls, init_for_ls,
                    self.asl_params['T1_artery'], self.asl_params['T_tau'],
                    self.asl_params['T2_factor'], self.asl_params['alpha_BS1'],
                    self.asl_params['alpha_PCASL'], self.asl_params['alpha_VSASL']
                )
                fit_times.append(time.time() - start_time)
                cbf_estimates.append(beta[0] * 6000) # Convert to ml/100g/min
                att_estimates.append(beta[1])
                successes += 1
            except Exception: # If LS refinement fails, use NN prediction
                fit_times.append(time.time() - start_time) # time for failed attempt + NN pred (already done)
                cbf_estimates.append(cbf_init_nn.numpy().squeeze()[i]) # Use NN's ml/100g/min prediction
                att_estimates.append(att_init_ls[i]) # ATT is already in ms
                # Consider if a failed LS means a non-success for hybrid, or if NN fallback is a "success"
                # For now, counting NN fallback as a success for the hybrid method
                successes +=1 
        
        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:,0], att_estimates, true_params_arr[:,1])
        return [ComparisonResult(
            method="Hybrid (NN+LS)", att_range_name=range_name_str, **metrics,
            cbf_ci_width=np.nan, att_ci_width=np.nan, # CIs would be from LS step if successful
            computation_time=np.nanmean(fit_times) if fit_times else np.nan,
            success_rate=(successes / n_samples * 100) if n_samples > 0 else 0
        )]

    def visualize_results(self, results_df: pd.DataFrame):
        """Create comprehensive visualization of comparison results"""
        if results_df.empty:
            logger.warning("Results DataFrame is empty. Skipping visualization.")
            return

        sns.set_style("whitegrid")
        fig, axes = plt.subplots(3, 2, figsize=(16, 20)) # Increased size slightly

        # Define a consistent order and colors for methods
        all_method_names = results_df['method'].unique()
        method_order = [m for m in ['PCASL-LS', 'VSASL-LS', 'MULTIVERSE-LS', 'Neural Network', 'Hybrid (NN+LS)'] if m in all_method_names]
        
        colors = {'PCASL-LS': 'blue', 'VSASL-LS': 'green', 
                  'MULTIVERSE-LS': 'red', 'Neural Network': 'purple',
                  'Hybrid (NN+LS)': 'orange', 'Other': 'grey'} # Fallback color

        # Metrics to plot (using the normalized versions)
        # Assuming 'cbf_nbias_perc', 'att_nbias_perc', 'cbf_cov', 'att_cov', 'cbf_nrmse_perc', 'att_nrmse_perc' exist in df
        metrics_to_plot = [
            ('cbf_nbias_perc', 'att_nbias_perc', 'Normalized Bias', '%', '%'),
            ('cbf_cov', 'att_cov', 'Coefficient of Variation', '%', '%'),
            ('cbf_nrmse_perc', 'att_nrmse_perc', 'Normalized RMSE', '%', '%')
        ]
        
        # Get unique ATT range names from the 'att_range_name' column
        att_range_names_sorted = sorted(results_df['att_range_name'].unique(), key=lambda x: (x.split(' ')[0] != "Short", x.split(' ')[0] != "Medium", x.split(' ')[0] != "Long", x))


        for row, (cbf_metric_key, att_metric_key, metric_disp_name, cbf_unit, att_unit) in enumerate(metrics_to_plot):
            ax_cbf = axes[row, 0]
            ax_att = axes[row, 1]

            for method_name in method_order:
                method_specific_df = results_df[results_df['method'] == method_name]
                
                # Ensure data is sorted by ATT range for consistent plotting
                method_specific_df = method_specific_df.set_index('att_range_name').reindex(att_range_names_sorted).reset_index()

                cbf_values = method_specific_df[cbf_metric_key]
                att_values = method_specific_df[att_metric_key]
                
                x_positions = range(len(att_range_names_sorted))

                ax_cbf.plot(x_positions, cbf_values, 'o-', 
                            color=colors.get(method_name, colors['Other']),
                            label=method_name, linewidth=2.5 if 'Neural' in method_name or 'Hybrid' in method_name else 2, 
                            markersize=7)
                ax_att.plot(x_positions, att_values, 'o-', 
                            color=colors.get(method_name, colors['Other']),
                            label=method_name, linewidth=2.5 if 'Neural' in method_name or 'Hybrid' in method_name else 2,
                            markersize=7)

            ax_cbf.set_ylabel(f'CBF {metric_disp_name} ({cbf_unit})')
            ax_cbf.set_title(f'CBF {metric_disp_name}')
            ax_cbf.set_xticks(x_positions)
            ax_cbf.set_xticklabels(att_range_names_sorted, rotation=30, ha='right')
            if row == 0: ax_cbf.legend(fontsize='small')

            ax_att.set_ylabel(f'ATT {metric_disp_name} ({att_unit})')
            ax_att.set_title(f'ATT {metric_disp_name}')
            ax_att.set_xticks(x_positions)
            ax_att.set_xticklabels(att_range_names_sorted, rotation=30, ha='right')
            if row == 0: ax_att.legend(fontsize='small')

            if 'Bias' in metric_disp_name:
                ax_cbf.axhline(0, color='k', linestyle='--', alpha=0.7)
                ax_att.axhline(0, color='k', linestyle='--', alpha=0.7)

        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to prevent title overlap
        fig.suptitle("Performance Comparison Across Methods and ATT Ranges", fontsize=16, fontweight='bold')
        plt.savefig(self.output_dir / 'comparison_figure1_style_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

        self._plot_computation_time(results_df)
        self._plot_success_rates(results_df)

    def _plot_computation_time(self, results_df: pd.DataFrame):
        if results_df.empty or 'computation_time' not in results_df.columns:
            logger.warning("Cannot plot computation time: DataFrame empty or 'computation_time' missing.")
            return
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Group by method and ATT range, then average computation time
        # Or just average over all ATT ranges for a simpler plot
        avg_times = results_df.groupby('method')['computation_time'].mean().sort_values()
        
        bars = ax.bar(avg_times.index, avg_times.values * 1000) # Convert to ms
        ax.set_ylabel('Avg. Computation Time per Sample (ms)')
        ax.set_title('Average Computation Time Comparison')
        ax.set_yscale('log')
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'computation_time_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close(fig)

    def _plot_success_rates(self, results_df: pd.DataFrame):
        if results_df.empty or 'success_rate' not in results_df.columns:
            logger.warning("Cannot plot success rates: DataFrame empty or 'success_rate' missing.")
            return
            
        # Use pivot_table for better handling of multi-index if methods are combined with att_range_name
        try:
            pivot_data = results_df.pivot_table(
                values='success_rate',
                index='att_range_name', # Rows are ATT ranges
                columns='method',       # Columns are methods
                aggfunc='mean' # Use mean if multiple entries, though should be one per combo
            ).reindex(sorted(results_df['att_range_name'].unique())) # Sort rows
            
            if pivot_data.empty:
                logger.warning("Pivot table for success rates is empty.")
                return

            pivot_data.plot(kind='bar', ax=plt.gca(), figsize=(12,7), width=0.8) # plt.gca() to get current axis for a new figure
            plt.ylabel('Success Rate (%)')
            plt.xlabel('ATT Range')
            plt.title('Fitting Success Rates by Method and ATT Range')
            plt.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left')
            plt.xticks(rotation=30, ha='right')
            plt.ylim(0, 105)
            plt.tight_layout()
            plt.savefig(self.output_dir / 'success_rates_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            plt.close()

        except Exception as e:
            logger.error(f"Error plotting success rates: {e}")


# Example usage (simplified, typically called from main.py or test_all.py)
# if __name__ == "__main__":
#     # This would require setting up dummy data and model path
#     # For a full run, main.py's run_comprehensive_asl_research is the entry point
#     logger.info("Running ComprehensiveComparison example (illustrative)")
#     # comparator = ComprehensiveComparison(nn_model_path=None) # No model for this dummy run
#     # plds = np.arange(500,3001,500)
#     # att_ranges_config = [
#     #     (500, 1500, "Short ATT"), (1500, 2500, "Medium ATT"), (2500, 4000, "Long ATT")
#     # ]
#     # dummy_true_params = np.array([[60, 1000], [50, 2000], [40, 3000]])
#     # dummy_data_shape_ls = (dummy_true_params.shape[0], len(plds), 2)
#     # dummy_data_shape_nn = (dummy_true_params.shape[0], len(plds)*2)

#     # dummy_test_data = {
#     #     'PCASL': np.random.rand(dummy_true_params.shape[0], len(plds)) * 0.01,
#     #     'VSASL': np.random.rand(dummy_true_params.shape[0], len(plds)) * 0.01,
#     #     'MULTIVERSE_LS_FORMAT': np.random.rand(*dummy_data_shape_ls) * 0.01,
#     #     'NN_INPUT_FORMAT': np.random.rand(*dummy_data_shape_nn) * 0.01
#     # }
#     # results_df = comparator.compare_methods(dummy_test_data, dummy_true_params, plds, att_ranges_config)
#     # if not results_df.empty:
#     #     comparator.visualize_results(results_df)
#     # else:
#     #     logger.info("Example run produced no results.")