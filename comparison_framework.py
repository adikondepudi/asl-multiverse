import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import time
from dataclasses import dataclass, asdict
import wandb
import inspect

from joblib import Parallel, delayed
import multiprocessing

from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from enhanced_asl_network import EnhancedASLNet

import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

def _fit_single_ls_voxel(signal_reshaped, pldti, asl_params):
    """Helper function to fit a single voxel, designed for use with joblib."""
    try:
        # Use a standard, reasonable initial guess for the fit
        init_guess = [50.0 / 6000.0, 1500.0]
        beta, conintval, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signal_reshaped, init_guess, **asl_params)
        cbf_est = beta[0] * 6000
        att_est = beta[1]
        cbf_ci = (conintval[0, 1] - conintval[0, 0]) * 6000
        att_ci = conintval[1, 1] - conintval[1, 0]
        return cbf_est, att_est, cbf_ci, att_ci, True
    except Exception:
        # If the fit fails for any reason, return NaNs and a failure flag
        return np.nan, np.nan, np.nan, np.nan, False

def apply_normalization_to_input_flat(flat_signal: np.ndarray,
                                      norm_stats: Dict,
                                      num_plds_per_modality: int,
                                      has_m0: bool) -> np.ndarray:
    if not norm_stats or not isinstance(norm_stats, dict):
        return flat_signal

    raw_signal_len = num_plds_per_modality * 2
    signal_part = flat_signal[:raw_signal_len]
    other_features_part = flat_signal[raw_signal_len:]

    pcasl_norm = (signal_part[:num_plds_per_modality] - norm_stats.get('pcasl_mean', 0)) / np.clip(norm_stats.get('pcasl_std', 1), a_min=1e-6, a_max=None)
    vsasl_norm = (signal_part[num_plds_per_modality:] - norm_stats.get('vsasl_mean', 0)) / np.clip(norm_stats.get('vsasl_std', 1), a_min=1e-6, a_max=None)

    return np.concatenate([pcasl_norm, vsasl_norm, other_features_part])

def denormalize_predictions(cbf_pred_norm: np.ndarray, att_pred_norm: np.ndarray,
                            cbf_log_var_norm: Optional[np.ndarray], att_log_var_norm: Optional[np.ndarray],
                            norm_stats: Dict) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    if not norm_stats:
        logger.warning("No norm_stats provided for de-normalization. Returning predictions as is.")
        cbf_unc_std_denorm = np.exp(cbf_log_var_norm / 2.0) if cbf_log_var_norm is not None else None
        att_unc_std_denorm = np.exp(att_log_var_norm / 2.0) if att_log_var_norm is not None else None
        return cbf_pred_norm, att_pred_norm, cbf_unc_std_denorm, att_unc_std_denorm

    y_mean_cbf = norm_stats.get('y_mean_cbf', 0.0)
    y_std_cbf = norm_stats.get('y_std_cbf', 1.0)
    if y_std_cbf < 1e-6: y_std_cbf = 1.0
    
    y_mean_att = norm_stats.get('y_mean_att', 0.0)
    y_std_att = norm_stats.get('y_std_att', 1.0)
    if y_std_att < 1e-6: y_std_att = 1.0

    cbf_pred_denorm = cbf_pred_norm * y_std_cbf + y_mean_cbf
    att_pred_denorm = att_pred_norm * y_std_att + y_mean_att

    cbf_unc_std_denorm = None
    if cbf_log_var_norm is not None:
        cbf_unc_std_denorm = np.exp(cbf_log_var_norm / 2.0) * y_std_cbf
        
    att_unc_std_denorm = None
    if att_log_var_norm is not None:
        att_unc_std_denorm = np.exp(att_log_var_norm / 2.0) * y_std_att
        
    return cbf_pred_denorm, att_pred_denorm, cbf_unc_std_denorm, att_unc_std_denorm


@dataclass
class ComparisonResult:
    method: str
    att_range_name: str
    cbf_bias: float
    att_bias: float
    cbf_cov: float
    att_cov: float
    cbf_rmse: float
    att_rmse: float
    cbf_nbias_perc: float
    att_nbias_perc: float
    cbf_nrmse_perc: float
    att_nrmse_perc: float
    cbf_ci_width: float
    att_ci_width: float
    computation_time: float
    success_rate: float


class ComprehensiveComparison:
    def __init__(self,
                 nn_model_path: Optional[str] = None,
                 output_dir: str = 'comparison_results',
                 base_nn_input_size: int = 12, 
                 nn_model_arch_config: Optional[Dict] = None, 
                 norm_stats: Optional[Dict] = None 
                 ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_nn_input_size = base_nn_input_size 
        self.nn_model_arch_config = nn_model_arch_config
        self.norm_stats = norm_stats

        if nn_model_path and Path(nn_model_path).exists():
            self.nn_model = self._load_nn_model(nn_model_path)
            logger.info(f"Loaded NN model from {nn_model_path}")
        elif nn_model_path:
            logger.warning(f"NN model path {nn_model_path} does not exist. NN evaluation will be skipped.")
            self.nn_model = None
        else:
            self.nn_model = None
            logger.info("No NN model path provided. NN evaluation will be skipped.")

        self.asl_params = {'T1_artery': 1850.0, 'T2_factor': 1.0, 'alpha_BS1': 1.0,
                           'alpha_PCASL': 0.85, 'alpha_VSASL': 0.56, 'T_tau': 1800.0}
        self.results_list = []

    def _load_nn_model(self, model_path: str) -> torch.nn.Module:
        model_params_to_use = self.nn_model_arch_config if self.nn_model_arch_config else {}
        if not self.nn_model_arch_config:
            logger.warning("Loading NN model using default parameters as nn_model_arch_config not provided.")

        # The constructor is designed to handle extra kwargs, so we can pass the whole config.
        model = EnhancedASLNet(input_size=self.base_nn_input_size, **model_params_to_use)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        return model

    def _calculate_detailed_metrics(self, pred_cbf, true_cbf, pred_att, true_att):
        pred_cbf, true_cbf, pred_att, true_att = map(np.asarray, [pred_cbf, true_cbf, pred_att, true_att])
        valid_mask_cbf, valid_mask_att = ~np.isnan(pred_cbf) & ~np.isnan(true_cbf), ~np.isnan(pred_att) & ~np.isnan(true_att)
        metrics = {k: np.nan for k in ['cbf_bias', 'att_bias', 'cbf_cov', 'att_cov', 'cbf_rmse', 'att_rmse',
                                       'cbf_nbias_perc', 'att_nbias_perc', 'cbf_nrmse_perc', 'att_nrmse_perc']}
        if np.sum(valid_mask_cbf) > 0:
            vc_pred_cbf, vc_true_cbf = pred_cbf[valid_mask_cbf], true_cbf[valid_mask_cbf]
            metrics['cbf_bias'] = np.mean(vc_pred_cbf - vc_true_cbf)
            mean_true_cbf_safe = np.mean(vc_true_cbf) if np.mean(vc_true_cbf) != 0 else 1e-9
            metrics['cbf_nbias_perc'] = (metrics['cbf_bias'] / mean_true_cbf_safe) * 100
            mean_pred_cbf_safe = np.mean(vc_pred_cbf) if np.mean(vc_pred_cbf) != 0 else 1e-9
            metrics['cbf_cov'] = (np.std(vc_pred_cbf) / mean_pred_cbf_safe) * 100 if mean_pred_cbf_safe != 0 else np.nan
            metrics['cbf_rmse'] = np.sqrt(np.mean((vc_pred_cbf - vc_true_cbf)**2))
            metrics['cbf_nrmse_perc'] = (metrics['cbf_rmse'] / mean_true_cbf_safe) * 100
        if np.sum(valid_mask_att) > 0:
            va_pred_att, va_true_att = pred_att[valid_mask_att], true_att[valid_mask_att]
            metrics['att_bias'] = np.mean(va_pred_att - va_true_att)
            mean_true_att_safe = np.mean(va_true_att) if np.mean(va_true_att) != 0 else 1e-9
            metrics['att_nbias_perc'] = (metrics['att_bias'] / mean_true_att_safe) * 100
            mean_pred_att_safe = np.mean(va_pred_att) if np.mean(va_pred_att) != 0 else 1e-9
            metrics['att_cov'] = (np.std(va_pred_att) / mean_pred_att_safe) * 100 if mean_pred_att_safe != 0 else np.nan
            metrics['att_rmse'] = np.sqrt(np.mean((va_pred_att - va_true_att)**2))
            metrics['att_nrmse_perc'] = (metrics['att_rmse'] / mean_true_att_safe) * 100
        return metrics

    def compare_methods(self, test_data_dict: Dict[str, np.ndarray], true_params_arr: np.ndarray,
                       plds_arr: np.ndarray, att_ranges_list: List[Tuple[float, float, str]]) -> pd.DataFrame:
        self.results_list = []
        for att_min, att_max, range_name_str in att_ranges_list:
            mask = (true_params_arr[:, 1] >= att_min) & (true_params_arr[:, 1] < att_max)
            if not np.any(mask):
                logger.warning(f"No test data for ATT range {range_name_str}. Skipping.")
                continue
            range_true_params = true_params_arr[mask]
            range_data_signals = {
                'LS_INPUT_FORMAT': test_data_dict['MULTIVERSE_LS_FORMAT'][mask],
                'NN_INPUT_FORMAT': test_data_dict['NN_INPUT_FORMAT'][mask]
            }
            logger.info(f"\nEvaluating {range_name_str} (n={mask.sum()})...")

            ls_eval_results = self._evaluate_least_squares(range_data_signals['LS_INPUT_FORMAT'], range_true_params, plds_arr, range_name_str)
            self.results_list.extend(ls_eval_results)

            if self.nn_model:
                nn_eval_results = self._evaluate_neural_network(range_data_signals['NN_INPUT_FORMAT'], range_true_params, plds_arr, range_name_str)
                self.results_list.extend(nn_eval_results)
            else:
                logger.info("Skipping Neural Network evaluation as model is not loaded.")

        if not self.results_list:
            logger.warning("No results generated from comparison. Returning empty DataFrame.")
            return pd.DataFrame()
        df = pd.DataFrame([asdict(r) for r in self.results_list])
        df_path = self.output_dir / 'comparison_results_detailed.csv'
        df.to_csv(df_path, index=False)
        logger.info(f"Comparison results saved to {df_path}")
        if wandb.run: wandb.save(str(df_path))
        return df

    def _evaluate_least_squares(self, data_signals: np.ndarray, true_params: np.ndarray, plds: np.ndarray, range_name: str) -> List[ComparisonResult]:
        logger.info("  Evaluating Least-Squares (LS)...")
        ls_res = self._fit_ls(data_signals, true_params, plds, range_name)
        return [ls_res] if ls_res else []

    # Helper function for a single voxel fit
    def _fit_single_ls_voxel(signal_reshaped, pldti, asl_params):
        try:
            beta, conintval, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signal_reshaped, [50.0/6000.0, 1500.0], **asl_params)
            cbf_est = beta[0] * 6000
            att_est = beta[1]
            cbf_ci = (conintval[0,1]-conintval[0,0])*6000
            att_ci = conintval[1,1]-conintval[1,0]
            return cbf_est, att_est, cbf_ci, att_ci, True
        except Exception:
            return np.nan, np.nan, np.nan, np.nan, False

    def _fit_ls(self, signals_arr: np.ndarray, true_params_arr: np.ndarray, plds_arr: np.ndarray, range_name_str: str) -> Optional[ComparisonResult]:
        n_samples = signals_arr.shape[0]
        if n_samples == 0:
            return None
        
        pldti = np.column_stack([plds_arr, plds_arr])
        # Use all available CPU cores on the node for maximum speed
        num_cores = multiprocessing.cpu_count()
        
        start_time = time.time()
        
        # Use joblib to run the fitting on all cores in parallel
        # The `delayed` function wraps our helper function for lazy execution
        results = Parallel(n_jobs=num_cores)(
            delayed(_fit_single_ls_voxel)(signals_arr[i], pldti, self.asl_params)
            for i in range(n_samples)
        )
        
        total_time = time.time() - start_time
        
        # Unpack the results from all the parallel jobs
        cbf_estimates, att_estimates, ci_widths_cbf, ci_widths_att, success_flags = zip(*results)
        
        # Convert to numpy arrays for calculation
        cbf_estimates = np.array(cbf_estimates)
        att_estimates = np.array(att_estimates)
        
        # Calculate success rate
        successes = sum(success_flags)
        success_rate = (successes / n_samples * 100) if n_samples > 0 else 0
        
        # Calculate performance metrics on the results
        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:, 0], att_estimates, true_params_arr[:, 1])
        
        # Return the final result object
        return ComparisonResult(method="LS",
                                att_range_name=range_name_str,
                                **metrics,
                                cbf_ci_width=np.nanmean(ci_widths_cbf),
                                att_ci_width=np.nanmean(ci_widths_att),
                                computation_time=total_time / n_samples if n_samples > 0 else np.nan,
                                success_rate=success_rate)

    def _evaluate_neural_network(self, nn_input_arr: np.ndarray, true_params_arr: np.ndarray, plds_arr: np.ndarray, range_name_str: str) -> List[ComparisonResult]:
        if self.nn_model is None or nn_input_arr.shape[0] == 0: return []
        logger.info("  Evaluating Neural Network (NN)...")

        current_n_plds = self.nn_model_arch_config.get('n_plds', 6) if self.nn_model_arch_config else 6
        current_m0_feature_flag = self.nn_model_arch_config.get('m0_input_feature', False) if self.nn_model_arch_config else False
        
        normalized_nn_input_arr_eval = nn_input_arr
        if self.norm_stats:
            normalized_nn_input_arr_eval = np.array([
                apply_normalization_to_input_flat(sig, self.norm_stats, current_n_plds, current_m0_feature_flag)
                for sig in nn_input_arr])

        input_tensor = torch.FloatTensor(normalized_nn_input_arr_eval)
        start_time = time.time()
        with torch.no_grad():
            cbf_pred_norm, att_pred_norm, cbf_log_var_norm, att_log_var_norm, _, _ = self.nn_model(input_tensor)
        inference_time_total = time.time() - start_time
        
        cbf_est, att_est, cbf_unc_std, att_unc_std = denormalize_predictions(
            cbf_pred_norm.numpy().squeeze(), att_pred_norm.numpy().squeeze(),
            cbf_log_var_norm.numpy().squeeze(), att_log_var_norm.numpy().squeeze(),
            self.norm_stats
        )
        
        metrics = self._calculate_detailed_metrics(cbf_est, true_params_arr[:,0], att_est, true_params_arr[:,1])
        return [ComparisonResult(method="NN", att_range_name=range_name_str, **metrics,
                                 cbf_ci_width=np.nanmean(cbf_unc_std)*1.96*2 if cbf_unc_std is not None else np.nan, 
                                 att_ci_width=np.nanmean(att_unc_std)*1.96*2 if att_unc_std is not None else np.nan,
                                 computation_time=inference_time_total/len(nn_input_arr) if len(nn_input_arr)>0 else np.nan, success_rate=100.0)]

    def visualize_results(self, results_df: pd.DataFrame, show_plots: bool = False): 
        if results_df.empty:
            logger.warning("Results DataFrame is empty. Skipping visualization.")
            return

        sns.set_style("whitegrid")
        fig_metrics, axes_metrics = plt.subplots(3, 2, figsize=(16, 20))
        all_method_names = results_df['method'].unique()
        method_order = [m for m in ['LS', 'NN'] if m in all_method_names]
        colors = {'LS': 'red', 'NN': 'purple'}
        
        metrics_to_plot = [
            ('cbf_nbias_perc', 'att_nbias_perc', 'Normalized Bias', '%', '%'),
            ('cbf_cov', 'att_cov', 'Coefficient of Variation', '%', '%'),
            ('cbf_nrmse_perc', 'att_nrmse_perc', 'Normalized RMSE', '%', '%')]
        att_range_names_sorted = sorted(results_df['att_range_name'].unique(), key=lambda x: (x.split(' ')[0] != "Short", x.split(' ')[0] != "Medium", x.split(' ')[0] != "Long", x))

        for row, (cbf_metric_key, att_metric_key, metric_disp_name, cbf_unit, att_unit) in enumerate(metrics_to_plot):
            ax_cbf, ax_att = axes_metrics[row, 0], axes_metrics[row, 1]
            for method_name in method_order:
                method_df = results_df[results_df['method'] == method_name].set_index('att_range_name').reindex(att_range_names_sorted).reset_index()
                x_pos = range(len(att_range_names_sorted))
                ax_cbf.plot(x_pos, method_df[cbf_metric_key], 'o-', color=colors.get(method_name), label=method_name, linewidth=2.5, markersize=7)
                ax_att.plot(x_pos, method_df[att_metric_key], 'o-', color=colors.get(method_name), label=method_name, linewidth=2.5, markersize=7)
            for ax, param_name, unit in [(ax_cbf, "CBF", cbf_unit), (ax_att, "ATT", att_unit)]:
                ax.set_ylabel(f'{param_name} {metric_disp_name} ({unit})'); ax.set_title(f'{param_name} {metric_disp_name}')
                ax.set_xticks(x_pos); ax.set_xticklabels(att_range_names_sorted, rotation=30, ha='right')
                if 'Bias' in metric_disp_name: ax.axhline(0, color='k', linestyle='--', alpha=0.7)
                if row == 0: ax.legend(fontsize='small')
        plt.tight_layout(rect=[0, 0, 1, 0.96]); fig_metrics.suptitle("Performance Comparison: LS vs. NN", fontsize=16, fontweight='bold')
        if show_plots: plt.show()
        plt.close(fig_metrics)

        self._plot_computation_time(results_df, show_plots)
        self._plot_success_rates(results_df, show_plots)

    def _plot_computation_time(self, results_df: pd.DataFrame, show_plots: bool = False):
        if results_df.empty or 'computation_time' not in results_df.columns: logger.warning("Cannot plot computation time."); return
        if not show_plots: return
        fig, ax = plt.subplots(figsize=(10, 6))
        avg_times = results_df.groupby('method')['computation_time'].mean().sort_values() * 1000 # ms
        bars = ax.bar(avg_times.index, avg_times.values, color=[{'LS':'red','NN':'purple'}.get(x) for x in avg_times.index])
        ax.set_ylabel('Avg. Computation Time per Sample (ms)'); ax.set_title('Average Computation Time Comparison'); ax.set_yscale('log')
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height) and height > 0: ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        plt.xticks(rotation=0); plt.tight_layout()
        if show_plots: plt.show()
        plt.close(fig)

    def _plot_success_rates(self, results_df: pd.DataFrame, show_plots: bool = False):
        if results_df.empty or 'success_rate' not in results_df.columns: logger.warning("Cannot plot success rates."); return
        if not show_plots: return
        
        filtered_df = results_df[results_df['method'].isin(['LS', 'NN'])]
        if filtered_df.empty: return
        
        try:
            pivot_data = filtered_df.pivot_table(values='success_rate', index='att_range_name', columns='method', aggfunc='mean').reindex(sorted(filtered_df['att_range_name'].unique()))
            if pivot_data.empty: logger.warning("Pivot table for success rates is empty."); return
            pivot_data.plot(kind='bar', figsize=(10, 6), width=0.8, color={'LS':'red','NN':'purple'})
            plt.ylabel('Success Rate (%)'); plt.xlabel('ATT Range'); plt.title('Fitting Success Rates by Method and ATT Range')
            plt.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left'); plt.xticks(rotation=30, ha='right')
            plt.ylim(0, 105); plt.tight_layout()
            if show_plots: plt.show()
            plt.close()
        except Exception as e: logger.error(f"Error plotting success rates: {e}")