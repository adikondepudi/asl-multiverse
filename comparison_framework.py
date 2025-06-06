import numpy as np
import torch
import matplotlib.pyplot as plt # Kept for plt.show() if needed during interactive use
import seaborn as sns # Kept for plt.show() if needed during interactive use
from typing import Dict, List, Tuple, Optional
import pandas as pd
from pathlib import Path
import time
from scipy import stats
from dataclasses import dataclass, asdict
import json
import wandb # Added for potential W&B artifact logging
import inspect # Added for the fix

from vsasl_functions import fit_VSASL_vectInit_pep
from pcasl_functions import fit_PCASL_vectInit_pep
from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
from enhanced_asl_network import EnhancedASLNet

import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Helper function for normalization (can be moved to a utils.py later)
def apply_normalization_to_input_flat(flat_signal: np.ndarray,
                                      norm_stats: Dict,
                                      num_plds_per_modality: int,
                                      has_m0: bool) -> np.ndarray:
    if not norm_stats or not isinstance(norm_stats, dict): return flat_signal

    pcasl_signal_part = flat_signal[:num_plds_per_modality]
    vsasl_signal_part = flat_signal[num_plds_per_modality : num_plds_per_modality*2]

    pcasl_norm = (pcasl_signal_part - norm_stats.get('pcasl_mean', 0)) / norm_stats.get('pcasl_std', 1)
    vsasl_norm = (vsasl_signal_part - norm_stats.get('vsasl_mean', 0)) / norm_stats.get('vsasl_std', 1)

    normalized_parts = [pcasl_norm, vsasl_norm]

    if has_m0:
        m0_signal_part = flat_signal[num_plds_per_modality*2:] # Assumes M0 is at the end
        if m0_signal_part.size > 0 : # Ensure M0 part exists
            m0_norm = (m0_signal_part - norm_stats.get('m0_mean', 0)) / norm_stats.get('m0_std', 1)
            normalized_parts.append(m0_norm)

    return np.concatenate(normalized_parts)

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
                 base_nn_input_size: int = 12, # Renamed from nn_input_size for clarity
                 nn_hidden_sizes: Optional[List[int]] = None,
                 nn_n_plds: int = 6,
                 nn_m0_input_feature: bool = False,
                 nn_use_transformer_temporal: bool = True,
                 nn_transformer_nlayers: int = 2,
                 nn_transformer_nhead: int = 4,
                 nn_model_arch_config: Optional[Dict] = None, # Added
                 norm_stats: Optional[Dict] = None # Added
                 ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.base_nn_input_size = base_nn_input_size # This is num_plds * 2
        self.nn_model_arch_config = nn_model_arch_config
        self.norm_stats = norm_stats

        # These are fallbacks if nn_model_arch_config is not provided
        self.nn_hidden_sizes = nn_hidden_sizes if nn_hidden_sizes is not None else [256, 128, 64]
        self.nn_n_plds = nn_n_plds
        self.nn_m0_input_feature = nn_m0_input_feature
        self.nn_use_transformer_temporal = nn_use_transformer_temporal
        self.nn_transformer_nlayers = nn_transformer_nlayers
        self.nn_transformer_nhead = nn_transformer_nhead


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
        if self.nn_model_arch_config:
            # FIX: Filter the config dictionary to only pass valid arguments to EnhancedASLNet
            model_param_keys = inspect.signature(EnhancedASLNet).parameters.keys()
            filtered_arch_config = {
                key: self.nn_model_arch_config[key]
                for key in self.nn_model_arch_config if key in model_param_keys
            }
            model = EnhancedASLNet(input_size=self.base_nn_input_size, **filtered_arch_config)
        else: # Fallback to individual parameters
            logger.warning("Loading NN model using individual parameters as nn_model_arch_config not provided.")
            model = EnhancedASLNet(input_size=self.base_nn_input_size,
                                   hidden_sizes=self.nn_hidden_sizes,
                                   n_plds=self.nn_n_plds,
                                   use_transformer_temporal=self.nn_use_transformer_temporal,
                                   transformer_nlayers=self.nn_transformer_nlayers,
                                   transformer_nhead=self.nn_transformer_nhead,
                                   m0_input_feature=self.nn_m0_input_feature)
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
            metrics['cbf_cov'] = (np.std(vc_pred_cbf) / mean_pred_cbf_safe) * 100
            metrics['cbf_rmse'] = np.sqrt(np.mean((vc_pred_cbf - vc_true_cbf)**2))
            metrics['cbf_nrmse_perc'] = (metrics['cbf_rmse'] / mean_true_cbf_safe) * 100
        if np.sum(valid_mask_att) > 0:
            va_pred_att, va_true_att = pred_att[valid_mask_att], true_att[valid_mask_att]
            metrics['att_bias'] = np.mean(va_pred_att - va_true_att)
            mean_true_att_safe = np.mean(va_true_att) if np.mean(va_true_att) != 0 else 1e-9
            metrics['att_nbias_perc'] = (metrics['att_bias'] / mean_true_att_safe) * 100
            mean_pred_att_safe = np.mean(va_pred_att) if np.mean(va_pred_att) != 0 else 1e-9
            metrics['att_cov'] = (np.std(va_pred_att) / mean_pred_att_safe) * 100
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
            range_data_signals = {'PCASL_LS': test_data_dict['PCASL'][mask], 'VSASL_LS': test_data_dict['VSASL'][mask],
                                  'MULTIVERSE_LS_FORMAT': test_data_dict['MULTIVERSE_LS_FORMAT'][mask],
                                  'NN_INPUT_FORMAT': test_data_dict['NN_INPUT_FORMAT'][mask]}
            logger.info(f"\nEvaluating {range_name_str} (n={mask.sum()})...")
            ls_eval_results = self._evaluate_least_squares(range_data_signals, range_true_params, plds_arr, range_name_str)
            self.results_list.extend(ls_eval_results)
            if self.nn_model:
                nn_eval_results = self._evaluate_neural_network(range_data_signals['NN_INPUT_FORMAT'], range_true_params, plds_arr, range_name_str)
                self.results_list.extend(nn_eval_results)
                hybrid_eval_results = self._evaluate_hybrid(range_data_signals['MULTIVERSE_LS_FORMAT'], range_data_signals['NN_INPUT_FORMAT'], range_true_params, plds_arr, range_name_str)
                self.results_list.extend(hybrid_eval_results)
            else: logger.info("Skipping Neural Network and Hybrid evaluations as model is not loaded.")
        if not self.results_list:
            logger.warning("No results generated from comparison. Returning empty DataFrame.")
            return pd.DataFrame()
        df = pd.DataFrame([asdict(r) for r in self.results_list])
        df_path = self.output_dir / 'comparison_results_detailed.csv'
        df.to_csv(df_path, index=False)
        logger.info(f"Comparison results saved to {df_path}")
        if wandb.run: wandb.save(str(df_path)) # Log to W&B artifacts
        return df

    def _evaluate_least_squares(self, data_signals: Dict[str, np.ndarray], true_params: np.ndarray, plds: np.ndarray, range_name: str) -> List[ComparisonResult]:
        ls_results_list = []
        logger.info("  Evaluating MULTIVERSE least-squares...")
        multiverse_res = self._fit_multiverse_ls(data_signals['MULTIVERSE_LS_FORMAT'], true_params, plds, range_name)
        if multiverse_res: ls_results_list.append(multiverse_res)
        logger.info("  Evaluating PCASL least-squares...")
        pcasl_res = self._fit_pcasl_ls(data_signals['PCASL_LS'], true_params, plds, range_name)
        if pcasl_res: ls_results_list.append(pcasl_res)
        logger.info("  Evaluating VSASL least-squares...")
        vsasl_res = self._fit_vsasl_ls(data_signals['VSASL_LS'], true_params, plds, range_name)
        if vsasl_res: ls_results_list.append(vsasl_res)
        return ls_results_list

    def _fit_multiverse_ls(self, signals_arr: np.ndarray, true_params_arr: np.ndarray, plds_arr: np.ndarray, range_name_str: str) -> Optional[ComparisonResult]:
        n_samples = signals_arr.shape[0]; successes = 0
        if n_samples == 0: return None
        cbf_estimates, att_estimates, ci_widths_cbf, ci_widths_att, fit_times = [], [], [], [], []
        pldti = np.column_stack([plds_arr, plds_arr])
        for i in range(n_samples):
            start_time = time.time()
            try:
                beta, conintval, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, signals_arr[i], [50.0/6000.0, 1500.0], **self.asl_params)
                fit_times.append(time.time() - start_time); cbf_estimates.append(beta[0]*6000); att_estimates.append(beta[1])
                ci_widths_cbf.append((conintval[0,1]-conintval[0,0])*6000); ci_widths_att.append(conintval[1,1]-conintval[1,0]); successes += 1
            except Exception: fit_times.append(time.time()-start_time); cbf_estimates.append(np.nan); att_estimates.append(np.nan); ci_widths_cbf.append(np.nan); ci_widths_att.append(np.nan)
        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:,0], att_estimates, true_params_arr[:,1])
        return ComparisonResult(method="MULTIVERSE-LS", att_range_name=range_name_str, **metrics,
                                cbf_ci_width=np.nanmean(ci_widths_cbf), att_ci_width=np.nanmean(ci_widths_att),
                                computation_time=np.nanmean(fit_times) if fit_times else np.nan, success_rate=(successes/n_samples*100) if n_samples > 0 else 0)

    def _fit_pcasl_ls(self, signals_arr: np.ndarray, true_params_arr: np.ndarray, plds_arr: np.ndarray, range_name_str: str) -> Optional[ComparisonResult]:
        n_samples = signals_arr.shape[0]; successes = 0
        if n_samples == 0: return None
        cbf_estimates, att_estimates, fit_times = [], [], []
        pcasl_params_subset = {k:v for k,v in self.asl_params.items() if 'VSASL' not in k}
        for i in range(n_samples):
            start_time = time.time()
            try:
                beta, _, _, _ = fit_PCASL_vectInit_pep(plds_arr, signals_arr[i], [50.0/6000.0, 1500.0], **pcasl_params_subset)
                fit_times.append(time.time()-start_time); cbf_estimates.append(beta[0]*6000); att_estimates.append(beta[1]); successes += 1
            except Exception: fit_times.append(time.time()-start_time); cbf_estimates.append(np.nan); att_estimates.append(np.nan)
        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:,0], att_estimates, true_params_arr[:,1])
        return ComparisonResult(method="PCASL-LS", att_range_name=range_name_str, **metrics, cbf_ci_width=np.nan, att_ci_width=np.nan,
                                computation_time=np.nanmean(fit_times) if fit_times else np.nan, success_rate=(successes/n_samples*100) if n_samples > 0 else 0)

    def _fit_vsasl_ls(self, signals_arr: np.ndarray, true_params_arr: np.ndarray, plds_arr: np.ndarray, range_name_str: str) -> Optional[ComparisonResult]:
        n_samples = signals_arr.shape[0]; successes = 0
        if n_samples == 0: return None
        cbf_estimates, att_estimates, fit_times = [], [], []
        vsasl_params_subset = {k:v for k,v in self.asl_params.items() if 'PCASL' not in k and 'T_tau' not in k}
        for i in range(n_samples):
            start_time = time.time()
            try:
                beta, _, _, _ = fit_VSASL_vectInit_pep(plds_arr, signals_arr[i], [50.0/6000.0, 1500.0], **vsasl_params_subset)
                fit_times.append(time.time()-start_time); cbf_estimates.append(beta[0]*6000); att_estimates.append(beta[1]); successes += 1
            except Exception: fit_times.append(time.time()-start_time); cbf_estimates.append(np.nan); att_estimates.append(np.nan)
        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:,0], att_estimates, true_params_arr[:,1])
        return ComparisonResult(method="VSASL-LS", att_range_name=range_name_str, **metrics, cbf_ci_width=np.nan, att_ci_width=np.nan,
                                computation_time=np.nanmean(fit_times) if fit_times else np.nan, success_rate=(successes/n_samples*100) if n_samples > 0 else 0)

    def _evaluate_neural_network(self, nn_input_arr: np.ndarray, true_params_arr: np.ndarray, plds_arr: np.ndarray, range_name_str: str) -> List[ComparisonResult]:
        if self.nn_model is None or nn_input_arr.shape[0] == 0: return []
        logger.info("  Evaluating Neural Network...")

        current_m0_feature_flag = self.nn_model_arch_config.get('m0_input_feature', self.nn_m0_input_feature) if self.nn_model_arch_config else self.nn_m0_input_feature
        current_n_plds = self.nn_model_arch_config.get('n_plds', self.nn_n_plds) if self.nn_model_arch_config else self.nn_n_plds

        if current_m0_feature_flag and nn_input_arr.shape[1] != (current_n_plds * 2 + 1):
            logger.error(f"NN expects M0, but input has {nn_input_arr.shape[1]} features. Expected {current_n_plds*2+1}.")
            return []

        # Apply normalization if norm_stats are available
        if self.norm_stats:
            normalized_nn_input_arr = np.array([
                apply_normalization_to_input_flat(sig, self.norm_stats, current_n_plds, current_m0_feature_flag)
                for sig in nn_input_arr])
            nn_input_arr = normalized_nn_input_arr

        input_tensor = torch.FloatTensor(nn_input_arr)
        start_time = time.time()
        with torch.no_grad(): cbf_pred, att_pred, cbf_log_var, att_log_var = self.nn_model(input_tensor)
        inference_time_total = time.time() - start_time
        cbf_est, att_est = cbf_pred.numpy().squeeze(), att_pred.numpy().squeeze()
        cbf_unc_std, att_unc_std = np.exp(cbf_log_var.numpy().squeeze()/2.0), np.exp(att_log_var.numpy().squeeze()/2.0)
        metrics = self._calculate_detailed_metrics(cbf_est, true_params_arr[:,0], att_est, true_params_arr[:,1])
        return [ComparisonResult(method="Neural Network", att_range_name=range_name_str, **metrics,
                                 cbf_ci_width=np.nanmean(cbf_unc_std)*1.96*2, att_ci_width=np.nanmean(att_unc_std)*1.96*2,
                                 computation_time=inference_time_total/len(nn_input_arr) if len(nn_input_arr)>0 else np.nan, success_rate=100.0)]

    def _evaluate_hybrid(self, multiverse_ls_signals: np.ndarray, nn_input_signals: np.ndarray, true_params_arr: np.ndarray, plds_arr: np.ndarray, range_name_str: str) -> List[ComparisonResult]:
        if self.nn_model is None or nn_input_signals.shape[0] == 0: return []
        logger.info("  Evaluating Hybrid approach...")

        current_m0_feature_flag = self.nn_model_arch_config.get('m0_input_feature', self.nn_m0_input_feature) if self.nn_model_arch_config else self.nn_m0_input_feature
        current_n_plds = self.nn_model_arch_config.get('n_plds', self.nn_n_plds) if self.nn_model_arch_config else self.nn_n_plds

        # Apply normalization to NN input signals for hybrid initialization
        if self.norm_stats:
            normalized_nn_input_signals = np.array([
                apply_normalization_to_input_flat(sig, self.norm_stats, current_n_plds, current_m0_feature_flag)
                for sig in nn_input_signals])
            nn_input_signals = normalized_nn_input_signals

        input_tensor = torch.FloatTensor(nn_input_signals)
        with torch.no_grad(): cbf_init_nn, att_init_nn, _, _ = self.nn_model(input_tensor)
        cbf_init_ls, att_init_ls = cbf_init_nn.numpy().squeeze()/6000.0, att_init_nn.numpy().squeeze()
        n_samples = multiverse_ls_signals.shape[0]; successes = 0 # Ensure n_samples is defined for this loop
        cbf_estimates, att_estimates, fit_times = [], [], []
        pldti = np.column_stack([plds_arr, plds_arr])
        for i in range(n_samples):
            start_time = time.time()
            try:
                beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(pldti, multiverse_ls_signals[i], [cbf_init_ls[i], att_init_ls[i]], **self.asl_params)
                fit_times.append(time.time()-start_time); cbf_estimates.append(beta[0]*6000); att_estimates.append(beta[1]); successes += 1
            except Exception: fit_times.append(time.time()-start_time); cbf_estimates.append(cbf_init_nn.numpy().squeeze()[i]); att_estimates.append(att_init_ls[i]); successes +=1
        metrics = self._calculate_detailed_metrics(cbf_estimates, true_params_arr[:,0], att_estimates, true_params_arr[:,1])
        return [ComparisonResult(method="Hybrid (NN+LS)", att_range_name=range_name_str, **metrics,
                                 cbf_ci_width=np.nan, att_ci_width=np.nan,
                                 computation_time=np.nanmean(fit_times) if fit_times else np.nan, success_rate=(successes/n_samples*100) if n_samples > 0 else 0)]

    def visualize_results(self, results_df: pd.DataFrame, show_plots: bool = False): # Added show_plots
        if results_df.empty:
            logger.warning("Results DataFrame is empty. Skipping visualization.")
            return
        if not show_plots: # If not showing, just log that visualization data is available
            logger.info("Visualization data prepared. Set show_plots=True to display.")
            return

        sns.set_style("whitegrid")
        fig_metrics, axes_metrics = plt.subplots(3, 2, figsize=(16, 20))
        all_method_names = results_df['method'].unique()
        method_order = [m for m in ['PCASL-LS', 'VSASL-LS', 'MULTIVERSE-LS', 'Neural Network', 'Hybrid (NN+LS)'] if m in all_method_names]
        colors = {'PCASL-LS': 'blue', 'VSASL-LS': 'green', 'MULTIVERSE-LS': 'red',
                  'Neural Network': 'purple', 'Hybrid (NN+LS)': 'orange', 'Other': 'grey'}
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
                ax_cbf.plot(x_pos, method_df[cbf_metric_key], 'o-', color=colors.get(method_name, colors['Other']), label=method_name, linewidth=2.5 if 'Neural' in method_name or 'Hybrid' in method_name else 2, markersize=7)
                ax_att.plot(x_pos, method_df[att_metric_key], 'o-', color=colors.get(method_name, colors['Other']), label=method_name, linewidth=2.5 if 'Neural' in method_name or 'Hybrid' in method_name else 2, markersize=7)
            for ax, param_name, unit in [(ax_cbf, "CBF", cbf_unit), (ax_att, "ATT", att_unit)]:
                ax.set_ylabel(f'{param_name} {metric_disp_name} ({unit})'); ax.set_title(f'{param_name} {metric_disp_name}')
                ax.set_xticks(x_pos); ax.set_xticklabels(att_range_names_sorted, rotation=30, ha='right')
                if 'Bias' in metric_disp_name: ax.axhline(0, color='k', linestyle='--', alpha=0.7)
                if row == 0: ax.legend(fontsize='small')
        plt.tight_layout(rect=[0, 0, 1, 0.96]); fig_metrics.suptitle("Performance Comparison Across Methods and ATT Ranges", fontsize=16, fontweight='bold')
        # plt.savefig(self.output_dir / 'comparison_metrics_summary.png', dpi=300, bbox_inches='tight'); # Removed savefig
        if show_plots: plt.show()
        plt.close(fig_metrics)

        self._plot_computation_time(results_df, show_plots)
        self._plot_success_rates(results_df, show_plots)
        self._plot_calibration_placeholder(show_plots) # Placeholder, does not save real data

    def _plot_computation_time(self, results_df: pd.DataFrame, show_plots: bool = False):
        if results_df.empty or 'computation_time' not in results_df.columns: logger.warning("Cannot plot computation time."); return
        if not show_plots: return
        fig, ax = plt.subplots(figsize=(12, 7))
        avg_times = results_df.groupby('method')['computation_time'].mean().sort_values() * 1000 # ms
        bars = ax.bar(avg_times.index, avg_times.values)
        ax.set_ylabel('Avg. Computation Time per Sample (ms)'); ax.set_title('Average Computation Time Comparison'); ax.set_yscale('log')
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height) and height > 0: ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.2f}', ha='center', va='bottom')
        plt.xticks(rotation=45, ha='right'); plt.tight_layout()
        # plt.savefig(self.output_dir / 'computation_time_comparison.png', dpi=300, bbox_inches='tight'); # Removed savefig
        if show_plots: plt.show()
        plt.close(fig)

    def _plot_success_rates(self, results_df: pd.DataFrame, show_plots: bool = False):
        if results_df.empty or 'success_rate' not in results_df.columns: logger.warning("Cannot plot success rates."); return
        if not show_plots: return
        try:
            pivot_data = results_df.pivot_table(values='success_rate', index='att_range_name', columns='method', aggfunc='mean').reindex(sorted(results_df['att_range_name'].unique()))
            if pivot_data.empty: logger.warning("Pivot table for success rates is empty."); return
            pivot_data.plot(kind='bar', figsize=(12,7), width=0.8)
            plt.ylabel('Success Rate (%)'); plt.xlabel('ATT Range'); plt.title('Fitting Success Rates by Method and ATT Range')
            plt.legend(title='Method', bbox_to_anchor=(1.02, 1), loc='upper left'); plt.xticks(rotation=30, ha='right')
            plt.ylim(0, 105); plt.tight_layout()
            # plt.savefig(self.output_dir / 'success_rates_comparison.png', dpi=300, bbox_inches='tight'); # Removed savefig
            if show_plots: plt.show()
            plt.close()
        except Exception as e: logger.error(f"Error plotting success rates: {e}")

    def _plot_calibration_placeholder(self, show_plots: bool = False):
        logger.info("Calibration plot generation is a placeholder.")
        if not show_plots: return
        fig, ax = plt.subplots(1,1,figsize=(8,6))
        ax.text(0.5, 0.5, "Calibration Plot Placeholder\n(Requires full implementation with raw predictions)",
                horizontalalignment='center', verticalalignment='center', fontsize=14,
                bbox=dict(boxstyle="round,pad=0.5", fc="aliceblue", ec="lightsteelblue", lw=2))
        ax.set_xticks([]); ax.set_yticks([])
        # plt.savefig(self.output_dir / 'calibration_plot_placeholder.png', dpi=150); # Removed savefig
        if show_plots: plt.show()
        plt.close(fig)