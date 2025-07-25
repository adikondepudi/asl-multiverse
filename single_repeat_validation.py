import numpy as np
import torch
from typing import Dict, Tuple, Optional, List 
from pathlib import Path 
import inspect 

from enhanced_asl_network import EnhancedASLNet
from enhanced_simulation import RealisticASLSimulator, ASLParameters
from comparison_framework import denormalize_predictions # Import de-normalization helper
from utils import engineer_signal_features

import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Helper function for normalization (can be moved to a utils.py later)
def apply_normalization_to_input_flat(flat_signal: np.ndarray,
                                      norm_stats: Dict,
                                      num_plds_per_modality: int,
                                      has_m0: bool) -> np.ndarray:
    if not norm_stats or not isinstance(norm_stats, dict):
        return flat_signal

    # Isolate signal part from other features (like engineered features)
    raw_signal_len = num_plds_per_modality * 2
    signal_part = flat_signal[:raw_signal_len]
    other_features_part = flat_signal[raw_signal_len:]

    pcasl_signal_part = signal_part[:num_plds_per_modality]
    vsasl_signal_part = signal_part[num_plds_per_modality:]
    
    pcasl_norm = (pcasl_signal_part - norm_stats.get('pcasl_mean', 0)) / norm_stats.get('pcasl_std', 1)
    vsasl_norm = (vsasl_signal_part - norm_stats.get('vsasl_mean', 0)) / norm_stats.get('vsasl_std', 1)

    # Reconcatenate normalized signal with the untouched other features.
    # The has_m0 flag becomes irrelevant as M0 would be part of other_features_part if present.
    return np.concatenate([pcasl_norm, vsasl_norm, other_features_part])

class SingleRepeatValidator:
    def __init__(self,
                 trained_model_path: Optional[str] = None,
                 base_nn_input_size: Optional[int] = 12, 
                 nn_model_arch_config: Optional[Dict] = None,
                 norm_stats: Optional[Dict] = None # For input and output de/normalization
                ):
        asl_sim_params = ASLParameters() 
        self.simulator = RealisticASLSimulator(params=asl_sim_params)
        self.plds = np.arange(500, 3001, 500) 

        self.base_nn_input_size = base_nn_input_size
        self.nn_model_arch_config = nn_model_arch_config if nn_model_arch_config is not None else {}
        self.norm_stats = norm_stats

        self.nn_n_plds_for_norm = self.nn_model_arch_config.get('n_plds', 6) 
        self.nn_m0_input_feature_for_norm = self.nn_model_arch_config.get('m0_input_feature', False)

        if trained_model_path and Path(trained_model_path).exists():
            self.model = self._load_trained_model(trained_model_path)
            logger.info(f"Loaded trained model from: {trained_model_path}")
        elif trained_model_path:
            logger.warning(f"Model path {trained_model_path} not found. NN predictions will be NaN.")
            self.model = None
        else:
            self.model = None
            logger.info("No trained model path provided. NN predictions will be NaN.")

    def _load_trained_model(self, model_path: str) -> Optional[EnhancedASLNet]:
        # The nn_model_arch_config contains all necessary parameters, including
        # those handled by **kwargs in the EnhancedASLNet constructor.
        # The constructor is robust enough to pick what it needs.
        model = EnhancedASLNet(
            input_size=self.base_nn_input_size,
            **self.nn_model_arch_config
        )
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}. Returning None.")
            return None

    def simulate_clinical_scenario(self, n_subjects: int = 100) -> Dict:
        logger.info(f"Simulating clinical scenario for {n_subjects} subjects.")
        dataset_params = self.simulator.generate_diverse_dataset(
            plds=self.plds, n_subjects=n_subjects, conditions=['healthy'], noise_levels=[5.0]
        )
        # Ensure we get unique parameters for the number of subjects requested
        unique_param_indices = np.sort(np.unique(dataset_params['parameters'], axis=0, return_index=True)[1])
        actual_n_subjects = min(n_subjects, len(unique_param_indices))
        selected_indices = unique_param_indices[:actual_n_subjects]
        
        cbf_values = dataset_params['parameters'][selected_indices, 0]
        att_values = dataset_params['parameters'][selected_indices, 1]


        datasets = {
            'ground_truth_params': {'cbf': cbf_values, 'att': att_values},
            'signals_multi_repeat_avg': [], 
            'signals_single_repeat_high_noise': [],
            'signals_single_repeat_low_noise': []
        }
        if self.nn_model_arch_config.get('m0_input_feature', False): 
            datasets['m0_values_high_noise'] = []
            datasets['m0_values_low_noise'] = []


        for i in range(actual_n_subjects):
            cbf, att = cbf_values[i], att_values[i]
            
            multi_repeat_raw = []
            for _ in range(4):
                signals_rep = self.simulator.generate_synthetic_data(
                    self.plds, np.array([att]), n_noise=1, tsnr=20.0, cbf_val=cbf 
                )
                multi_repeat_raw.append(signals_rep['MULTIVERSE'][0, 0, :, :])
            datasets['signals_multi_repeat_avg'].append(np.mean(multi_repeat_raw, axis=0).flatten())

            for noise_key, snr_val in [('high_noise', 5.0), ('low_noise', 10.0)]:
                single_rep_dict = self.simulator.generate_synthetic_data(
                    self.plds, np.array([att]), n_noise=1, tsnr=snr_val, cbf_val=cbf
                )
                flat_signal = np.concatenate([
                    single_rep_dict['PCASL'][0,0,:], single_rep_dict['VSASL'][0,0,:]
                ])
                datasets[f'signals_single_repeat_{noise_key}'].append(flat_signal)
                if self.nn_model_arch_config.get('m0_input_feature', False):
                    dummy_m0_val = 1.0 
                    datasets[f'm0_values_{noise_key}'].append(dummy_m0_val)


        for key in datasets:
            if 'signals_' in key or 'm0_values_' in key:
                datasets[key] = np.array(datasets[key])
        return datasets

    def compare_methods(self, datasets: Dict) -> Dict:
        logger.info("Comparing estimation methods...")
        results = {
            'conventional_multi_repeat_avg': {'cbf': [], 'att': []},
            'nn_single_repeat_high_noise': {'cbf': [], 'att': []},
            'nn_single_repeat_low_noise': {'cbf': [], 'att': []},
            'conventional_single_repeat_high_noise': {'cbf': [], 'att': []}
        }
        n_subjects = len(datasets['ground_truth_params']['cbf'])

        for i in range(n_subjects):
            if (i + 1) % max(1, n_subjects // 10) == 0: logger.info(f"Processing subject {i+1}/{n_subjects}")

            signal_mr_avg_flat = datasets['signals_multi_repeat_avg'][i]
            signal_mr_avg_reshaped = signal_mr_avg_flat.reshape(len(self.plds), 2)
            cbf_ls_mr, att_ls_mr = self._fit_conventional(signal_mr_avg_reshaped)
            results['conventional_multi_repeat_avg']['cbf'].append(cbf_ls_mr)
            results['conventional_multi_repeat_avg']['att'].append(att_ls_mr)

            for noise_key, signal_key_prefix in [('high_noise', 'signals_single_repeat_high_noise'), 
                                                 ('low_noise', 'signals_single_repeat_low_noise')]:
                signal_sr_flat_raw = datasets[signal_key_prefix][i]
                
                # Apply the same feature engineering as used during training
                engineered_features = engineer_signal_features(
                    signal_sr_flat_raw,
                    num_plds=self.nn_n_plds_for_norm
                )
                nn_input_for_pred = np.concatenate([signal_sr_flat_raw, engineered_features.flatten()])

                if self.nn_model_arch_config.get('m0_input_feature', False):
                    m0_val = datasets[f'm0_values_{noise_key}'][i]
                    nn_input_for_pred = np.append(nn_input_for_pred, m0_val) 
                
                if self.model:
                    cbf_nn, att_nn, _, _ = self._predict_neural_network(nn_input_for_pred) # Ignored uncertainty for this validation
                    results[f'nn_single_repeat_{noise_key}']['cbf'].append(cbf_nn)
                    results[f'nn_single_repeat_{noise_key}']['att'].append(att_nn)
                else:
                    results[f'nn_single_repeat_{noise_key}']['cbf'].append(np.nan)
                    results[f'nn_single_repeat_{noise_key}']['att'].append(np.nan)
            
            signal_sr_high_reshaped = datasets['signals_single_repeat_high_noise'][i].reshape(len(self.plds), 2)
            cbf_ls_sr, att_ls_sr = self._fit_conventional(signal_sr_high_reshaped)
            results['conventional_single_repeat_high_noise']['cbf'].append(cbf_ls_sr)
            results['conventional_single_repeat_high_noise']['att'].append(att_ls_sr)

        for method_key in results:
            for param_key in results[method_key]:
                results[method_key][param_key] = np.array(results[method_key][param_key])
        return results

    def _fit_conventional(self, signal_reshaped: np.ndarray) -> Tuple[float, float]:
        from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep 
        pldti = np.column_stack([self.plds, self.plds])
        init_cbf_ls, init_att_ls = 50.0 / 6000.0, 1500.0
        try:
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                pldti, signal_reshaped, [init_cbf_ls, init_att_ls],
                T1_artery=self.simulator.params.T1_artery, T_tau=self.simulator.params.T_tau,
                T2_factor=self.simulator.params.T2_factor, alpha_BS1=self.simulator.params.alpha_BS1,
                alpha_PCASL=self.simulator.params.alpha_PCASL, alpha_VSASL=self.simulator.params.alpha_VSASL
            )
            return beta[0] * 6000.0, beta[1] 
        except Exception: return np.nan, np.nan

    def _predict_neural_network(self, signal_flat_with_m0_if_needed: np.ndarray) -> Tuple[float, float, float, float]:
        if self.model is None: return np.nan, np.nan, np.nan, np.nan

        normalized_signal_input = signal_flat_with_m0_if_needed
        if self.norm_stats:
            normalized_signal_input = apply_normalization_to_input_flat(
                signal_flat_with_m0_if_needed,
                self.norm_stats,
                self.nn_n_plds_for_norm, 
                self.nn_m0_input_feature_for_norm 
            )

        input_tensor = torch.FloatTensor(normalized_signal_input).unsqueeze(0).to(next(self.model.parameters()).device)
        with torch.no_grad():
            cbf_pred_norm, att_pred_norm, cbf_log_var_norm, att_log_var_norm, _, _ = self.model(input_tensor)
        
        # De-normalize predictions
        cbf_pred_denorm, att_pred_denorm, cbf_std_denorm, att_std_denorm = denormalize_predictions(
            cbf_pred_norm.cpu().numpy(), att_pred_norm.cpu().numpy(),
            cbf_log_var_norm.cpu().numpy(), att_log_var_norm.cpu().numpy(),
            self.norm_stats
        )
        return cbf_pred_denorm.item(), att_pred_denorm.item(), cbf_std_denorm.item(), att_std_denorm.item()


    def calculate_scan_time_benefits(self, results: Dict, ground_truth_params: Dict) -> Dict:
        logger.info("Calculating scan time benefits and performance.")
        single_repeat_time, multi_repeat_4x_time = 2.5, 10.0
        true_cbf_arr, true_att_arr = ground_truth_params['cbf'], ground_truth_params['att']
        performance_summary = {}

        for method_key, estimates in results.items():
            pred_cbf, pred_att = estimates['cbf'], estimates['att']
            valid_mask = ~np.isnan(pred_cbf) & ~np.isnan(pred_att)
            if not np.any(valid_mask):
                logger.warning(f"No valid estimates for method {method_key}.")
                performance_summary[method_key] = {'cbf_rmse': np.nan, 'att_rmse': np.nan, 'cbf_bias': np.nan, 'att_bias': np.nan, 'cbf_cov': np.nan, 'att_cov': np.nan, 'scan_time_minutes': multi_repeat_4x_time if 'multi_repeat' in method_key else single_repeat_time, 'efficiency_score': np.nan, 'num_valid_fits': 0}
                continue
            vp_cbf, vt_cbf = pred_cbf[valid_mask], true_cbf_arr[valid_mask]
            vp_att, vt_att = pred_att[valid_mask], true_att_arr[valid_mask]
            metrics = {
                'cbf_rmse': np.sqrt(np.mean((vp_cbf - vt_cbf)**2)),
                'att_rmse': np.sqrt(np.mean((vp_att - vt_att)**2)),
                'cbf_bias': np.mean(vp_cbf - vt_cbf),
                'att_bias': np.mean(vp_att - vt_att),
                'cbf_cov': (np.std(vp_cbf) / np.mean(vp_cbf) * 100) if np.mean(vp_cbf) !=0 else np.nan,
                'att_cov': (np.std(vp_att) / np.mean(vp_att) * 100) if np.mean(vp_att) !=0 else np.nan,
                'scan_time_minutes': multi_repeat_4x_time if 'multi_repeat' in method_key else single_repeat_time,
                'num_valid_fits': np.sum(valid_mask)
            }
            denom = metrics['cbf_rmse'] * metrics['att_rmse'] * metrics['scan_time_minutes']
            metrics['efficiency_score'] = 1.0 / denom if denom > 1e-9 else np.nan
            performance_summary[method_key] = metrics
        return performance_summary

def run_single_repeat_validation_main(
    model_path: Optional[str] = None,
    base_nn_input_size_for_model_load: Optional[int] = None,
    nn_arch_config_for_model_load: Optional[Dict] = None,
    norm_stats_for_nn: Optional[Dict] = None # Added norm_stats
):
    logger.info("=== Clinical Validation: Single-Repeat NN vs Multi-Repeat Conventional ===")
    if model_path is None: logger.warning("No model path provided. NN results will be NaN.")
    
    if nn_arch_config_for_model_load is None:
        logger.warning("No NN arch config provided for SingleRepeatValidator. Using defaults.")
        nn_arch_config_for_model_load = {
            'hidden_sizes': [256,128,64], 
            'n_plds': 6, 
            'm0_input_feature': False
        }
    if base_nn_input_size_for_model_load is None:
        # Calculate from config if possible, otherwise default
        n_plds = len(nn_arch_config_for_model_load.get('pld_values', range(500,3001,500)))
        base_nn_input_size_for_model_load = n_plds * 2 + 4 # 2 for modalities, 4 for engineered features

    validator = SingleRepeatValidator(trained_model_path=model_path,
                                      base_nn_input_size=base_nn_input_size_for_model_load,
                                      nn_model_arch_config=nn_arch_config_for_model_load,
                                      norm_stats=norm_stats_for_nn) # Pass norm_stats
    logger.info("1. Simulating clinical acquisition scenarios...")
    datasets = validator.simulate_clinical_scenario(n_subjects=50) 
    logger.info("2. Comparing methods...")
    estimation_results = validator.compare_methods(datasets)
    logger.info("3. Calculating scan time benefits and performance metrics...")
    performance_metrics = validator.calculate_scan_time_benefits(estimation_results, datasets['ground_truth_params'])
    logger.info("\n=== SINGLE REPEAT VALIDATION RESULTS ===")
    for method, perf in performance_metrics.items():
        logger.info(f"\nMethod: {method.upper()}")
        logger.info(f"  Valid Fits: {perf['num_valid_fits']}/{len(datasets['ground_truth_params']['cbf'])}")
        logger.info(f"  CBF RMSE: {perf['cbf_rmse']:.2f}, Bias: {perf['cbf_bias']:.2f}, CoV: {perf['cbf_cov']:.2f}%")
        logger.info(f"  ATT RMSE: {perf['att_rmse']:.2f}, Bias: {perf['att_bias']:.2f}, CoV: {perf['att_cov']:.2f}%")
        logger.info(f"  Scan Time: {perf['scan_time_minutes']:.1f} min, Efficiency: {perf['efficiency_score']:.4f}")
    return estimation_results, performance_metrics

if __name__ == "__main__":
    example_model_path_main = None 
    example_base_input_size = 16 # 12 raw + 4 engineered
    example_nn_arch_config = {
        'hidden_sizes': [256, 128, 64], 
        'n_plds': 6, 
        'm0_input_feature': False, 
    }
    example_norm_stats = None 

    run_single_repeat_validation_main(model_path=example_model_path_main,
                                      base_nn_input_size_for_model_load=example_base_input_size,
                                      nn_arch_config_for_model_load=example_nn_arch_config,
                                      norm_stats_for_nn=example_norm_stats)