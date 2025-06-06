# single_repeat_validation.py
"""
Validation framework for comparing single-repeat NN performance
against multi-repeat conventional methods (key proposal objective)
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List 
from pathlib import Path 

from enhanced_asl_network import EnhancedASLNet
from enhanced_simulation import RealisticASLSimulator, ASLParameters

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

class SingleRepeatValidator:
    def __init__(self,
                 trained_model_path: Optional[str] = None,
                 base_nn_input_size: Optional[int] = 12, # Base input size (n_plds*2)
                 nn_model_arch_config: Optional[Dict] = None,
                 norm_stats: Optional[Dict] = None
                ):
        asl_sim_params = ASLParameters() 
        self.simulator = RealisticASLSimulator(params=asl_sim_params)
        self.plds = np.arange(500, 3001, 500) 

        self.base_nn_input_size = base_nn_input_size
        self.nn_model_arch_config = nn_model_arch_config if nn_model_arch_config is not None else {}
        self.norm_stats = norm_stats

        # Extract n_plds and m0_input_feature from arch_config for use in normalization helper
        self.nn_n_plds_for_norm = self.nn_model_arch_config.get('n_plds', 6) # Default to 6 if not in config
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
        model = EnhancedASLNet(
            input_size=self.base_nn_input_size, # Base size (num_plds*2)
            # Unpack the rest of the architecture config
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
        unique_params, _ = np.unique(dataset_params['parameters'], axis=0, return_index=True)
        if unique_params.shape[0] < n_subjects:
            logger.warning(f"Generated {unique_params.shape[0]} unique params, requested {n_subjects}.")
        
        cbf_values = unique_params[:n_subjects, 0]
        att_values = unique_params[:n_subjects, 1]
        actual_n_subjects = len(cbf_values)

        datasets = {
            'ground_truth_params': {'cbf': cbf_values, 'att': att_values},
            'signals_multi_repeat_avg': [], 
            'signals_single_repeat_high_noise': [],
            'signals_single_repeat_low_noise': []
        }
        if self.nn_model_arch_config.get('m0_input_feature', False): # If model uses M0, store M0 values too
            datasets['m0_values_high_noise'] = []
            datasets['m0_values_low_noise'] = []


        for i in range(actual_n_subjects):
            cbf, att = cbf_values[i], att_values[i]
            # generate_synthetic_data takes cbf_val in ml/100g/min
            
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
                    # Placeholder for M0 generation. If M0 is needed, it must be generated here.
                    # For now, a dummy M0 value.
                    dummy_m0_val = 1.0 # Or some function of cbf/att/condition
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
                signal_sr_flat = datasets[signal_key_prefix][i]
                
                nn_input_for_pred = signal_sr_flat
                if self.nn_model_arch_config.get('m0_input_feature', False):
                    m0_val = datasets[f'm0_values_{noise_key}'][i]
                    nn_input_for_pred = np.append(signal_sr_flat, m0_val) # Append M0
                
                if self.model:
                    cbf_nn, att_nn = self._predict_neural_network(nn_input_for_pred)
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

    def _predict_neural_network(self, signal_flat_with_m0_if_needed: np.ndarray) -> Tuple[float, float]:
        if self.model is None: return np.nan, np.nan

        normalized_signal_input = signal_flat_with_m0_if_needed
        if self.norm_stats:
            normalized_signal_input = apply_normalization_to_input_flat(
                signal_flat_with_m0_if_needed,
                self.norm_stats,
                self.nn_n_plds_for_norm, # Get from arch_config
                self.nn_m0_input_feature_for_norm # Get from arch_config
            )

        input_tensor = torch.FloatTensor(normalized_signal_input).unsqueeze(0).to(next(self.model.parameters()).device)
        with torch.no_grad():
            cbf_pred, att_pred, _, _ = self.model(input_tensor)
            return cbf_pred.item(), att_pred.item()

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
    norm_stats_for_nn: Optional[Dict] = None
):
    logger.info("=== Clinical Validation: Single-Repeat NN vs Multi-Repeat Conventional ===")
    if model_path is None: logger.warning("No model path provided. NN results will be NaN.")
    
    # Default NN config if not provided
    if nn_arch_config_for_model_load is None:
        logger.warning("No NN arch config provided for SingleRepeatValidator. Using defaults.")
        nn_arch_config_for_model_load = {
            'hidden_sizes': [256,128,64], 
            'n_plds': 6, 
            'm0_input_feature': False
            # Add other EnhancedASLNet defaults here if needed
        }
    if base_nn_input_size_for_model_load is None:
        base_nn_input_size_for_model_load = 12 # Default for 6 PLDs * 2 modalities

    validator = SingleRepeatValidator(trained_model_path=model_path,
                                      base_nn_input_size=base_nn_input_size_for_model_load,
                                      nn_model_arch_config=nn_arch_config_for_model_load,
                                      norm_stats=norm_stats_for_nn)
    logger.info("1. Simulating clinical acquisition scenarios...")
    datasets = validator.simulate_clinical_scenario(n_subjects=50) # Reduced for faster example
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
    # Example: run with a placeholder model path and basic config
    # You would get nn_config from your main ResearchConfig for consistency.
    example_model_path_main = None # "path/to/your/trained_model.pt"
    example_base_input_size = 12 # For 6 PLDs * 2 modalities
    example_nn_arch_config = {
        'hidden_sizes': [256, 128, 64], 
        'n_plds': 6, 
        'm0_input_feature': False, # Assuming M0 is not used by default for this test
        # ... add other necessary EnhancedASLNet parameters from model_creation_config
    }
    example_norm_stats = None # Path to or dict of norm_stats if used

    run_single_repeat_validation_main(model_path=example_model_path_main,
                                      base_nn_input_size_for_model_load=example_base_input_size,
                                      nn_arch_config_for_model_load=example_nn_arch_config,
                                      norm_stats_for_nn=example_norm_stats)