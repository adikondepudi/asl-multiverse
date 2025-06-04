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


class SingleRepeatValidator:
    def __init__(self,
                 trained_model_path: Optional[str] = None,
                 nn_input_size: int = 12, # Base input size (n_plds*2)
                 nn_hidden_sizes: Optional[List[int]] = None,
                 nn_n_plds: int = 6,
                 nn_m0_input_feature: bool = False # If model uses M0
                ):
        asl_sim_params = ASLParameters() 
        self.simulator = RealisticASLSimulator(params=asl_sim_params)
        self.plds = np.arange(500, 3001, 500) 

        self.nn_input_size_base = nn_input_size # Base size for ASL signals
        self.nn_hidden_sizes = nn_hidden_sizes if nn_hidden_sizes is not None else [256, 128, 64]
        self.nn_n_plds = nn_n_plds 
        self.nn_m0_input_feature = nn_m0_input_feature

        if trained_model_path and Path(trained_model_path).exists():
            self.model = self._load_trained_model(trained_model_path)
            logger.info(f"Loaded trained model from: {trained_model_path}")
        elif trained_model_path:
            logger.warning(f"Model path {trained_model_path} not found. NN predictions will be dummy values.")
            self.model = None
        else:
            self.model = None
            logger.info("No trained model path provided. NN predictions will be dummy values.")

    def _load_trained_model(self, model_path: str) -> Optional[EnhancedASLNet]:
        # Actual input size for model depends on M0 flag
        model_constructor_input_size = self.nn_input_size_base 
        # The EnhancedASLNet constructor internally adjusts if m0_input_feature=True
        
        model = EnhancedASLNet(
            input_size=model_constructor_input_size, # Pass base size
            hidden_sizes=self.nn_hidden_sizes,
            n_plds=self.nn_n_plds,
            use_transformer_temporal=True, # Assuming new default
            m0_input_feature=self.nn_m0_input_feature # Pass the flag
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
        if self.nn_m0_input_feature: # If model uses M0, store M0 values too
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
                if self.nn_m0_input_feature:
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
                if self.nn_m0_input_feature:
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
        input_tensor = torch.FloatTensor(signal_flat_with_m0_if_needed).unsqueeze(0).to(next(self.model.parameters()).device)
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

def run_single_repeat_validation_main(model_path: Optional[str] = None, nn_config: Optional[Dict]=None):
    logger.info("=== Clinical Validation: Single-Repeat NN vs Multi-Repeat Conventional ===")
    if model_path is None: logger.warning("No model path provided. NN results will be NaN.")
    
    # Default NN config if not provided
    if nn_config is None:
        nn_config = {
            'nn_input_size': 12, 'nn_hidden_sizes': [256,128,64], 
            'nn_n_plds': 6, 'nn_m0_input_feature': False
        }
    
    validator = SingleRepeatValidator(trained_model_path=model_path, **nn_config)
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
    example_model_path = None # "path/to/your/trained_model.pt"
    example_nn_config = {
        'nn_input_size': 12, # For 6 PLDs * 2 modalities
        'nn_hidden_sizes': [256, 128, 64], 
        'nn_n_plds': 6, 
        'nn_m0_input_feature': False # Assuming M0 is not used by default for this test
    }
    run_single_repeat_validation_main(model_path=example_model_path, nn_config=example_nn_config)
