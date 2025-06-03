# single_repeat_validation.py
"""
Validation framework for comparing single-repeat NN performance
against multi-repeat conventional methods (key proposal objective)
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List # Added Optional, List
from pathlib import Path # Added Path

# Assuming EnhancedASLNet is correctly imported
from enhanced_asl_network import EnhancedASLNet
from enhanced_simulation import RealisticASLSimulator, ASLParameters

# For logging within this file
import logging
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


class SingleRepeatValidator:
    """Validate NN performance on single-repeat data vs multi-repeat baseline"""

    def __init__(self,
                 trained_model_path: Optional[str] = None,
                 # NN model config (should come from main config ideally)
                 nn_input_size: int = 12,
                 nn_hidden_sizes: Optional[List[int]] = None,
                 nn_n_plds: int = 6):

        # Use default ASLParameters and then RealisticASLSimulator
        asl_sim_params = ASLParameters() # Using default physiological params for RealisticASLSimulator
        self.simulator = RealisticASLSimulator(params=asl_sim_params)
        self.plds = np.arange(500, 3001, 500) # Default PLDs

        self.nn_input_size = nn_input_size
        self.nn_hidden_sizes = nn_hidden_sizes if nn_hidden_sizes is not None else [256, 128, 64]
        self.nn_n_plds = nn_n_plds # Should match input_size / 2

        if trained_model_path and Path(trained_model_path).exists():
            self.model = self._load_trained_model(trained_model_path)
            logger.info(f"Loaded trained model from: {trained_model_path}")
        elif trained_model_path:
            logger.warning(f"Model path {trained_model_path} not found. NN predictions will be dummy values.")
            self.model = None
        else:
            self.model = None
            logger.info("No trained model path provided. NN predictions will be dummy values.")


    def _load_trained_model(self, model_path: str) -> EnhancedASLNet:
        """Load pre-trained neural network model"""
        model = EnhancedASLNet(
            input_size=self.nn_input_size,
            hidden_sizes=self.nn_hidden_sizes,
            n_plds=self.nn_n_plds
            )
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            return model
        except FileNotFoundError:
            logger.error(f"Model file not found at {model_path}. Returning None.")
            return None
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}. Returning None.")
            return None


    def simulate_clinical_scenario(self, n_subjects: int = 100) -> Dict:
        """
        Simulate clinical scenario:
        - Ground truth parameters
        - Multi-repeat (4x) conventional LS "best estimate" (using averaged signal)
        - Single-repeat noisy data for NN processing and single-repeat LS.
        """
        logger.info(f"Simulating clinical scenario for {n_subjects} subjects.")
        # Generate ground truth parameters using RealisticASLSimulator's diversity
        # This will create a mix of CBF/ATT based on 'healthy' conditions
        # We need to ensure the simulator used here has its CBF parameter set per subject for generate_synthetic_data,
        # OR, we use generate_diverse_dataset for more controlled parameter generation.
        # For simplicity, let's use generate_diverse_dataset for parameter generation.

        dataset_params = self.simulator.generate_diverse_dataset(
            plds=self.plds,
            n_subjects=n_subjects,
            conditions=['healthy'], # Focus on one condition for this specific validation
            noise_levels=[5.0] # Noise level here is for parameter generation, actual test noise is set below
        )
        # Extract a unique set of CBF/ATT parameters
        unique_params, unique_indices = np.unique(dataset_params['parameters'], axis=0, return_index=True)
        if unique_params.shape[0] < n_subjects:
            logger.warning(f"Generated fewer than {n_subjects} unique parameter sets ({unique_params.shape[0]}). Using available unique sets.")
        
        cbf_values = unique_params[:n_subjects, 0]
        att_values = unique_params[:n_subjects, 1]
        actual_n_subjects = len(cbf_values)


        datasets = {
            'ground_truth_params': {'cbf': cbf_values, 'att': att_values},
            'signals_multi_repeat_avg': [], # Averaged signal for multi-repeat LS
            'signals_single_repeat_high_noise': [], # For NN and single-repeat LS
            'signals_single_repeat_low_noise': []  # For NN (alternative scenario)
        }

        for i in range(actual_n_subjects):
            cbf, att = cbf_values[i], att_values[i]
            # Important: Set the simulator's CBF for generate_synthetic_data
            self.simulator.params.CBF = cbf # Set current CBF for the base ASLSimulator methods
            self.simulator.cbf = cbf / 6000.0 # Update internal cbf in ml/g/s

            # Simulate multi-repeat acquisition (4 repeats, effective SNR is higher)
            multi_repeat_signals_raw = [] # Store individual repeats
            for _ in range(4):
                # generate_synthetic_data returns a dict {'PCASL': ..., 'VSASL': ..., 'MULTIVERSE': ...}
                # signals['MULTIVERSE'] has shape (n_noise, n_att, n_plds, 2)
                # Here n_noise=1, n_att=1 (since we pass np.array([att]))
                signals_one_repeat = self.simulator.generate_synthetic_data(
                    self.plds, np.array([att]), n_noise=1, tsnr=20.0 # High SNR for individual repeats
                )
                multi_repeat_signals_raw.append(signals_one_repeat['MULTIVERSE'][0, 0, :, :]) # Get (n_plds, 2)
            
            averaged_signal = np.mean(multi_repeat_signals_raw, axis=0) # Average over repeats -> (n_plds, 2)
            datasets['signals_multi_repeat_avg'].append(averaged_signal.flatten()) # Flatten to (n_plds*2,)

            # Single repeat with higher noise (SNR~5, typical for single acquisition)
            single_high_noise_dict = self.simulator.generate_synthetic_data(
                self.plds, np.array([att]), n_noise=1, tsnr=5.0
            )
            datasets['signals_single_repeat_high_noise'].append(single_high_noise_dict['MULTIVERSE'][0, 0, :, :].flatten())

            # Single repeat with moderate noise (SNR~10)
            single_low_noise_dict = self.simulator.generate_synthetic_data(
                self.plds, np.array([att]), n_noise=1, tsnr=10.0
            )
            datasets['signals_single_repeat_low_noise'].append(single_low_noise_dict['MULTIVERSE'][0, 0, :, :].flatten())

        # Convert signal lists to numpy arrays
        for key in datasets:
            if 'signals_' in key:
                datasets[key] = np.array(datasets[key])
        
        return datasets


    def compare_methods(self, datasets: Dict) -> Dict:
        """Compare NN on single-repeat vs conventional on multi-repeat"""
        logger.info("Comparing estimation methods...")
        results = {
            'conventional_multi_repeat_avg': {'cbf': [], 'att': []}, # LS on 4x averaged signal
            'nn_single_repeat_high_noise': {'cbf': [], 'att': []},
            'nn_single_repeat_low_noise': {'cbf': [], 'att': []},
            'conventional_single_repeat_high_noise': {'cbf': [], 'att': []} # LS on 1x high noise signal
        }

        n_subjects = len(datasets['ground_truth_params']['cbf'])

        for i in range(n_subjects):
            if (i + 1) % (n_subjects // 10 if n_subjects >= 10 else 1) == 0:
                 logger.info(f"Processing subject {i+1}/{n_subjects}")

            # 1. Conventional method on multi-repeat averaged data (clinical standard)
            # Signal shape (n_plds*2,) -> need to reshape to (n_plds, 2) for _fit_conventional
            signal_mr_avg_flat = datasets['signals_multi_repeat_avg'][i]
            signal_mr_avg_reshaped = signal_mr_avg_flat.reshape(len(self.plds), 2)
            cbf_ls_mr, att_ls_mr = self._fit_conventional(signal_mr_avg_reshaped)
            results['conventional_multi_repeat_avg']['cbf'].append(cbf_ls_mr)
            results['conventional_multi_repeat_avg']['att'].append(att_ls_mr)

            # 2. NN on single-repeat high-noise data
            signal_sr_high_flat = datasets['signals_single_repeat_high_noise'][i] # Already (n_plds*2,)
            if self.model:
                cbf_nn_high, att_nn_high = self._predict_neural_network(signal_sr_high_flat)
                results['nn_single_repeat_high_noise']['cbf'].append(cbf_nn_high)
                results['nn_single_repeat_high_noise']['att'].append(att_nn_high)
            else: # Append NaN if no model
                results['nn_single_repeat_high_noise']['cbf'].append(np.nan)
                results['nn_single_repeat_high_noise']['att'].append(np.nan)


            # 3. NN on single-repeat low-noise data
            signal_sr_low_flat = datasets['signals_single_repeat_low_noise'][i]
            if self.model:
                cbf_nn_low, att_nn_low = self._predict_neural_network(signal_sr_low_flat)
                results['nn_single_repeat_low_noise']['cbf'].append(cbf_nn_low)
                results['nn_single_repeat_low_noise']['att'].append(att_nn_low)
            else:
                results['nn_single_repeat_low_noise']['cbf'].append(np.nan)
                results['nn_single_repeat_low_noise']['att'].append(np.nan)

            # 4. Conventional method on single-repeat high-noise data (for fair comparison to NN on high noise)
            signal_sr_high_reshaped = signal_sr_high_flat.reshape(len(self.plds), 2)
            cbf_ls_sr, att_ls_sr = self._fit_conventional(signal_sr_high_reshaped)
            results['conventional_single_repeat_high_noise']['cbf'].append(cbf_ls_sr)
            results['conventional_single_repeat_high_noise']['att'].append(att_ls_sr)

        # Convert result lists to numpy arrays
        for method_key in results:
            for param_key in results[method_key]:
                results[method_key][param_key] = np.array(results[method_key][param_key])
        
        return results

    def _fit_conventional(self, signal_reshaped: np.ndarray) -> Tuple[float, float]:
        """Fit using conventional MULTIVERSE least-squares. Expects signal (n_plds, 2)."""
        from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep # Local import
        pldti = np.column_stack([self.plds, self.plds])
        # Default fixed initial guess for LS fitting robustness
        init_cbf_ls, init_att_ls = 50.0 / 6000.0, 1500.0
        try:
            # Assuming default ASL params from self.simulator.params are appropriate
            # These should ideally be consistent with how data was generated/trained
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                pldti, signal_reshaped, [init_cbf_ls, init_att_ls],
                T1_artery=self.simulator.params.T1_artery, T_tau=self.simulator.params.T_tau,
                T2_factor=self.simulator.params.T2_factor, alpha_BS1=self.simulator.params.alpha_BS1,
                alpha_PCASL=self.simulator.params.alpha_PCASL, alpha_VSASL=self.simulator.params.alpha_VSASL
            )
            return beta[0] * 6000.0, beta[1] # CBF in ml/100g/min, ATT in ms
        except Exception as e:
            # logger.debug(f"LS fitting failed in _fit_conventional: {e}")
            return np.nan, np.nan # Return NaN if fitting fails

    def _predict_neural_network(self, signal_flat: np.ndarray) -> Tuple[float, float]:
        """Predict using neural network. Expects signal_flat (n_plds*2,)."""
        if self.model is None:
            return np.nan, np.nan

        input_tensor = torch.FloatTensor(signal_flat).unsqueeze(0) # Add batch dimension
        with torch.no_grad():
            # NN is trained to output CBF in ml/100g/min and ATT in ms directly
            cbf_pred, att_pred, _, _ = self.model(input_tensor)
            # No *6000 needed here if network trained on ml/100g/min targets
            return cbf_pred.item(), att_pred.item()


    def calculate_scan_time_benefits(self, results: Dict, ground_truth_params: Dict) -> Dict:
        """Calculate scan time reduction benefits and performance metrics."""
        logger.info("Calculating scan time benefits and performance.")
        # Typical scan times (minutes, placeholder values)
        single_repeat_time = 2.5
        multi_repeat_4x_time = 10.0

        true_cbf_arr = ground_truth_params['cbf']
        true_att_arr = ground_truth_params['att']
        
        performance_summary = {}

        for method_key, estimates in results.items():
            pred_cbf = estimates['cbf']
            pred_att = estimates['att']

            # Filter out NaNs for robust metric calculation
            valid_mask = ~np.isnan(pred_cbf) & ~np.isnan(pred_att)
            if not np.any(valid_mask):
                logger.warning(f"No valid estimates for method {method_key}. Skipping metrics.")
                performance_summary[method_key] = {
                    'cbf_rmse': np.nan, 'att_rmse': np.nan,
                    'cbf_bias': np.nan, 'att_bias': np.nan,
                    'cbf_cov': np.nan, 'att_cov': np.nan,
                    'scan_time_minutes': multi_repeat_4x_time if 'multi_repeat' in method_key else single_repeat_time,
                    'efficiency_score': np.nan,
                    'num_valid_fits': 0
                }
                continue

            # Use only valid estimates and corresponding true values
            vp_cbf, vt_cbf = pred_cbf[valid_mask], true_cbf_arr[valid_mask]
            vp_att, vt_att = pred_att[valid_mask], true_att_arr[valid_mask]

            cbf_rmse = np.sqrt(np.mean((vp_cbf - vt_cbf)**2))
            att_rmse = np.sqrt(np.mean((vp_att - vt_att)**2))
            cbf_bias = np.mean(vp_cbf - vt_cbf)
            att_bias = np.mean(vp_att - vt_att)
            cbf_cov = np.std(vp_cbf) / np.mean(vp_cbf) * 100 if np.mean(vp_cbf) !=0 else np.nan
            att_cov = np.std(vp_att) / np.mean(vp_att) * 100 if np.mean(vp_att) !=0 else np.nan
            
            scan_time = multi_repeat_4x_time if 'multi_repeat' in method_key else single_repeat_time
            
            # Efficiency score (higher is better, avoid division by zero)
            efficiency_denominator = cbf_rmse * att_rmse * scan_time
            efficiency_score = 1.0 / efficiency_denominator if efficiency_denominator > 1e-9 else np.nan
            
            performance_summary[method_key] = {
                'cbf_rmse': cbf_rmse, 'att_rmse': att_rmse,
                'cbf_bias': cbf_bias, 'att_bias': att_bias,
                'cbf_cov': cbf_cov, 'att_cov': att_cov,
                'scan_time_minutes': scan_time,
                'efficiency_score': efficiency_score,
                'num_valid_fits': np.sum(valid_mask)
            }
        return performance_summary

def run_single_repeat_validation_main(model_path: Optional[str] = None):
    """Main function to run clinical validation for single repeat performance."""
    logger.info("=== Clinical Validation: Single-Repeat NN vs Multi-Repeat Conventional ===")

    # Path to a trained model (replace with actual path after training with main.py)
    # Example: model_path = "comprehensive_results/asl_research_YYYYMMDD_HHMMSS/trained_models/ensemble_model_0.pt"
    if model_path is None:
        logger.warning("No model path provided to run_single_repeat_validation_main. NN results will be NaN.")
    
    validator = SingleRepeatValidator(trained_model_path=model_path)

    logger.info("1. Simulating clinical acquisition scenarios...")
    datasets = validator.simulate_clinical_scenario(n_subjects=100) # Reduced n_subjects for faster run

    logger.info("2. Comparing methods...")
    estimation_results = validator.compare_methods(datasets)
    
    logger.info("3. Calculating scan time benefits and performance metrics...")
    performance_metrics = validator.calculate_scan_time_benefits(estimation_results, datasets['ground_truth_params'])

    logger.info("\n=== SINGLE REPEAT VALIDATION RESULTS ===")
    for method, perf in performance_metrics.items():
        logger.info(f"\nMethod: {method.upper()}")
        logger.info(f"  Number of Valid Fits: {perf['num_valid_fits']}/{len(datasets['ground_truth_params']['cbf'])}")
        logger.info(f"  CBF RMSE: {perf['cbf_rmse']:.2f} mL/100g/min, Bias: {perf['cbf_bias']:.2f}, CoV: {perf['cbf_cov']:.2f}%")
        logger.info(f"  ATT RMSE: {perf['att_rmse']:.2f} ms, Bias: {perf['att_bias']:.2f}, CoV: {perf['att_cov']:.2f}%")
        logger.info(f"  Scan Time: {perf['scan_time_minutes']:.1f} minutes")
        logger.info(f"  Efficiency Score: {perf['efficiency_score']:.4f}")

    return estimation_results, performance_metrics

if __name__ == "__main__":
    # Example: run with a placeholder model path
    # Replace 'path/to/your/trained_model.pt' with an actual model file after training
    # For testing without a model, it will show NaNs for NN methods.
    example_model_path = None # Or "path/to/your/trained_model.pt" 
    run_single_repeat_validation_main(model_path=example_model_path)