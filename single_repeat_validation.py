# single_repeat_validation.py
"""
Validation framework for comparing single-repeat NN performance 
against multi-repeat conventional methods (key proposal objective)
"""

import numpy as np
import torch
from typing import Dict, Tuple
from enhanced_asl_network import EnhancedASLNet
from enhanced_simulation import RealisticASLSimulator

class SingleRepeatValidator:
    """Validate NN performance on single-repeat data vs multi-repeat baseline"""
    
    def __init__(self, trained_model_path: str = None):
        self.simulator = RealisticASLSimulator()
        self.plds = np.arange(500, 3001, 500)
        
        if trained_model_path:
            self.model = self._load_trained_model(trained_model_path)
        else:
            self.model = None
    
    def _load_trained_model(self, model_path: str) -> EnhancedASLNet:
        """Load pre-trained neural network model"""
        model = EnhancedASLNet(input_size=12, n_plds=6)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model
    
    def simulate_clinical_scenario(self, n_subjects: int = 100) -> Dict:
        """
        Simulate clinical scenario: 
        - High-quality 4-repeat data as "ground truth"
        - Single-repeat noisy data for NN processing
        """
        
        # Generate ground truth parameters
        cbf_values = np.random.uniform(30, 90, n_subjects)  # Realistic CBF range
        att_values = np.random.uniform(800, 3200, n_subjects)  # Wide ATT range
        
        datasets = {
            'ground_truth': {'cbf': cbf_values, 'att': att_values},
            'multi_repeat_4x': [],
            'single_repeat_high_noise': [],
            'single_repeat_low_noise': []
        }
        
        for i, (cbf, att) in enumerate(zip(cbf_values, att_values)):
            # Simulate multi-repeat acquisition (4 repeats, SNR~20)
            multi_repeat_signals = []
            for repeat in range(4):
                signals = self.simulator.generate_synthetic_data(
                    self.plds, np.array([att]), n_noise=1, tsnr=20.0
                )
                multi_repeat_signals.append(signals['MULTIVERSE'][0, 0])
            
            # Average the repeats (this is the clinical standard)
            averaged_signal = np.mean(multi_repeat_signals, axis=0)
            datasets['multi_repeat_4x'].append(averaged_signal)
            
            # Single repeat with higher noise (SNR~5, typical for single acquisition)
            single_high_noise = self.simulator.generate_synthetic_data(
                self.plds, np.array([att]), n_noise=1, tsnr=5.0
            )
            datasets['single_repeat_high_noise'].append(single_high_noise['MULTIVERSE'][0, 0])
            
            # Single repeat with moderate noise (SNR~10)
            single_low_noise = self.simulator.generate_synthetic_data(
                self.plds, np.array([att]), n_noise=1, tsnr=10.0
            )
            datasets['single_repeat_low_noise'].append(single_low_noise['MULTIVERSE'][0, 0])
        
        # Convert to numpy arrays
        for key in ['multi_repeat_4x', 'single_repeat_high_noise', 'single_repeat_low_noise']:
            datasets[key] = np.array(datasets[key])
        
        return datasets
    
    def compare_methods(self, datasets: Dict) -> Dict:
        """Compare NN on single-repeat vs conventional on multi-repeat"""
        
        results = {
            'conventional_multi_repeat': {'cbf': [], 'att': []},
            'nn_single_repeat_high_noise': {'cbf': [], 'att': []},
            'nn_single_repeat_low_noise': {'cbf': [], 'att': []},
            'conventional_single_repeat': {'cbf': [], 'att': []}  # For comparison
        }
        
        # Process each subject
        n_subjects = len(datasets['ground_truth']['cbf'])
        
        for i in range(n_subjects):
            # 1. Conventional method on multi-repeat data (clinical standard)
            conv_multi = self._fit_conventional(datasets['multi_repeat_4x'][i])
            results['conventional_multi_repeat']['cbf'].append(conv_multi[0] * 6000)
            results['conventional_multi_repeat']['att'].append(conv_multi[1])
            
            # 2. Neural network on single-repeat high-noise data
            if self.model:
                nn_high = self._predict_neural_network(datasets['single_repeat_high_noise'][i])
                results['nn_single_repeat_high_noise']['cbf'].append(nn_high[0])
                results['nn_single_repeat_high_noise']['att'].append(nn_high[1])
                
                # 3. Neural network on single-repeat low-noise data
                nn_low = self._predict_neural_network(datasets['single_repeat_low_noise'][i])
                results['nn_single_repeat_low_noise']['cbf'].append(nn_low[0])
                results['nn_single_repeat_low_noise']['att'].append(nn_low[1])
            
            # 4. Conventional method on single-repeat data (for fair comparison)
            conv_single = self._fit_conventional(datasets['single_repeat_high_noise'][i])
            results['conventional_single_repeat']['cbf'].append(conv_single[0] * 6000)
            results['conventional_single_repeat']['att'].append(conv_single[1])
        
        # Convert to numpy arrays
        for method in results:
            for param in results[method]:
                results[method][param] = np.array(results[method][param])
        
        return results
    
    def _fit_conventional(self, signal: np.ndarray) -> Tuple[float, float]:
        """Fit using conventional least-squares method"""
        from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
        
        # Create PLDTI array (matched PLDs/TIs)
        pldti = np.column_stack([self.plds, self.plds])
        
        # Reshape signal for fitting
        signal_2d = signal.reshape(-1, 2)
        
        try:
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                pldti, signal_2d, [50/6000, 1500],
                1850, 1800, 1.0, 1.0, 0.85, 0.56
            )
            return beta[0], beta[1]
        except:
            return 50/6000, 1500  # Return initial guess if fitting fails
    
    def _predict_neural_network(self, signal: np.ndarray) -> Tuple[float, float]:
        """Predict using neural network"""
        if self.model is None:
            return 60.0, 1500.0  # Dummy values if no model
        
        # Reshape for network input
        input_tensor = torch.FloatTensor(signal.flatten()).unsqueeze(0)
        
        with torch.no_grad():
            cbf_pred, att_pred, _, _ = self.model(input_tensor)
            return cbf_pred.item() * 6000, att_pred.item()
    
    def calculate_scan_time_benefits(self, results: Dict) -> Dict:
        """Calculate scan time reduction benefits"""
        
        # Typical scan times (minutes)
        single_repeat_time = 2.5  # Single acquisition
        multi_repeat_4x_time = 10.0  # Four acquisitions
        
        # Performance comparison
        ground_truth_cbf = results['ground_truth']['cbf']
        ground_truth_att = results['ground_truth']['att']
        
        performance = {}
        
        for method in ['conventional_multi_repeat', 'nn_single_repeat_high_noise', 
                      'nn_single_repeat_low_noise', 'conventional_single_repeat']:
            
            cbf_rmse = np.sqrt(np.mean((results[method]['cbf'] - ground_truth_cbf)**2))
            att_rmse = np.sqrt(np.mean((results[method]['att'] - ground_truth_att)**2))
            
            scan_time = multi_repeat_4x_time if 'multi_repeat' in method else single_repeat_time
            
            performance[method] = {
                'cbf_rmse': cbf_rmse,
                'att_rmse': att_rmse,
                'scan_time_minutes': scan_time,
                'efficiency_score': 1.0 / (cbf_rmse * att_rmse * scan_time)  # Higher is better
            }
        
        return performance

def run_clinical_validation():
    """Main function to run clinical validation"""
    
    print("=== Clinical Validation: Single-Repeat NN vs Multi-Repeat Conventional ===")
    
    validator = SingleRepeatValidator()
    
    # Step 1: Simulate clinical scenario
    print("1. Simulating clinical acquisition scenarios...")
    datasets = validator.simulate_clinical_scenario(n_subjects=200)
    
    # Step 2: Compare methods
    print("2. Comparing methods...")
    results = validator.compare_methods(datasets)
    results['ground_truth'] = datasets['ground_truth']
    
    # Step 3: Calculate benefits
    print("3. Calculating scan time benefits...")
    performance = validator.calculate_scan_time_benefits(results)
    
    # Step 4: Report results
    print("\n=== RESULTS ===")
    for method, perf in performance.items():
        print(f"\n{method.upper()}:")
        print(f"  CBF RMSE: {perf['cbf_rmse']:.2f} mL/100g/min")
        print(f"  ATT RMSE: {perf['att_rmse']:.2f} ms")
        print(f"  Scan Time: {perf['scan_time_minutes']:.1f} minutes")
        print(f"  Efficiency Score: {perf['efficiency_score']:.4f}")
    
    return results, performance

if __name__ == "__main__":
    results, performance = run_clinical_validation()