import unittest
import numpy as np
from scipy.stats import pearsonr
import os
import tempfile
import nibabel as nib
import matplotlib.pyplot as plt

from vsasl_functions import (fun_VSASL_1comp_vect_pep, fit_VSASL_vect_pep, 
                           fit_VSASL_vect_nopep, fit_VSASL_vectInit_pep)
from pcasl_functions import (fun_PCASL_1comp_vect_pep, fit_PCASL_vectInit_pep)
from multiverse_functions import (fun_PCVSASL_misMatchPLD_vect_pep, 
                                fit_PCVSASL_misMatchPLD_vectInit_pep)
from asl_pipeline import ASLDataProcessor
from data_loaders import MockLoader

class TestVSASLFunctions(unittest.TestCase):
    """Test VSASL functionality"""
    
    def setUp(self):
        """Set up test parameters"""
        self.CBF = 60
        self.cbf = self.CBF/6000
        self.T1_artery = 1850
        self.T2_factor = 1
        self.alpha_BS1 = 1
        self.alpha_VSASL = 0.56
        self.True_ATT = 1600
        self.PLDs = np.arange(500, 3001, 500)
        
        # MATLAB reference values
        self.matlab_signal = np.array([0.0047, 0.0072, 0.0083, 0.0068, 0.0052, 0.0039])
    
    def test_signal_generation(self):
        """Test VSASL signal generation against MATLAB reference"""
        beta = [self.cbf, self.True_ATT]
        python_signal = fun_VSASL_1comp_vect_pep(beta, self.PLDs, self.T1_artery,
                                                self.T2_factor, self.alpha_BS1, 
                                                self.alpha_VSASL)
        
        np.testing.assert_allclose(python_signal, self.matlab_signal, rtol=2e-2)
    
    def test_fitting_recovery(self):
        """Test parameter recovery from fitting"""
        # Generate synthetic data
        true_beta = [self.cbf, self.True_ATT]
        clean_signal = fun_VSASL_1comp_vect_pep(true_beta, self.PLDs, self.T1_artery,
                                               self.T2_factor, self.alpha_BS1, 
                                               self.alpha_VSASL)
        
        # Add noise
        np.random.seed(42)
        noise_level = 0.0002
        noisy_signal = clean_signal + np.random.normal(0, noise_level, clean_signal.shape)
        
        # Fit
        Init = [50/6000, 1500]
        beta, conintval, rmse, df = fit_VSASL_vectInit_pep(self.PLDs, noisy_signal, Init,
                                                       self.T1_artery, self.T2_factor,
                                                       self.alpha_BS1, self.alpha_VSASL)
        
        # Check recovery within 5%
        self.assertLess(abs(beta[0] - self.cbf)/self.cbf, 0.05)
        self.assertLess(abs(beta[1] - self.True_ATT)/self.True_ATT, 0.05)

class TestPCASLFunctions(unittest.TestCase):
    """Test PCASL functionality"""
    
    def setUp(self):
        """Set up test parameters"""
        self.CBF = 60
        self.cbf = self.CBF/6000
        self.T1_artery = 1850
        self.T_tau = 1800
        self.T2_factor = 1
        self.alpha_BS1 = 1
        self.alpha_PCASL = 0.85
        self.True_ATT = 1600
        self.PLDs = np.arange(500, 3001, 500)
    
    def test_fitting_recovery(self):
        """Test parameter recovery from fitting"""
        # Generate synthetic data
        true_beta = [self.cbf, self.True_ATT]
        clean_signal = fun_PCASL_1comp_vect_pep(true_beta, self.PLDs, self.T1_artery,
                                               self.T_tau, self.T2_factor, 
                                               self.alpha_BS1, self.alpha_PCASL)
        
        # Add noise
        np.random.seed(42)
        noise_level = 0.0002
        noisy_signal = clean_signal + np.random.normal(0, noise_level, clean_signal.shape)
        
        # Fit
        Init = [50/6000, 1500]
        beta, conintval, rmse, df = fit_PCASL_vectInit_pep(self.PLDs, noisy_signal, Init,
                                                           self.T1_artery, self.T_tau,
                                                           self.T2_factor, self.alpha_BS1,
                                                           self.alpha_PCASL)
        
        # Check recovery within 5%
        self.assertLess(abs(beta[0] - self.cbf)/self.cbf, 0.05)
        self.assertLess(abs(beta[1] - self.True_ATT)/self.True_ATT, 0.05)

class TestMULTIVERSEFunctions(unittest.TestCase):
    """Test MULTIVERSE functionality"""
    
class TestMULTIVERSEFunctions(unittest.TestCase):
    """Test MULTIVERSE functionality"""
    
    def setUp(self):
        """Set up test parameters and data"""
        # Parameters for MULTIVERSE tests
        self.CBF = 60
        self.cbf = self.CBF/6000
        self.T1_artery = 1850
        self.T_tau = 1800
        self.T2_factor = 1
        self.alpha_BS1 = 1
        self.alpha_PCASL = 0.85
        self.alpha_VSASL = 0.56
        self.True_ATT = 1600
        
        # Create matched PLD/TI pairs
        delays = np.arange(500, 3001, 500)
        self.PLDTI = np.column_stack((delays, delays))

        # Create synthetic data for testing
        nx, ny, nz = 64, 64, 20
        self.synthetic_data = np.random.normal(100, 10, (nx, ny, nz, 2))
        self.synthetic_data[:,:,:,1] = self.synthetic_data[:,:,:,0] - 2
        
        # Create mock loader with synthetic data
        mock_loader = MockLoader(self.synthetic_data)
        self.processor = ASLDataProcessor(data_loader=mock_loader)
        
        # Create temporary path for testing
        self.temp_dir = tempfile.mkdtemp()
        self.test_filepath = os.path.join(self.temp_dir, 'test_data.nii.gz')
        
    def test_fitting_recovery(self):
        """Test parameter recovery from fitting"""
        # Generate synthetic data
        true_beta = [self.cbf, self.True_ATT]
        clean_signal = fun_PCVSASL_misMatchPLD_vect_pep(true_beta, self.PLDTI,
                                                       self.T1_artery, self.T_tau,
                                                       self.T2_factor, self.alpha_BS1,
                                                       self.alpha_PCASL, self.alpha_VSASL)
        
        # Add noise
        np.random.seed(42)
        noise_level = 0.0002
        noisy_signal = clean_signal + np.random.normal(0, noise_level, clean_signal.shape)
        
        # Fit
        Init = [50/6000, 1500]
        beta, conintval, rmse, df = fit_PCVSASL_misMatchPLD_vectInit_pep(
            self.PLDTI, noisy_signal, Init, self.T1_artery, self.T_tau,
            self.T2_factor, self.alpha_BS1, self.alpha_PCASL, self.alpha_VSASL)
        
        # Check recovery within 5%
        self.assertLess(abs(beta[0] - self.cbf)/self.cbf, 0.05)
        self.assertLess(abs(beta[1] - self.True_ATT)/self.True_ATT, 0.05)
    
    def tearDown(self):
        """Clean up temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_data_loading(self):
        """Test data loading"""
        loaded_data = self.processor.load_data(self.test_filepath)
        np.testing.assert_array_equal(loaded_data, self.synthetic_data)
    
    def test_motion_correction(self):
        """Test motion correction"""
        self.processor.load_data(self.test_filepath)
        corrected_data = self.processor.motion_correction()
        
        # Check data shape preserved
        self.assertEqual(corrected_data.shape, self.synthetic_data.shape)
        
        # Check motion metrics reasonable
        metrics = self.processor.quality_control()
        self.assertIn('mean_motion', metrics)
        self.assertGreaterEqual(metrics['mean_motion'], 0)
    
    def test_perfusion_computation(self):
        """Test perfusion map computation"""
        self.processor.load_data(self.test_filepath)
        
        # Test each method
        for method in ['vsasl', 'pcasl', 'multiverse']:
            maps = self.processor.compute_perfusion_map(method=method)
            
            # Check outputs
            self.assertIn('CBF', maps)
            self.assertIn('ATT', maps)
            
            # Check map shapes
            self.assertEqual(maps['CBF'].shape, self.synthetic_data.shape[:3])
            self.assertEqual(maps['ATT'].shape, self.synthetic_data.shape[:3])
            
            # Check value ranges
            self.assertTrue(np.all(maps['CBF'] >= 0))
            self.assertTrue(np.all(maps['ATT'] >= 0))
    
    def test_quality_metrics(self):
        """Test quality control metrics"""
        self.processor.load_data(self.test_filepath)
        metrics = self.processor.quality_control()
        
        # Check essential metrics present
        self.assertIn('tSNR', metrics)
        self.assertIn('mean_motion', metrics)
        
        # Check metric values reasonable
        self.assertGreaterEqual(metrics['tSNR'], 0)
        self.assertGreaterEqual(metrics['mean_motion'], 0)

def run_synthetic_data_test(asl_type='vsasl', snr_levels=[5, 10, 20], 
                          att_values=[800, 1600, 2400]):
    """
    Run comprehensive synthetic data tests with varying SNR and ATT values.
    
    Parameters
    ----------
    asl_type : str
        Type of ASL sequence to test ('vsasl', 'pcasl', or 'multiverse')
    snr_levels : list
        List of SNR levels to test
    att_values : list
        List of ATT values to test
        
    Returns
    -------
    dict
        Dictionary containing test results with keys:
        - 'snr': SNR levels tested
        - 'att': ATT values tested
        - 'cbf_error': CBF percent errors
        - 'att_error': ATT percent errors
    """
    # Common parameters
    cbf = 60/6000
    T1_artery = 1850
    T2_factor = 1
    alpha_BS1 = 1
    
    results = {
        'snr': [],
        'att': [],
        'cbf_error': [],
        'att_error': []
    }
    
    for snr in snr_levels:
        for true_att in att_values:
            # Generate clean signal based on ASL type
            if asl_type == 'vsasl':
                alpha = 0.56
                PLDs = np.arange(500, 3001, 500)
                clean_signal = fun_VSASL_1comp_vect_pep([cbf, true_att], PLDs, 
                    T1_artery, T2_factor, alpha_BS1, alpha)
                noise_level = np.mean(np.abs(clean_signal)) / snr
                noisy_signal = clean_signal + np.random.normal(0, noise_level, clean_signal.shape)
                beta, _, _, _ = fit_VSASL_vect_pep(PLDs, noisy_signal, [50/6000, 1500],
                    T1_artery, T2_factor, alpha_BS1, alpha)
            
            elif asl_type == 'pcasl':
                alpha = 0.85
                T_tau = 1800
                PLDs = np.arange(500, 3001, 500)
                clean_signal = fun_PCASL_1comp_vect_pep([cbf, true_att], PLDs,
                    T1_artery, T_tau, T2_factor, alpha_BS1, alpha)
                noise_level = np.mean(np.abs(clean_signal)) / snr
                noisy_signal = clean_signal + np.random.normal(0, noise_level, clean_signal.shape)
                beta, _, _, _ = fit_PCASL_vectInit_pep(PLDs, noisy_signal, [50/6000, 1500],
                    T1_artery, T_tau, T2_factor, alpha_BS1, alpha)
            
            else:  # multiverse
                alpha_pcasl = 0.85
                alpha_vsasl = 0.56
                T_tau = 1800
                delays = np.arange(500, 3001, 500)
                PLDTI = np.column_stack((delays, delays))
                clean_signal = fun_PCVSASL_misMatchPLD_vect_pep([cbf, true_att], PLDTI,
                    T1_artery, T_tau, T2_factor, alpha_BS1, alpha_pcasl, alpha_vsasl)
                noise_level = np.mean(np.abs(clean_signal)) / snr
                noisy_signal = clean_signal + np.random.normal(0, noise_level, clean_signal.shape)
                beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(PLDTI, noisy_signal,
                    [50/6000, 1500], T1_artery, T_tau, T2_factor, alpha_BS1,
                    alpha_pcasl, alpha_vsasl)
            
            # Record results
            results['snr'].append(snr)
            results['att'].append(true_att)
            results['cbf_error'].append(abs(beta[0]*6000 - 60)/60 * 100)  # percent error
            results['att_error'].append(abs(beta[1] - true_att)/true_att * 100)  # percent error
    
    return results

def plot_synthetic_test_results(results):
    """
    Plot results from synthetic data testing.
    
    Parameters
    ----------
    results : dict
        Results dictionary from run_synthetic_data_test
    """
    import matplotlib.pyplot as plt
    
    # Convert lists to numpy arrays for easier manipulation
    snr = np.array(results['snr'])
    att = np.array(results['att'])
    cbf_error = np.array(results['cbf_error'])
    att_error = np.array(results['att_error'])
    
    # Create unique SNR and ATT values
    unique_snr = np.unique(snr)
    unique_att = np.unique(att)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot CBF errors
    for i, snr_val in enumerate(unique_snr):
        mask = snr == snr_val
        ax1.plot(att[mask], cbf_error[mask], 'o-', label=f'SNR={snr_val}')
    
    ax1.set_xlabel('True ATT (ms)')
    ax1.set_ylabel('CBF Error (%)')
    ax1.set_title('CBF Estimation Error vs ATT')
    ax1.grid(True)
    ax1.legend()
    
    # Plot ATT errors
    for i, snr_val in enumerate(unique_snr):
        mask = snr == snr_val
        ax2.plot(att[mask], att_error[mask], 'o-', label=f'SNR={snr_val}')
    
    ax2.set_xlabel('True ATT (ms)')
    ax2.set_ylabel('ATT Error (%)')
    ax2.set_title('ATT Estimation Error vs ATT')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    return fig

def main():
    """Run all synthetic data tests and plot results"""
    # Run tests for each method
    methods = ['vsasl', 'pcasl', 'multiverse']
    all_results = {}
    
    for method in methods:
        print(f"\nRunning synthetic tests for {method.upper()}...")
        results = run_synthetic_data_test(
            asl_type=method,
            snr_levels=[5, 10, 20],
            att_values=[800, 1600, 2400, 3200]
        )
        all_results[method] = results
        
        # Print summary statistics
        print(f"\n{method.upper()} Results Summary:")
        print("--------------------")
        print(f"Mean CBF Error: {np.mean(results['cbf_error']):.2f}%")
        print(f"Mean ATT Error: {np.mean(results['att_error']):.2f}%")
        print(f"Max CBF Error: {np.max(results['cbf_error']):.2f}%")
        print(f"Max ATT Error: {np.max(results['att_error']):.2f}%")
        
        # Plot results
        fig = plot_synthetic_test_results(results)
        fig.suptitle(f'{method.upper()} Performance')
        plt.show()

if __name__ == '__main__':
    # Run unit tests if run with -m unittest
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'test':
        unittest.main(argv=['dummy'])
    else:
        # Run synthetic data tests and plotting
        main()