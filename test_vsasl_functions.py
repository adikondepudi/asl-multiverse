import numpy as np
import matplotlib.pyplot as plt
from vsasl_functions import fun_VSASL_1comp_vect_pep, fit_VSASL_vect_pep

def test_vsasl_signal():
    """Test VSASL signal generation against known MATLAB reference values"""
    # Test parameters from MATLAB example
    CBF = 60
    cbf = CBF/6000
    T1_artery = 1850
    T2_factor = 1
    alpha_BS1 = 1
    alpha_VSASL = 0.56
    True_ATT = 1600
    
    # PLDs from MATLAB example
    myPLDs = np.arange(500, 3001, 500)
    
    # Generate signal using our Python implementation
    beta = [cbf, True_ATT]
    python_signal = fun_VSASL_1comp_vect_pep(beta, myPLDs, T1_artery, T2_factor, 
                                            alpha_BS1, alpha_VSASL)
    
    # MATLAB reference values
    matlab_signal = np.array([0.0047, 0.0072, 0.0083, 0.0068, 0.0052, 0.0039])
    
    # Compare results
    print("\nVSASL Signal Generation Test")
    print("----------------------------")
    print("PLD (ms) | Python Signal | MATLAB Signal | Difference")
    print("-------------------------------------------------")
    for pld, py_sig, mat_sig in zip(myPLDs, python_signal, matlab_signal):
        print(f"{pld:7.0f} | {py_sig:12.4f} | {mat_sig:12.4f} | {abs(py_sig-mat_sig):10.4f}")
    
    # Calculate relative error
    rel_error = np.abs(python_signal - matlab_signal) / matlab_signal * 100
    print("\nMaximum relative error: {:.2f}%".format(np.max(rel_error)))
    
    return python_signal, matlab_signal

def test_vsasl_fitting():
    """Test VSASL fitting with synthetic data"""
    # Generate synthetic data
    CBF_true = 60/6000
    ATT_true = 1600
    T1_artery = 1850
    T2_factor = 1
    alpha_BS1 = 1
    alpha_VSASL = 0.56
    
    # Create PLDs and true signal
    PLDs = np.arange(500, 3001, 500)
    true_signal = fun_VSASL_1comp_vect_pep([CBF_true, ATT_true], PLDs, T1_artery, 
                                          T2_factor, alpha_BS1, alpha_VSASL)
    
    # Add noise
    np.random.seed(42)
    noise_level = 0.0002
    noisy_signal = true_signal + np.random.normal(0, noise_level, true_signal.shape)
    
    # Initial guess
    Init = [50/6000, 1500]  # From MATLAB example
    
    # Fit the noisy data
    beta, conintval, rmse, df = fit_VSASL_vect_pep(PLDs, noisy_signal, Init, T1_artery, 
                                                   T2_factor, alpha_BS1, alpha_VSASL)
    
    # Print results
    print("\nVSASL Fitting Test")
    print("------------------")
    print("Parameter      True Value    Fitted Value    95% CI")
    print("--------------------------------------------------")
    print("CBF (mL/g/s)  {:.6f}     {:.6f}      [{:.6f}, {:.6f}]".format(
        CBF_true, beta[0], conintval[0,0], conintval[0,1]))
    print("ATT (ms)      {:.1f}        {:.1f}         [{:.1f}, {:.1f}]".format(
        ATT_true, beta[1], conintval[1,0], conintval[1,1]))
    print("\nRMSE: {:.6f}".format(rmse))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(PLDs, noisy_signal, 'ko', label='Noisy Data')
    plt.plot(PLDs, true_signal, 'b-', label='True Signal')
    fitted_signal = fun_VSASL_1comp_vect_pep(beta, PLDs, T1_artery, T2_factor, 
                                            alpha_BS1, alpha_VSASL)
    plt.plot(PLDs, fitted_signal, 'r--', label='Fitted Signal')
    plt.xlabel('PLD (ms)')
    plt.ylabel('Signal')
    plt.legend()
    plt.title('VSASL Fitting Test')
    plt.grid(True)
    plt.show()
    
    return beta, conintval, rmse

if __name__ == "__main__":
    # Run all tests
    print("Running VSASL implementation tests...")
    
    # Test signal generation
    python_signal, matlab_signal = test_vsasl_signal()
    
    # Test fitting
    beta, conintval, rmse = test_vsasl_fitting()