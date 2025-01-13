import numpy as np
from vsasl_functions import fun_VSASL_1comp_vect_pep, fit_VSASL_vect_pep
import matplotlib.pyplot as plt

def verify_vsasl_translation():
    """Verify Python implementation matches MATLAB reference values"""
    # Test parameters matching MATLAB example
    CBF = 60
    cbf = CBF/6000
    T1_artery = 1850
    T2_factor = 1
    alpha_BS1 = 1
    alpha_VSASL = 0.56
    True_ATT = 1600
    
    # PLDs matching MATLAB example
    myPLDs = np.arange(500, 3001, 500)
    
    # MATLAB reference values
    matlab_signal = np.array([0.0047, 0.0072, 0.0083, 0.0068, 0.0052, 0.0039])
    
    # Generate signal using Python implementation
    beta = [cbf, True_ATT]
    python_signal = fun_VSASL_1comp_vect_pep(beta, myPLDs, T1_artery, T2_factor, 
                                            alpha_BS1, alpha_VSASL)
    
    # Print comparison
    print("\nVSASL Signal Comparison")
    print("----------------------")
    print("PLD (ms) | Python Signal | MATLAB Signal | Difference")
    print("-------------------------------------------------")
    for pld, py_sig, mat_sig in zip(myPLDs, python_signal, matlab_signal):
        print(f"{pld:7.0f} | {py_sig:12.4f} | {mat_sig:12.4f} | {abs(py_sig-mat_sig):10.4f}")
    
    # Calculate relative error
    rel_error = np.abs(python_signal - matlab_signal) / matlab_signal * 100
    print(f"\nMaximum relative error: {np.max(rel_error):.2f}%")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(myPLDs, matlab_signal, 'bo-', label='MATLAB Reference')
    plt.plot(myPLDs, python_signal, 'rx--', label='Python Implementation')
    plt.xlabel('PLD (ms)')
    plt.ylabel('Signal')
    plt.title('VSASL Signal Comparison: MATLAB vs Python')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Now test the fitting function
    # Add some noise to the signal
    np.random.seed(42)
    noise_level = 0.0002
    noisy_signal = matlab_signal + np.random.normal(0, noise_level, matlab_signal.shape)
    
    # Try to recover the original parameters
    Init = [50/6000, 1500]  # Initial guess
    beta_fit, conintval, rmse, df = fit_VSASL_vect_pep(myPLDs, noisy_signal, Init,
                                                       T1_artery, T2_factor, 
                                                       alpha_BS1, alpha_VSASL)
    
    print("\nParameter Recovery Test")
    print("----------------------")
    print("Parameter      True Value    Fitted Value    95% CI")
    print("--------------------------------------------------")
    print(f"CBF (ml/g/s)  {cbf:.6f}     {beta_fit[0]:.6f}      [{conintval[0,0]:.6f}, {conintval[0,1]:.6f}]")
    print(f"ATT (ms)      {True_ATT:.1f}        {beta_fit[1]:.1f}         [{conintval[1,0]:.1f}, {conintval[1,1]:.1f}]")
    print(f"\nRMSE: {rmse:.6f}")
    
    # Plot fitting results
    fitted_signal = fun_VSASL_1comp_vect_pep(beta_fit, myPLDs, T1_artery, T2_factor, 
                                            alpha_BS1, alpha_VSASL)
    
    plt.figure(figsize=(10, 6))
    plt.plot(myPLDs, noisy_signal, 'ko', label='Noisy Data')
    plt.plot(myPLDs, matlab_signal, 'b-', label='True Signal')
    plt.plot(myPLDs, fitted_signal, 'r--', label='Fitted Signal')
    plt.xlabel('PLD (ms)')
    plt.ylabel('Signal')
    plt.title('VSASL Fitting Test')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    verify_vsasl_translation()