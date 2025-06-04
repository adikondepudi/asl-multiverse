import numpy as np
import matplotlib.pyplot as plt # Kept for interactive plt.show()
from vsasl_functions import fun_VSASL_1comp_vect_pep, fit_VSASL_vect_pep # Assuming fit_VSASL_vectInit_pep is preferred for init

def test_vsasl_signal():
    CBF,cbf,T1_artery,T2_factor,alpha_BS1,alpha_VSASL,True_ATT = 60,60/6000,1850,1,1,0.56,1600
    myPLDs = np.arange(500, 3001, 500)
    python_signal = fun_VSASL_1comp_vect_pep([cbf,True_ATT], myPLDs, T1_artery, T2_factor, alpha_BS1, alpha_VSASL)
    matlab_signal = np.array([0.0047, 0.0072, 0.0083, 0.0068, 0.0052, 0.0039])
    print("\nVSASL Signal Generation Test"); print("----------------------------")
    print("PLD (ms) | Python Signal | MATLAB Signal | Difference"); print("-------------------------------------------------")
    for pld, py_sig, mat_sig in zip(myPLDs, python_signal, matlab_signal): print(f"{pld:7.0f} | {py_sig:12.4f} | {mat_sig:12.4f} | {abs(py_sig-mat_sig):10.4f}")
    rel_error = np.abs(python_signal - matlab_signal) / matlab_signal * 100
    print(f"\nMaximum relative error: {np.max(rel_error):.2f}%")
    return python_signal, matlab_signal

def test_vsasl_fitting():
    CBF_true,ATT_true,T1_artery,T2_factor,alpha_BS1,alpha_VSASL = 60/6000,1600,1850,1,1,0.56
    PLDs = np.arange(500, 3001, 500)
    true_signal = fun_VSASL_1comp_vect_pep([CBF_true,ATT_true], PLDs, T1_artery, T2_factor, alpha_BS1, alpha_VSASL)
    np.random.seed(42); noise_level = 0.0002
    noisy_signal = true_signal + np.random.normal(0, noise_level, true_signal.shape)
    Init = [50/6000, 1500]
    # Using fit_VSASL_vect_pep from vsasl_functions.py which internally calls fit_VSASL_vectInit_pep with default Init
    # Or, directly call fit_VSASL_vectInit_pep if Init is to be explicitly passed:
    from vsasl_functions import fit_VSASL_vectInit_pep
    beta, conintval, rmse, df = fit_VSASL_vectInit_pep(PLDs, noisy_signal, Init, T1_artery, T2_factor, alpha_BS1, alpha_VSASL)

    print("\nVSASL Fitting Test"); print("------------------")
    print("Parameter      True Value    Fitted Value    95% CI"); print("--------------------------------------------------")
    print(f"CBF (mL/g/s)  {CBF_true:.6f}     {beta[0]:.6f}      [{conintval[0,0]:.6f}, {conintval[0,1]:.6f}]")
    print(f"ATT (ms)      {ATT_true:.1f}        {beta[1]:.1f}         [{conintval[1,0]:.1f}, {conintval[1,1]:.1f}]")
    print(f"\nRMSE: {rmse:.6f}")
    
    plt.figure(figsize=(10, 6))
    plt.plot(PLDs, noisy_signal, 'ko', label='Noisy Data')
    plt.plot(PLDs, true_signal, 'b-', label='True Signal')
    fitted_signal = fun_VSASL_1comp_vect_pep(beta, PLDs, T1_artery, T2_factor, alpha_BS1, alpha_VSASL)
    plt.plot(PLDs, fitted_signal, 'r--', label='Fitted Signal')
    plt.xlabel('PLD (ms)'); plt.ylabel('Signal'); plt.legend(); plt.title('VSASL Fitting Test'); plt.grid(True)
    plt.show() # Kept for interactive test run
    return beta, conintval, rmse

if __name__ == "__main__":
    print("Running VSASL implementation tests...")
    python_signal, matlab_signal = test_vsasl_signal()
    beta, conintval, rmse = test_vsasl_fitting()
