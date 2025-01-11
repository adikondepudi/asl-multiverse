import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t

def fun_PCVSASL_misMatchPLD_vect_pep(beta, PLDTI, T1_artery, T_tau, T2_factor, 
                                    alpha_BS1, alpha_PCASL, alpha_VSASL):
    """
    Calculate combined PCASL and VSASL signal for different PLDs/TIs.
    
    Parameters
    ----------
    beta : array_like
        [CBF, ATT] parameters
    PLDTI : array_like
        Array of [PLD, TI] pairs
    T1_artery : float
        T1 of arterial blood
    T_tau : float
        Labeling duration
    T2_factor : float
        T2 decay factor
    alpha_BS1 : float
        Background suppression factor
    alpha_PCASL : float
        PCASL labeling efficiency
    alpha_VSASL : float
        VSASL labeling efficiency
        
    Returns
    -------
    diff_sig : ndarray
        Difference signals for PCASL and VSASL
    """
    CBF = beta[0]
    ATT = beta[1]
    
    # Split PLDTI into PLD and TI arrays
    PLD = PLDTI[:, 0]
    TI = PLDTI[:, 1]
    
    # Constants
    M0_b = 1
    lambda_blood = 0.90
    
    # Initialize output array [PCASL, VSASL]
    diff_sig = np.zeros((len(PLDTI), 2), dtype=float)
    
    # Scale factors
    alpha1 = alpha_PCASL * (alpha_BS1**4)  # PCASL
    alpha2 = alpha_VSASL * (alpha_BS1**3)  # VSASL
    
    # PCASL calculations
    index_1_p = ATT <= PLD
    index_0 = PLD < (ATT - T_tau)
    
    # PCASL: PLD >= ATT
    if np.any(index_1_p):
        diff_sig[index_1_p, 0] = (2 * M0_b * CBF * alpha1 / lambda_blood * T1_artery / 1000 *
                                 np.exp(-PLD[index_1_p]/T1_artery) * 
                                 (1 - np.exp(-T_tau/T1_artery)) * T2_factor)
    
    # PCASL: ATT-tau <= PLD < ATT
    mask = (~index_1_p) & (~index_0)
    if np.any(mask):
        diff_sig[mask, 0] = (2 * M0_b * CBF * alpha1 / lambda_blood * T1_artery / 1000 *
                            (np.exp(-ATT/T1_artery) - 
                             np.exp(-(T_tau + PLD[mask])/T1_artery)) * T2_factor)
    
    # PCASL: PLD < ATT - tau
    if np.any(index_0):
        diff_sig[index_0, 0] = 0
    
    # VSASL calculations
    index_1_v = ATT <= TI
    
    # VSASL: TI >= ATT
    if np.any(index_1_v):
        diff_sig[index_1_v, 1] = (2 * M0_b * CBF * alpha2 / lambda_blood * ATT / 1000 *
                                 np.exp(-TI[index_1_v]/T1_artery) * T2_factor)
    
    # VSASL: TI < ATT
    if np.any(~index_1_v):
        diff_sig[~index_1_v, 1] = (2 * M0_b * CBF * alpha2 / lambda_blood * TI[~index_1_v] / 1000 *
                                  np.exp(-TI[~index_1_v]/T1_artery) * T2_factor)
    
    return diff_sig

def fit_PCVSASL_misMatchPLD_vectInit_pep(PLDTI, diff_sig, Init, T1_artery, T_tau, T2_factor,
                                        alpha_BS1, alpha_PCASL, alpha_VSASL):
    """
    Fit combined PCASL and VSASL data to estimate CBF and ATT.
    
    Parameters
    ----------
    PLDTI : array_like
        Array of [PLD, TI] pairs
    diff_sig : array_like
        Measured difference signals for PCASL and VSASL
    Init : array_like
        Initial values for [CBF, ATT]
    T1_artery : float
        T1 of arterial blood
    T_tau : float
        Labeling duration
    T2_factor : float
        T2 decay factor
    alpha_BS1 : float
        Background suppression factor
    alpha_PCASL : float
        PCASL labeling efficiency
    alpha_VSASL : float
        VSASL labeling efficiency
        
    Returns
    -------
    beta : ndarray
        Fitted parameters [CBF, ATT]
    conintval : ndarray
        95% confidence intervals for parameters
    rmse : float
        Root mean square error of the fit
    df : int
        Degrees of freedom
    """
    # Set bounds for optimization
    bounds = ([1/6000, 100], [100/6000, 6000])
    
    # Define residual function for optimization
    def residuals(x):
        model_sig = fun_PCVSASL_misMatchPLD_vect_pep(x, PLDTI, T1_artery, T_tau, T2_factor,
                                                    alpha_BS1, alpha_PCASL, alpha_VSASL)
        return (model_sig - diff_sig).ravel()
    
    # Perform optimization
    result = least_squares(residuals, Init, bounds=bounds)
    beta = result.x
    
    # Calculate confidence intervals
    residual = result.fun
    jacobian = result.jac
    
    # Degrees of freedom
    n = len(residual)
    p = len(beta)
    df = n - p
    
    # Mean squared error
    mse = np.sum(residual**2) / df
    
    # Parameter covariance matrix
    pcov = np.linalg.inv(jacobian.T @ jacobian) * mse
    
    # Standard errors
    se = np.sqrt(np.diag(pcov))
    
    # 95% confidence intervals
    t_val = t.ppf(0.975, df)
    conintval = np.column_stack([beta - t_val*se, beta + t_val*se])
    
    # Calculate RMSE
    rmse = np.sqrt(np.sum(residual**2) / df)
    
    return beta, conintval, rmse, df