import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t

def fun_PCASL_1comp_vect_pep(beta, PLD, T1_artery, T_tau, T2_factor, alpha_BS1, alpha_PCASL):
    """
    Calculate the PCASL general kinetic curve with consideration of T2 factor and BS suppression.
    """
    CBF = beta[0]
    ATT = beta[1]
    
    # Constants
    M0_b = 1
    lambda_blood = 0.90
    
    # Convert PLD to numpy array if not already
    PLD = np.asarray(PLD)
    
    # Initialize output array
    diff_sig = np.zeros_like(PLD, dtype=float)
    
    # PCASL scale factor, 4 BS
    alpha1 = alpha_PCASL * (alpha_BS1**4)
    
    # Calculate indices for different conditions
    index_0 = PLD < (ATT - T_tau)
    index_1 = (PLD < ATT) & (PLD >= (ATT - T_tau))
    index_2 = PLD >= ATT
    
    # Calculate signal for each condition
    if np.any(index_0):
        diff_sig[index_0] = 0
        
    if np.any(index_1):
        diff_sig[index_1] = (2 * M0_b * CBF * alpha1 / lambda_blood * T1_artery / 1000 * 
                            (np.exp(-ATT/T1_artery) - 
                             np.exp(-(T_tau + PLD[index_1])/T1_artery)) * T2_factor)
        
    if np.any(index_2):
        diff_sig[index_2] = (2 * M0_b * CBF * alpha1 / lambda_blood * T1_artery / 1000 * 
                            np.exp(-PLD[index_2]/T1_artery) * 
                            (1 - np.exp(-T_tau/T1_artery)) * T2_factor)
    
    return diff_sig

def fit_PCASL_vectInit_pep(PLD, diff_sig, Init, T1_artery, T_tau, T2_factor, alpha_BS1, alpha_PCASL):
    """
    Fit PCASL data to estimate CBF and ATT.
    
    Parameters
    ----------
    PLD : array_like
        Post-labeling delays in ms
    diff_sig : array_like
        Measured difference signals
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
        return (fun_PCASL_1comp_vect_pep(x, PLD, T1_artery, T_tau, T2_factor, 
                                        alpha_BS1, alpha_PCASL) - diff_sig)
    
    # Perform optimization with more robust settings
    result = least_squares(residuals, Init, bounds=bounds, method='trf', 
                         ftol=1e-8, xtol=1e-8, gtol=1e-8)
    beta = result.x
    
    # Calculate confidence intervals
    residual = result.fun
    jacobian = result.jac
    
    # Degrees of freedom
    n = len(diff_sig)
    p = len(beta)
    df = n - p
    
    # Mean squared error
    mse = np.sum(residual**2) / df
    
    try:
        # Add small regularization term for numerical stability
        reg_term = 1e-10 * np.eye(jacobian.shape[1])
        # Parameter covariance matrix with regularization
        pcov = np.linalg.inv(jacobian.T @ jacobian + reg_term) * mse
        
        # Standard errors
        se = np.sqrt(np.diag(pcov))
        
        # 95% confidence intervals
        t_val = t.ppf(0.975, df)
        conintval = np.column_stack([beta - t_val*se, beta + t_val*se])
    except np.linalg.LinAlgError:
        # If matrix is still singular, return wide confidence intervals
        conintval = np.array([[0, 100/6000], [0, 6000]])
    
    # Calculate RMSE
    rmse = np.sqrt(np.sum(residual**2) / df)
    
    return beta, conintval, rmse, df