import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t

def fun_VSASL_1comp_vect_pep(beta, PLD, T1_artery, T2_factor, alpha_BS1, alpha_VSASL):
    """
    Function to calculate the VSASL signal with consideration of T2 factor and BS suppression.
    
    Parameters
    ----------
    beta : array_like
        [CBF, ATT] parameters to fit
    PLD : array_like
        Post-labeling delays in ms
    T1_artery : float
        T1 of arterial blood in ms
    T2_factor : float
        T2 decay factor
    alpha_BS1 : float
        Background suppression factor
    alpha_VSASL : float
        VSASL labeling efficiency
        
    Returns
    -------
    diff_sig : ndarray
        Difference signal for each PLD
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
    
    # VSASL scale factor
    alpha2 = alpha_VSASL * (alpha_BS1**3)
    
    # Calculate indices for different conditions
    index_1 = PLD <= ATT
    
    # Calculate signal for PLD >= ATT
    if np.any(~index_1):
        diff_sig[~index_1] = (2 * M0_b * CBF * alpha2 / lambda_blood * ATT / 1000 * 
                             np.exp(-PLD[~index_1]/T1_artery) * T2_factor)
    
    # Calculate signal for PLD < ATT
    if np.any(index_1):
        diff_sig[index_1] = (2 * M0_b * CBF * alpha2 / lambda_blood * PLD[index_1] / 1000 * 
                            np.exp(-PLD[index_1]/T1_artery) * T2_factor)
    
    return diff_sig

def fit_VSASL_vect_pep(PLD, diff_sig, Init, T1_artery, T2_factor, alpha_BS1, alpha_VSASL):
    """
    Fit VSASL data to estimate CBF and ATT.
    
    Parameters
    ----------
    PLD : array_like
        Post-labeling delays in ms
    diff_sig : array_like
        Measured difference signals
    Init : array_like
        Initial values for [CBF, ATT] in format [CBF(ml/g/s), ATT(ms)]
    T1_artery : float
        T1 of arterial blood in ms
    T2_factor : float
        T2 decay factor
    alpha_BS1 : float
        Background suppression factor
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
        return (fun_VSASL_1comp_vect_pep(x, PLD, T1_artery, T2_factor, alpha_BS1, alpha_VSASL) - 
                diff_sig)
    
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


def fit_VSASL_vect_nopep(PLD, diff_sig):
    """
    Fit VSASL data to estimate CBF and ATT without external parameters (nopep version).
    Uses default values for T1_artery, T2_factor, alpha_BS1, alpha_VSASL.
    
    Parameters
    ----------
    PLD : array_like
        Post-labeling delays in ms
    diff_sig : array_like
        Measured difference signals
        
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
    # Default parameters as in MATLAB code
    T1_artery = 1800
    T2_factor = 1
    alpha_BS1 = 1
    alpha_VSASL = 0.56
    
    # Initial values
    Init = [50/6000, 1500]
    
    # Use the main fitting function with default parameters
    return fit_VSASL_vect_pep(PLD, diff_sig, Init, T1_artery, T2_factor, 
                             alpha_BS1, alpha_VSASL)

def fit_VSASL_vectInit_pep(PLD, diff_sig, Init, T1_artery, T2_factor, alpha_BS1, alpha_VSASL):
    """
    Fit VSASL data to estimate CBF and ATT using provided initial values.
    Same as fit_VSASL_vect_pep but with explicit initialization.
    """
    return fit_VSASL_vect_pep(PLD, diff_sig, Init, T1_artery, T2_factor, alpha_BS1, alpha_VSASL)
