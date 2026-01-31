# validation_metrics.py
"""
Comprehensive Validation Metrics for ASL Parameter Estimation.

This module provides statistical metrics for evaluating neural network
estimators against ground truth and baseline methods (e.g., least squares).

Metrics Implemented:
-------------------
1. Bland-Altman Analysis: Decomposes error into bias and limits of agreement
2. Intraclass Correlation Coefficient (ICC): Measures reliability
3. Concordance Correlation Coefficient (CCC): Composite of accuracy and precision
4. Structural Similarity Index (SSIM): For spatial map quality
5. Coefficient of Variation (CoV): Measures precision/repeatability
6. Win Rate: Percentage of cases where estimator A beats estimator B

References:
-----------
- Bland & Altman (1986): Statistical methods for assessing agreement
- McGraw & Wong (1996): Intraclass correlation coefficient
- Lin (1989): Concordance correlation coefficient
- Wang et al. (2004): Image quality assessment with SSIM

Author: Claude Code / ASL Multiverse Project
"""

import numpy as np
from typing import Dict, Tuple, Optional, List
from scipy import stats
import warnings


def bland_altman_analysis(predictions: np.ndarray,
                         ground_truth: np.ndarray,
                         confidence: float = 0.95) -> Dict[str, float]:
    """
    Perform Bland-Altman analysis to assess agreement.

    Decomposes the error into:
    - Systematic bias (mean difference)
    - Random error (limits of agreement)

    Args:
        predictions: (N,) predicted values
        ground_truth: (N,) true values
        confidence: Confidence level for limits of agreement (default 95%)

    Returns:
        Dict with:
        - bias: Mean difference (systematic error)
        - loa_lower: Lower limit of agreement
        - loa_upper: Upper limit of agreement
        - std_diff: Standard deviation of differences
        - ci_bias: Confidence interval for bias
        - proportional_bias: Slope of regression (0 = no proportional bias)
    """
    # Remove NaN values
    mask = ~np.isnan(predictions) & ~np.isnan(ground_truth)
    pred = predictions[mask]
    truth = ground_truth[mask]

    if len(pred) < 3:
        return {k: np.nan for k in ['bias', 'loa_lower', 'loa_upper', 'std_diff',
                                     'ci_bias_lower', 'ci_bias_upper', 'proportional_bias']}

    # Differences
    diff = pred - truth
    mean_values = (pred + truth) / 2

    # Bias (mean difference)
    bias = np.mean(diff)
    std_diff = np.std(diff, ddof=1)

    # Limits of agreement (typically 95%)
    z_score = stats.norm.ppf((1 + confidence) / 2)
    loa_lower = bias - z_score * std_diff
    loa_upper = bias + z_score * std_diff

    # Confidence interval for bias
    se_bias = std_diff / np.sqrt(len(diff))
    t_crit = stats.t.ppf((1 + confidence) / 2, len(diff) - 1)
    ci_bias_lower = bias - t_crit * se_bias
    ci_bias_upper = bias + t_crit * se_bias

    # Check for proportional bias (regression of diff on mean)
    # If slope != 0, bias varies with magnitude
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            slope, intercept, r, p, se = stats.linregress(mean_values, diff)
            proportional_bias = slope
        except:
            proportional_bias = np.nan

    return {
        'bias': bias,
        'loa_lower': loa_lower,
        'loa_upper': loa_upper,
        'std_diff': std_diff,
        'ci_bias_lower': ci_bias_lower,
        'ci_bias_upper': ci_bias_upper,
        'proportional_bias': proportional_bias,
    }


def intraclass_correlation(predictions: np.ndarray,
                          ground_truth: np.ndarray,
                          icc_type: str = 'ICC(3,1)') -> Dict[str, float]:
    """
    Compute Intraclass Correlation Coefficient.

    ICC measures the reliability of measurements - how consistent
    predictions are relative to ground truth.

    ICC Types:
    - ICC(1,1): One-way random effects (absolute agreement)
    - ICC(2,1): Two-way random effects (absolute agreement)
    - ICC(3,1): Two-way mixed effects (consistency) - RECOMMENDED for NN evaluation

    Args:
        predictions: (N,) predicted values
        ground_truth: (N,) true values
        icc_type: Type of ICC to compute

    Returns:
        Dict with ICC value and 95% confidence interval
    """
    # Remove NaN values
    mask = ~np.isnan(predictions) & ~np.isnan(ground_truth)
    pred = predictions[mask]
    truth = ground_truth[mask]

    if len(pred) < 3:
        return {'icc': np.nan, 'ci_lower': np.nan, 'ci_upper': np.nan}

    n = len(pred)

    # Arrange as two-column matrix (raters)
    Y = np.column_stack([truth, pred])
    k = 2  # Number of raters

    # Mean squares
    grand_mean = np.mean(Y)
    row_means = np.mean(Y, axis=1)
    col_means = np.mean(Y, axis=0)

    # Sum of squares
    ss_total = np.sum((Y - grand_mean) ** 2)
    ss_between = k * np.sum((row_means - grand_mean) ** 2)
    ss_within = np.sum((Y - row_means[:, np.newaxis]) ** 2)
    ss_columns = n * np.sum((col_means - grand_mean) ** 2)
    ss_error = ss_within - ss_columns

    # Mean squares
    ms_between = ss_between / (n - 1)
    ms_within = ss_within / (n * (k - 1))
    ms_error = ss_error / ((n - 1) * (k - 1)) if (n - 1) * (k - 1) > 0 else 0
    ms_columns = ss_columns / (k - 1) if k > 1 else 0

    # ICC calculation based on type
    if icc_type == 'ICC(1,1)':
        # One-way random: (MSB - MSW) / (MSB + (k-1)*MSW)
        icc = (ms_between - ms_within) / (ms_between + (k - 1) * ms_within)
    elif icc_type == 'ICC(2,1)':
        # Two-way random, absolute: (MSB - MSE) / (MSB + (k-1)*MSE + k*(MSC-MSE)/n)
        icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error +
                                          k * (ms_columns - ms_error) / n)
    else:  # ICC(3,1) - default
        # Two-way mixed, consistency: (MSB - MSE) / (MSB + (k-1)*MSE)
        icc = (ms_between - ms_error) / (ms_between + (k - 1) * ms_error)

    # Confidence intervals using F-distribution
    # (Simplified - exact CI is complex)
    f_value = ms_between / ms_error if ms_error > 0 else np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            f_lower = stats.f.ppf(0.025, n - 1, (n - 1) * (k - 1))
            f_upper = stats.f.ppf(0.975, n - 1, (n - 1) * (k - 1))

            ci_lower = (f_value / f_upper - 1) / (f_value / f_upper + k - 1)
            ci_upper = (f_value / f_lower - 1) / (f_value / f_lower + k - 1)
        except:
            ci_lower, ci_upper = np.nan, np.nan

    return {
        'icc': float(icc),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
    }


def concordance_correlation_coefficient(predictions: np.ndarray,
                                        ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Compute Lin's Concordance Correlation Coefficient.

    CCC = ρ_c = ρ × C_b

    Where:
    - ρ is Pearson correlation (precision/correlation)
    - C_b is bias correction factor (accuracy/deviation from 45° line)

    CCC penalizes both:
    1. Lack of correlation (scatter)
    2. Systematic deviation from the identity line

    Interpretation:
    - CCC > 0.99: Almost perfect
    - 0.95 < CCC < 0.99: Substantial
    - 0.90 < CCC < 0.95: Moderate
    - CCC < 0.90: Poor

    Args:
        predictions: (N,) predicted values
        ground_truth: (N,) true values

    Returns:
        Dict with CCC, Pearson r, bias correction factor Cb, and 95% CI
    """
    # Remove NaN values
    mask = ~np.isnan(predictions) & ~np.isnan(ground_truth)
    pred = predictions[mask]
    truth = ground_truth[mask]

    if len(pred) < 3:
        return {'ccc': np.nan, 'pearson_r': np.nan, 'cb': np.nan,
                'ci_lower': np.nan, 'ci_upper': np.nan}

    n = len(pred)

    # Means and variances
    mean_pred = np.mean(pred)
    mean_truth = np.mean(truth)
    var_pred = np.var(pred, ddof=1)
    var_truth = np.var(truth, ddof=1)
    sd_pred = np.sqrt(var_pred)
    sd_truth = np.sqrt(var_truth)

    # Pearson correlation
    cov = np.cov(pred, truth, ddof=1)[0, 1]
    pearson_r = cov / (sd_pred * sd_truth) if sd_pred * sd_truth > 0 else 0

    # Bias correction factor
    # Cb = 2 / ((σ_x/σ_y + σ_y/σ_x) + (μ_x - μ_y)²/(σ_x*σ_y))
    if sd_pred > 0 and sd_truth > 0:
        v = sd_pred / sd_truth
        u = (mean_pred - mean_truth) / np.sqrt(sd_pred * sd_truth)
        cb = 2 / (v + 1/v + u**2)
    else:
        cb = 0

    # CCC
    ccc = pearson_r * cb

    # Confidence interval using Fisher's z-transformation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            z = 0.5 * np.log((1 + ccc) / (1 - ccc + 1e-10))
            se_z = np.sqrt(1 / (n - 3))
            z_lower = z - 1.96 * se_z
            z_upper = z + 1.96 * se_z
            ci_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
            ci_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
        except:
            ci_lower, ci_upper = np.nan, np.nan

    return {
        'ccc': float(ccc),
        'pearson_r': float(pearson_r),
        'cb': float(cb),
        'ci_lower': float(ci_lower),
        'ci_upper': float(ci_upper),
    }


def structural_similarity_index(image1: np.ndarray,
                               image2: np.ndarray,
                               mask: np.ndarray = None,
                               k1: float = 0.01,
                               k2: float = 0.03) -> Dict[str, float]:
    """
    Compute Structural Similarity Index (SSIM) for 2D maps.

    SSIM combines three components:
    1. Luminance: Compare mean intensities
    2. Contrast: Compare variance/dynamic range
    3. Structure: Compare normalized patterns

    SSIM ∈ [-1, 1], where 1 = perfect structural similarity

    For ASL parameter maps, SSIM captures:
    - Whether spatial patterns are preserved
    - Whether tissue contrasts are maintained
    - Whether regional relationships are correct

    Args:
        image1: (H, W) First image (e.g., NN prediction)
        image2: (H, W) Second image (e.g., ground truth)
        mask: (H, W) Optional brain mask
        k1, k2: Stability constants

    Returns:
        Dict with SSIM and component scores (luminance, contrast, structure)
    """
    if mask is None:
        mask = np.ones_like(image1, dtype=bool)

    # Flatten masked regions
    x = image1[mask].astype(float)
    y = image2[mask].astype(float)

    if len(x) < 10:
        return {'ssim': np.nan, 'luminance': np.nan, 'contrast': np.nan, 'structure': np.nan}

    # Dynamic range
    L = max(x.max() - x.min(), y.max() - y.min()) + 1e-10
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    C3 = C2 / 2

    # Statistics
    mu_x, mu_y = np.mean(x), np.mean(y)
    sigma_x, sigma_y = np.std(x, ddof=1), np.std(y, ddof=1)
    sigma_xy = np.cov(x, y, ddof=1)[0, 1]

    # Component scores
    luminance = (2 * mu_x * mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
    contrast = (2 * sigma_x * sigma_y + C2) / (sigma_x**2 + sigma_y**2 + C2)
    structure = (sigma_xy + C3) / (sigma_x * sigma_y + C3)

    # Full SSIM
    ssim = luminance * contrast * structure

    return {
        'ssim': float(ssim),
        'luminance': float(luminance),
        'contrast': float(contrast),
        'structure': float(structure),
    }


def coefficient_of_variation(predictions: np.ndarray,
                            ground_truth: np.ndarray) -> Dict[str, float]:
    """
    Compute Coefficient of Variation metrics.

    CoV = σ / μ × 100%

    Reports:
    - CoV of predictions (precision of estimator)
    - CoV of errors (variability of error)
    - Within-subject CoV (if multiple measurements)

    Args:
        predictions: (N,) predicted values
        ground_truth: (N,) true values

    Returns:
        Dict with CoV metrics
    """
    # Remove NaN values
    mask = ~np.isnan(predictions) & ~np.isnan(ground_truth)
    pred = predictions[mask]
    truth = ground_truth[mask]

    if len(pred) < 3:
        return {'cov_pred': np.nan, 'cov_error': np.nan, 'cov_truth': np.nan}

    # CoV of predictions
    cov_pred = np.std(pred, ddof=1) / np.mean(np.abs(pred) + 1e-10) * 100

    # CoV of ground truth
    cov_truth = np.std(truth, ddof=1) / np.mean(np.abs(truth) + 1e-10) * 100

    # CoV of errors (RMSE / mean)
    errors = pred - truth
    rmse = np.sqrt(np.mean(errors ** 2))
    cov_error = rmse / np.mean(np.abs(truth) + 1e-10) * 100

    return {
        'cov_pred': float(cov_pred),
        'cov_truth': float(cov_truth),
        'cov_error': float(cov_error),
    }


def win_rate(pred_a: np.ndarray,
             pred_b: np.ndarray,
             ground_truth: np.ndarray,
             margin: float = 0.0) -> Dict[str, float]:
    """
    Compute win rate: percentage of cases where estimator A beats estimator B.

    A "wins" when |pred_a - truth| < |pred_b - truth| - margin

    Args:
        pred_a: (N,) Predictions from method A (e.g., neural network)
        pred_b: (N,) Predictions from method B (e.g., least squares)
        ground_truth: (N,) True values
        margin: Required margin for A to "win" (default 0 = any improvement)

    Returns:
        Dict with win rate, tie rate, lose rate for A vs B
    """
    # Remove NaN values (need valid data from both methods)
    mask = ~np.isnan(pred_a) & ~np.isnan(pred_b) & ~np.isnan(ground_truth)
    a = pred_a[mask]
    b = pred_b[mask]
    truth = ground_truth[mask]

    if len(a) == 0:
        return {'win_rate': np.nan, 'tie_rate': np.nan, 'lose_rate': np.nan, 'n_valid': 0}

    # Absolute errors
    err_a = np.abs(a - truth)
    err_b = np.abs(b - truth)

    # Count wins, ties, losses
    n = len(a)
    wins = np.sum(err_a < err_b - margin)
    ties = np.sum(np.abs(err_a - err_b) <= margin)
    losses = np.sum(err_a > err_b + margin)

    return {
        'win_rate': float(wins / n),
        'tie_rate': float(ties / n),
        'lose_rate': float(losses / n),
        'n_valid': int(n),
    }


def compute_all_metrics(nn_pred: np.ndarray,
                       ls_pred: np.ndarray,
                       ground_truth: np.ndarray,
                       parameter_name: str = 'CBF') -> Dict[str, Dict]:
    """
    Compute comprehensive metrics comparing NN to LS against ground truth.

    Args:
        nn_pred: (N,) Neural network predictions
        ls_pred: (N,) Least squares predictions
        ground_truth: (N,) Ground truth values
        parameter_name: Name for logging ('CBF' or 'ATT')

    Returns:
        Dict with all metrics for both methods
    """
    results = {
        'parameter': parameter_name,
        'neural_network': {},
        'least_squares': {},
        'comparison': {},
    }

    # Basic statistics
    mask = ~np.isnan(nn_pred) & ~np.isnan(ground_truth)
    nn_valid = nn_pred[mask]
    truth_valid = ground_truth[mask]

    results['neural_network']['mae'] = float(np.mean(np.abs(nn_valid - truth_valid)))
    results['neural_network']['rmse'] = float(np.sqrt(np.mean((nn_valid - truth_valid) ** 2)))
    results['neural_network']['r2'] = float(1 - np.var(nn_valid - truth_valid) / np.var(truth_valid))
    results['neural_network']['failure_rate'] = float(np.mean(np.isnan(nn_pred)))

    mask_ls = ~np.isnan(ls_pred) & ~np.isnan(ground_truth)
    ls_valid = ls_pred[mask_ls]
    truth_ls = ground_truth[mask_ls]

    if len(ls_valid) > 0:
        results['least_squares']['mae'] = float(np.mean(np.abs(ls_valid - truth_ls)))
        results['least_squares']['rmse'] = float(np.sqrt(np.mean((ls_valid - truth_ls) ** 2)))
        results['least_squares']['r2'] = float(1 - np.var(ls_valid - truth_ls) / np.var(truth_ls))
    else:
        results['least_squares']['mae'] = np.nan
        results['least_squares']['rmse'] = np.nan
        results['least_squares']['r2'] = np.nan
    results['least_squares']['failure_rate'] = float(np.mean(np.isnan(ls_pred)))

    # Detailed metrics for NN
    results['neural_network']['bland_altman'] = bland_altman_analysis(nn_pred, ground_truth)
    results['neural_network']['icc'] = intraclass_correlation(nn_pred, ground_truth)
    results['neural_network']['ccc'] = concordance_correlation_coefficient(nn_pred, ground_truth)
    results['neural_network']['cov'] = coefficient_of_variation(nn_pred, ground_truth)

    # Detailed metrics for LS
    results['least_squares']['bland_altman'] = bland_altman_analysis(ls_pred, ground_truth)
    results['least_squares']['icc'] = intraclass_correlation(ls_pred, ground_truth)
    results['least_squares']['ccc'] = concordance_correlation_coefficient(ls_pred, ground_truth)
    results['least_squares']['cov'] = coefficient_of_variation(ls_pred, ground_truth)

    # Comparison metrics
    results['comparison']['win_rate'] = win_rate(nn_pred, ls_pred, ground_truth)

    return results


def format_metrics_report(metrics: Dict) -> str:
    """
    Format metrics into a readable report.

    Args:
        metrics: Output from compute_all_metrics()

    Returns:
        Formatted string report
    """
    lines = []
    lines.append(f"\n{'='*60}")
    lines.append(f"VALIDATION REPORT: {metrics['parameter']}")
    lines.append(f"{'='*60}")

    for method in ['neural_network', 'least_squares']:
        m = metrics[method]
        lines.append(f"\n--- {method.upper().replace('_', ' ')} ---")
        lines.append(f"  MAE: {m['mae']:.4f}")
        lines.append(f"  RMSE: {m['rmse']:.4f}")
        lines.append(f"  R²: {m['r2']:.4f}")
        lines.append(f"  Failure Rate: {m['failure_rate']:.1%}")

        if 'ccc' in m and not np.isnan(m['ccc']['ccc']):
            lines.append(f"  CCC: {m['ccc']['ccc']:.4f} [{m['ccc']['ci_lower']:.4f}, {m['ccc']['ci_upper']:.4f}]")

        if 'icc' in m and not np.isnan(m['icc']['icc']):
            lines.append(f"  ICC: {m['icc']['icc']:.4f}")

        if 'bland_altman' in m and not np.isnan(m['bland_altman']['bias']):
            ba = m['bland_altman']
            lines.append(f"  Bland-Altman Bias: {ba['bias']:.4f} [{ba['loa_lower']:.4f}, {ba['loa_upper']:.4f}]")

    # Comparison
    comp = metrics['comparison']['win_rate']
    lines.append(f"\n--- NN vs LS COMPARISON ---")
    lines.append(f"  NN Win Rate: {comp['win_rate']:.1%}")
    lines.append(f"  Tie Rate: {comp['tie_rate']:.1%}")
    lines.append(f"  LS Win Rate: {comp['lose_rate']:.1%}")
    lines.append(f"  Valid Samples: {comp['n_valid']}")

    lines.append(f"\n{'='*60}\n")

    return '\n'.join(lines)
