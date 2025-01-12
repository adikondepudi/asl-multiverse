import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    mae_cbf: float
    mae_att: float
    rmse_cbf: float
    rmse_att: float
    rel_error_cbf: float
    rel_error_att: float
    calibration_error_cbf: float
    calibration_error_att: float
    uncertainty_correlation_cbf: float
    uncertainty_correlation_att: float

class EnhancedEvaluator:
    """Enhanced evaluation with detailed performance analysis"""
    
    def __init__(self, att_ranges: List[Tuple[int, int, str]] = None):
        self.att_ranges = att_ranges or [
            (500, 1500, "Short ATT"),
            (1500, 2500, "Medium ATT"),
            (2500, 4000, "Long ATT")
        ]
        
    def evaluate_model(self,
                      trainer: 'EnhancedASLTrainer',
                      test_signals: np.ndarray,
                      true_params: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Comprehensive model evaluation"""
        
        # Get predictions with uncertainty
        cbf_pred, att_pred, cbf_uncertainty, att_uncertainty = trainer.predict(test_signals)
        
        # Overall results
        overall_results = self._compute_metrics(
            cbf_pred, att_pred,
            true_params[:, 0], true_params[:, 1],
            cbf_uncertainty, att_uncertainty
        )
        
        results = {'overall': overall_results}
        
        # Results for each ATT range
        for att_min, att_max, range_name in self.att_ranges:
            mask = (true_params[:, 1] >= att_min) & (true_params[:, 1] < att_max)
            
            range_results = self._compute_metrics(
                cbf_pred[mask], att_pred[mask],
                true_params[mask, 0], true_params[mask, 1],
                cbf_uncertainty[mask], att_uncertainty[mask]
            )
            
            results[range_name] = range_results
        
        return results
    
    def _compute_metrics(self,
                        cbf_pred: np.ndarray,
                        att_pred: np.ndarray,
                        cbf_true: np.ndarray,
                        att_true: np.ndarray,
                        cbf_uncertainty: np.ndarray,
                        att_uncertainty: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive evaluation metrics"""
        
        # Basic error metrics
        metrics = {
            'mae_cbf': mean_absolute_error(cbf_true, cbf_pred),
            'mae_att': mean_absolute_error(att_true, att_pred),
            'rmse_cbf': np.sqrt(mean_squared_error(cbf_true, cbf_pred)),
            'rmse_att': np.sqrt(mean_squared_error(att_true, att_pred))
        }
        
        # Relative errors
        metrics['rel_error_cbf'] = np.mean(np.abs(cbf_pred - cbf_true) / cbf_true)
        metrics['rel_error_att'] = np.mean(np.abs(att_pred - att_true) / att_true)
        
        # Uncertainty calibration
        metrics['calibration_error_cbf'] = self._compute_calibration_error(
            cbf_pred, cbf_true, cbf_uncertainty)
        metrics['calibration_error_att'] = self._compute_calibration_error(
            att_pred, att_true, att_uncertainty)
        
        # Correlation between uncertainty and error
        cbf_errors = np.abs(cbf_pred - cbf_true)
        att_errors = np.abs(att_pred - att_true)
        
        metrics['uncertainty_correlation_cbf'] = np.corrcoef(cbf_errors, cbf_uncertainty)[0,1]
        metrics['uncertainty_correlation_att'] = np.corrcoef(att_errors, att_uncertainty)[0,1]
        
        return metrics
    
    def _compute_calibration_error(self,
                                 pred: np.ndarray,
                                 true: np.ndarray,
                                 uncertainty: np.ndarray,
                                 n_bins: int = 10) -> float:
        """Compute uncertainty calibration error"""
        
        sorted_indices = np.argsort(uncertainty)
        bin_size = len(sorted_indices) // n_bins
        
        calibration_error = 0
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(sorted_indices)
            bin_indices = sorted_indices[start_idx:end_idx]
            
            # Compute empirical error rate
            errors = np.abs(pred[bin_indices] - true[bin_indices])
            empirical_error = np.mean(errors)
            
            # Compare with predicted uncertainty
            predicted_uncertainty = np.mean(uncertainty[bin_indices])
            calibration_error += np.abs(empirical_error - predicted_uncertainty)
            
        return calibration_error / n_bins
    
    def plot_error_analysis(self,
                          cbf_pred: np.ndarray,
                          att_pred: np.ndarray,
                          cbf_true: np.ndarray,
                          att_true: np.ndarray,
                          cbf_uncertainty: np.ndarray,
                          att_uncertainty: np.ndarray) -> None:
        """Generate comprehensive error analysis plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # CBF Error vs ATT
        axes[0,0].scatter(att_true, np.abs(cbf_pred - cbf_true), alpha=0.5)
        axes[0,0].set_xlabel('True ATT (ms)')
        axes[0,0].set_ylabel('CBF Absolute Error')
        axes[0,0].set_title('CBF Error vs ATT')
        
        # ATT Error vs ATT
        axes[0,1].scatter(att_true, np.abs(att_pred - att_true), alpha=0.5)
        axes[0,1].set_xlabel('True ATT (ms)')
        axes[0,1].set_ylabel('ATT Absolute Error')
        axes[0,1].set_title('ATT Error vs ATT')
        
        # CBF Uncertainty vs Error
        axes[0,2].scatter(cbf_uncertainty, np.abs(cbf_pred - cbf_true), alpha=0.5)
        axes[0,2].set_xlabel('Predicted CBF Uncertainty')
        axes[0,2].set_ylabel('CBF Absolute Error')
        axes[0,2].set_title('CBF Uncertainty Correlation')
        
        # ATT Uncertainty vs Error
        axes[1,0].scatter(att_uncertainty, np.abs(att_pred - att_true), alpha=0.5)
        axes[1,0].set_xlabel('Predicted ATT Uncertainty')
        axes[1,0].set_ylabel('ATT Absolute Error')
        axes[1,0].set_title('ATT Uncertainty Correlation')
        
        # CBF Calibration Plot
        self._plot_calibration(axes[1,1], cbf_pred, cbf_true, cbf_uncertainty, 'CBF')
        
        # ATT Calibration Plot
        self._plot_calibration(axes[1,2], att_pred, att_true, att_uncertainty, 'ATT')
        
        plt.tight_layout()
        plt.show()
    
    def _plot_calibration(self,
                         ax: plt.Axes,
                         pred: np.ndarray,
                         true: np.ndarray,
                         uncertainty: np.ndarray,
                         param_name: str,
                         n_bins: int = 10) -> None:
        """Plot uncertainty calibration"""
        
        sorted_indices = np.argsort(uncertainty)
        bin_size = len(sorted_indices) // n_bins
        
        empirical_errors = []
        predicted_uncertainties = []
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = start_idx + bin_size if i < n_bins - 1 else len(sorted_indices)
            bin_indices = sorted_indices[start_idx:end_idx]
            
            # Compute empirical error rate
            errors = np.abs(pred[bin_indices] - true[bin_indices])
            empirical_error = np.mean(errors)
            empirical_errors.append(empirical_error)
            
            # Get predicted uncertainty
            predicted_uncertainty = np.mean(uncertainty[bin_indices])
            predicted_uncertainties.append(predicted_uncertainty)
        
        # Plot calibration curve
        ax.plot([0, max(predicted_uncertainties)], [0, max(predicted_uncertainties)],
                'k--', label='Perfect Calibration')
        ax.scatter(predicted_uncertainties, empirical_errors, alpha=0.7,
                  label='Observed')
        ax.set_xlabel(f'Predicted {param_name} Uncertainty')
        ax.set_ylabel(f'Empirical {param_name} Error')
        ax.set_title(f'{param_name} Uncertainty Calibration')
        ax.legend()