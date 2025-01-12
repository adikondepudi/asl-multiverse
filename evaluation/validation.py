import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass

@dataclass
class ValidationMetrics:
    """Container for detailed validation metrics"""
    mae_cbf: float
    mae_att: float
    rmse_cbf: float
    rmse_att: float
    rel_error_cbf: float
    rel_error_att: float
    uncertainty_correlation_cbf: float
    uncertainty_correlation_att: float
    calibration_metrics: Dict[str, float]
    confidence_intervals: Dict[str, Tuple[float, float]]

class ValidationTesting:
    """Comprehensive validation and testing utilities"""
    
    def __init__(self, 
                 output_dir: str = 'validation_results',
                 att_ranges: List[Tuple[int, int, str]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.att_ranges = att_ranges or [
            (500, 1500, "Short ATT"),
            (1500, 2500, "Medium ATT"),
            (2500, 4000, "Long ATT")
        ]
    
    def validate_model(self,
                      ensemble: ModelEnsemble,
                      val_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, ValidationMetrics]:
        """Perform comprehensive model validation"""
        X_val, y_val = val_data
        results = {}
        
        # Get ensemble predictions
        ensemble_result = ensemble.predict(X_val)
        
        # Validate for each ATT range
        for att_min, att_max, range_name in self.att_ranges:
            mask = (y_val[:, 1] >= att_min) & (y_val[:, 1] < att_max)
            
            if np.any(mask):
                metrics = self._compute_validation_metrics(
                    ensemble_result.cbf_mean[mask],
                    ensemble_result.att_mean[mask],
                    y_val[mask, 0],
                    y_val[mask, 1],
                    ensemble_result.cbf_uncertainty[mask],
                    ensemble_result.att_uncertainty[mask]
                )
                results[range_name] = metrics
        
        self._plot_validation_results(ensemble_result, y_val, results)
        return results
    
    def _compute_validation_metrics(self,
                                  cbf_pred: np.ndarray,
                                  att_pred: np.ndarray,
                                  cbf_true: np.ndarray,
                                  att_true: np.ndarray,
                                  cbf_uncertainty: np.ndarray,
                                  att_uncertainty: np.ndarray) -> ValidationMetrics:
        """Compute comprehensive validation metrics"""
        
        # Basic error metrics
        metrics = ValidationMetrics(
            mae_cbf=np.mean(np.abs(cbf_pred - cbf_true)),
            mae_att=np.mean(np.abs(att_pred - att_true)),
            rmse_cbf=np.sqrt(np.mean((cbf_pred - cbf_true)**2)),
            rmse_att=np.sqrt(np.mean((att_pred - att_true)**2)),
            rel_error_cbf=np.mean(np.abs(cbf_pred - cbf_true) / cbf_true),
            rel_error_att=np.mean(np.abs(att_pred - att_true) / att_true),
            uncertainty_correlation_cbf=np.corrcoef(
                np.abs(cbf_pred - cbf_true), cbf_uncertainty)[0,1],
            uncertainty_correlation_att=np.corrcoef(
                np.abs(att_pred - att_true), att_uncertainty)[0,1],
            calibration_metrics={},
            confidence_intervals={}
        )
        
        # Calibration metrics
        for q in [0.68, 0.95]:  # 1 and 2 sigma
            cbf_calib = np.mean(np.abs(cbf_pred - cbf_true) <= cbf_uncertainty * q)
            att_calib = np.mean(np.abs(att_pred - att_true) <= att_uncertainty * q)
            
            metrics.calibration_metrics.update({
                f'calibration_{int(q*100)}_cbf': cbf_calib,
                f'calibration_{int(q*100)}_att': att_calib
            })
        
        # Confidence intervals
        for q in [0.95]:
            cbf_ci = np.percentile(cbf_pred + np.random.normal(0, cbf_uncertainty, size=1000), 
                                 [2.5, 97.5])
            att_ci = np.percentile(att_pred + np.random.normal(0, att_uncertainty, size=1000),
                                 [2.5, 97.5])
            
            metrics.confidence_intervals.update({
                'cbf': tuple(cbf_ci),
                'att': tuple(att_ci)
            })
        
        return metrics
    
    def _plot_validation_results(self,
                               ensemble_result: EnsembleResult,
                               true_params: np.ndarray,
                               metrics: Dict[str, ValidationMetrics]) -> None:
        """Generate comprehensive validation plots"""
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 3)
        
        # Prediction vs Truth plots
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(true_params[:, 0], ensemble_result.cbf_mean, alpha=0.5)
        ax1.plot([0, np.max(true_params[:, 0])], [0, np.max(true_params[:, 0])], 'r--')
        ax1.set_xlabel('True CBF')
        ax1.set_ylabel('Predicted CBF')
        ax1.set_title('CBF Predictions')
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.scatter(true_params[:, 1], ensemble_result.att_mean, alpha=0.5)
        ax2.plot([0, np.max(true_params[:, 1])], [0, np.max(true_params[:, 1])], 'r--')
        ax2.set_xlabel('True ATT')
        ax2.set_ylabel('Predicted ATT')
        ax2.set_title('ATT Predictions')
        
        # Error distribution plots
        ax3 = fig.add_subplot(gs[1, 0])
        sns.histplot(ensemble_result.cbf_mean - true_params[:, 0], ax=ax3)
        ax3.set_xlabel('CBF Error')
        ax3.set_title('CBF Error Distribution')
        
        ax4 = fig.add_subplot(gs[1, 1])
        sns.histplot(ensemble_result.att_mean - true_params[:, 1], ax=ax4)
        ax4.set_xlabel('ATT Error')
        ax4.set_title('ATT Error Distribution')
        
        # Uncertainty correlation plots
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.scatter(ensemble_result.cbf_uncertainty,
                   np.abs(ensemble_result.cbf_mean - true_params[:, 0]),
                   alpha=0.5)
        ax5.set_xlabel('CBF Uncertainty')
        ax5.set_ylabel('|CBF Error|')
        ax5.set_title('CBF Uncertainty Correlation')
        
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.scatter(ensemble_result.att_uncertainty,
                   np.abs(ensemble_result.att_mean - true_params[:, 1]),
                   alpha=0.5)
        ax6.set_xlabel('ATT Uncertainty')
        ax6.set_ylabel('|ATT Error|')
        ax6.set_title('ATT Uncertainty Correlation')
        
        # Performance by ATT range
        ax7 = fig.add_subplot(gs[:, 2])
        self._plot_range_performance(ax7, metrics)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'validation_results.png')
        plt.close()
    
    def _plot_range_performance(self,
                              ax: plt.Axes,
                              metrics: Dict[str, ValidationMetrics]) -> None:
        """Plot performance metrics by ATT range"""
        
        range_names = list(metrics.keys())
        metric_names = ['mae_cbf', 'mae_att', 'rmse_cbf', 'rmse_att']
        
        x = np.arange(len(range_names))
        width = 0.2
        
        for i, metric in enumerate(metric_names):
            values = [getattr(metrics[range_name], metric) for range_name in range_names]
            ax.bar(x + i * width, values, width, label=metric)
        
        ax.set_xlabel('ATT Range')
        ax.set_ylabel('Error')
        ax.set_title('Performance by ATT Range')
        ax.set_xticks(x + width * (len(metric_names) - 1) / 2)
        ax.set_xticklabels(range_names)
        ax.legend()

class FailureAnalyzer:
    """Analyze and categorize prediction failures"""
    
    def __init__(self, output_dir: str = 'failure_analysis'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_failures(self,
                        ensemble_result: EnsembleResult,
                        true_params: np.ndarray,
                        threshold_cbf: float = 10.0,  # ml/100g/min
                        threshold_att: float = 200.0  # ms
                        ) -> Dict[str, Dict[str, float]]:
        """Analyze prediction failures and their characteristics"""
        
        # Identify failures
        cbf_failures = np.abs(ensemble_result.cbf_mean - true_params[:, 0]) > threshold_cbf
        att_failures = np.abs(ensemble_result.att_mean - true_params[:, 1]) > threshold_att
        
        # Analyze failure cases
        failure_analysis = {
            'cbf': self._analyze_failure_cases(
                ensemble_result.cbf_mean[cbf_failures],
                true_params[cbf_failures, 0],
                ensemble_result.cbf_uncertainty[cbf_failures],
                'CBF'
            ),
            'att': self._analyze_failure_cases(
                ensemble_result.att_mean[att_failures],
                true_params[att_failures, 1],
                ensemble_result.att_uncertainty[att_failures],
                'ATT'
            )
        }
        
        # Plot failure analysis
        self._plot_failure_analysis(
            ensemble_result, true_params,
            cbf_failures, att_failures,
            threshold_cbf, threshold_att
        )
        
        return failure_analysis
    
    def _analyze_failure_cases(self,
                             pred: np.ndarray,
                             true: np.ndarray,
                             uncertainty: np.ndarray,
                             param_name: str) -> Dict[str, float]:
        """Analyze characteristics of failure cases"""
        
        if len(pred) == 0:
            return {'failure_rate': 0.0}
        
        analysis = {
            'failure_rate': len(pred) / len(true),
            'mean_error': np.mean(np.abs(pred - true)),
            'mean_uncertainty': np.mean(uncertainty),
            'uncertainty_correlation': np.corrcoef(
                np.abs(pred - true), uncertainty)[0,1],
            'systematic_bias': np.mean(pred - true),
            'error_std': np.std(pred - true)
        }
        
        return analysis
    
    def _plot_failure_analysis(self,
                             ensemble_result: EnsembleResult,
                             true_params: np.ndarray,
                             cbf_failures: np.ndarray,
                             att_failures: np.ndarray,
                             threshold_cbf: float,
                             threshold_att: float) -> None:
        """Generate comprehensive failure analysis plots"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot CBF failures
        axes[0,0].scatter(true_params[~cbf_failures, 0],
                         ensemble_result.cbf_mean[~cbf_failures],
                         alpha=0.5, label='Success')
        axes[0,0].scatter(true_params[cbf_failures, 0],
                         ensemble_result.cbf_mean[cbf_failures],
                         alpha=0.5, c='r', label='Failure')
        axes[0,0].plot([0, np.max(true_params[:, 0])],
                      [0, np.max(true_params[:, 0])], 'k--')
        axes[0,0].set_xlabel('True CBF')
        axes[0,0].set_ylabel('Predicted CBF')
        axes[0,0].set_title('CBF Predictions with Failures')
        axes[0,0].legend()
        
        # Plot ATT failures
        axes[0,1].scatter(true_params[~att_failures, 1],
                         ensemble_result.att_mean[~att_failures],
                         alpha=0.5, label='Success')
        axes[0,1].scatter(true_params[att_failures, 1],
                         ensemble_result.att_mean[att_failures],
                         alpha=0.5, c='r', label='Failure')
        axes[0,1].plot([0, np.max(true_params[:, 1])],
                      [0, np.max(true_params[:, 1])], 'k--')
        axes[0,1].set_xlabel('True ATT')
        axes[0,1].set_ylabel('Predicted ATT')
        axes[0,1].set_title('ATT Predictions with Failures')
        axes[0,1].legend()
        
        # Plot CBF uncertainty vs error
        axes[1,0].scatter(ensemble_result.cbf_uncertainty[~cbf_failures],
                         np.abs(ensemble_result.cbf_mean[~cbf_failures] - 
                               true_params[~cbf_failures, 0]),
                         alpha=0.5, label='Success')
        axes[1,0].scatter(ensemble_result.cbf_uncertainty[cbf_failures],
                         np.abs(ensemble_result.cbf_mean[cbf_failures] - 
                               true_params[cbf_failures, 0]),
                         alpha=0.5, c='r', label='Failure')
        axes[1,0].axhline(y=threshold_cbf, c='k', ls='--')
        axes[1,0].set_xlabel('CBF Uncertainty')
        axes[1,0].set_ylabel('|CBF Error|')
        axes[1,0].set_title('CBF Error vs Uncertainty')
        axes[1,0].legend()
        
        # Plot ATT uncertainty vs error
        axes[1,1].scatter(ensemble_result.att_uncertainty[~att_failures],
                         np.abs(ensemble_result.att_mean[~att_failures] - 
                               true_params[~att_failures, 1]),
                         alpha=0.5, label='Success')
        axes[1,1].scatter(ensemble_result.att_uncertainty[att_failures],
                         np.abs(ensemble_result.att_mean[att_failures] - 
                               true_params[att_failures, 1]),
                         alpha=0.5, c='r', label='Failure')
        axes[1,1].axhline(y=threshold_att, c='k', ls='--')
        axes[1,1].set_xlabel('ATT Uncertainty')
        axes[1,1].set_ylabel('|ATT Error|')
        axes[1,1].set_title('ATT Error vs Uncertainty')
        axes[1,1].legend()
        
        # Plot failure rates by ATT range
        att_bins = np.linspace(np.min(true_params[:, 1]),
                             np.max(true_params[:, 1]), 10)
        att_bin_centers = (att_bins[:-1] + att_bins[1:]) / 2
        
        att_failure_rates = []
        for i in range(len(att_bins)-1):
            mask = ((true_params[:, 1] >= att_bins[i]) & 
                   (true_params[:, 1] < att_bins[i+1]))
            if np.any(mask):
                att_failure_rates.append(np.mean(att_failures[mask]))
            else:
                att_failure_rates.append(0)
        
        axes[1,2].plot(att_bin_centers, att_failure_rates, 'o-')
        axes[1,2].set_xlabel('ATT Range')
        axes[1,2].set_ylabel('Failure Rate')
        axes[1,2].set_title('Failure Rate by ATT Range')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'failure_analysis.png')
        plt.close()
        
    def analyze_uncertainty(self,
                          ensemble_result: EnsembleResult,
                          true_params: np.ndarray) -> Dict[str, Dict[str, float]]:
        """Analyze uncertainty estimation quality"""
        
        uncertainty_analysis = {}
        
        # Analyze CBF uncertainty
        uncertainty_analysis['cbf'] = self._analyze_parameter_uncertainty(
            ensemble_result.cbf_mean,
            true_params[:, 0],
            ensemble_result.cbf_uncertainty,
            'CBF'
        )
        
        # Analyze ATT uncertainty
        uncertainty_analysis['att'] = self._analyze_parameter_uncertainty(
            ensemble_result.att_mean,
            true_params[:, 1],
            ensemble_result.att_uncertainty,
            'ATT'
        )
        
        # Plot uncertainty analysis
        self._plot_uncertainty_analysis(
            ensemble_result, true_params,
            uncertainty_analysis
        )
        
        return uncertainty_analysis
        
    def _analyze_parameter_uncertainty(self,
                                    pred: np.ndarray,
                                    true: np.ndarray,
                                    uncertainty: np.ndarray,
                                    param_name: str) -> Dict[str, float]:
        """Analyze uncertainty estimation quality for a parameter"""
        
        # Compute standardized errors
        z_scores = (pred - true) / uncertainty
        
        analysis = {
            'mean_uncertainty': np.mean(uncertainty),
            'uncertainty_std': np.std(uncertainty),
            'z_score_mean': np.mean(z_scores),
            'z_score_std': np.std(z_scores),
            'uncertainty_correlation': np.corrcoef(
                np.abs(pred - true), uncertainty)[0,1]
        }
        
        # Compute calibration metrics
        for p in [0.68, 0.95]:  # 1 and 2 sigma intervals
            coverage = np.mean(np.abs(z_scores) <= p)
            analysis[f'calibration_{int(p*100)}'] = coverage
        
        return analysis
    
    def _plot_uncertainty_analysis(self,
                                 ensemble_result: EnsembleResult,
                                 true_params: np.ndarray,
                                 uncertainty_analysis: Dict[str, Dict[str, float]]) -> None:
        """Plot uncertainty analysis results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        
        # Plot CBF uncertainty distribution
        sns.histplot(ensemble_result.cbf_uncertainty, ax=axes[0,0])
        axes[0,0].set_xlabel('CBF Uncertainty')
        axes[0,0].set_title('CBF Uncertainty Distribution')
        
        # Plot ATT uncertainty distribution
        sns.histplot(ensemble_result.att_uncertainty, ax=axes[0,1])
        axes[0,1].set_xlabel('ATT Uncertainty')
        axes[0,1].set_title('ATT Uncertainty Distribution')
        
        # Plot CBF standardized errors
        z_scores_cbf = ((ensemble_result.cbf_mean - true_params[:, 0]) / 
                       ensemble_result.cbf_uncertainty)
        sns.histplot(z_scores_cbf, ax=axes[1,0])
        axes[1,0].set_xlabel('CBF Standardized Error')
        axes[1,0].set_title('CBF Error Distribution')
        axes[1,0].axvline(x=-1, c='r', ls='--')
        axes[1,0].axvline(x=1, c='r', ls='--')
        
        # Plot ATT standardized errors
        z_scores_att = ((ensemble_result.att_mean - true_params[:, 1]) / 
                       ensemble_result.att_uncertainty)
        sns.histplot(z_scores_att, ax=axes[1,1])
        axes[1,1].set_xlabel('ATT Standardized Error')
        axes[1,1].set_title('ATT Error Distribution')
        axes[1,1].axvline(x=-1, c='r', ls='--')
        axes[1,1].axvline(x=1, c='r', ls='--')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'uncertainty_analysis.png')
        plt.close()