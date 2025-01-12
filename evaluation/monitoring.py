import numpy as np
from typing import Dict
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import json
from collections import defaultdict

class PerformanceMonitor:
    """Monitor and track detailed performance metrics during training"""
    
    def __init__(self, 
                 trainer: 'EnhancedASLTrainer',
                 evaluator: 'EnhancedEvaluator',
                 log_dir: str = 'logs'):
        self.trainer = trainer
        self.evaluator = evaluator
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_history = defaultdict(list)
        self.grad_norms_history = defaultdict(list)
        self.uncertainty_history = defaultdict(list)
        
    def update(self, 
               epoch: int,
               stage: int,
               val_signals: np.ndarray,
               val_params: np.ndarray) -> Dict[str, float]:
        """Update monitoring metrics"""
        
        # Get predictions and uncertainties
        cbf_pred, att_pred, cbf_uncertainty, att_uncertainty = self.trainer.predict(val_signals)
        
        # Compute metrics for each ATT range
        metrics = {}
        for att_min, att_max, range_name in self.evaluator.att_ranges:
            mask = (val_params[:, 1] >= att_min) & (val_params[:, 1] < att_max)
            
            if np.any(mask):
                range_metrics = self.evaluator._compute_metrics(
                    cbf_pred[mask], att_pred[mask],
                    val_params[mask, 0], val_params[mask, 1],
                    cbf_uncertainty[mask], att_uncertainty[mask]
                )
                
                # Store metrics
                for key, value in range_metrics.items():
                    metric_name = f"{range_name}_{key}"
                    self.metrics_history[metric_name].append(value)
                    metrics[metric_name] = value
        
        # Track gradient norms
        for model_idx, model in enumerate(self.trainer.models):
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            self.grad_norms_history[f"model_{model_idx}"].append(total_norm)
        
        # Track mean uncertainties
        self.uncertainty_history['cbf'].append(np.mean(cbf_uncertainty))
        self.uncertainty_history['att'].append(np.mean(att_uncertainty))
        
        # Log metrics
        self._log_metrics(epoch, stage, metrics)
        
        return metrics
    
    def _log_metrics(self, epoch: int, stage: int, metrics: Dict[str, float]) -> None:
        """Log metrics to file"""
        log_file = self.log_dir / f'metrics_stage_{stage}.csv'
        
        metrics_df = pd.DataFrame([{
            'epoch': epoch,
            'stage': stage,
            **metrics
        }])
        
        if log_file.exists():
            metrics_df.to_csv(log_file, mode='a', header=False, index=False)
        else:
            metrics_df.to_csv(log_file, index=False)
    
    def plot_training_progress(self) -> None:
        """Plot detailed training progress"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot metrics history
        for range_name in ['Short ATT', 'Medium ATT', 'Long ATT']:
            axes[0,0].plot(self.metrics_history[f'{range_name}_mae_cbf'],
                         label=f'{range_name}')
        axes[0,0].set_title('CBF MAE by ATT Range')
        axes[0,0].set_xlabel('Update Step')
        axes[0,0].set_ylabel('MAE')
        axes[0,0].legend()
        
        # Plot gradient norms
        for model_idx in range(len(self.trainer.models)):
            axes[0,1].plot(self.grad_norms_history[f'model_{model_idx}'],
                         label=f'Model {model_idx}')
        axes[0,1].set_title('Gradient Norms')
        axes[0,1].set_xlabel('Update Step')
        axes[0,1].set_ylabel('Norm')
        axes[0,1].legend()
        
        # Plot uncertainty evolution
        axes[1,0].plot(self.uncertainty_history['cbf'], label='CBF')
        axes[1,0].plot(self.uncertainty_history['att'], label='ATT')
        axes[1,0].set_title('Mean Uncertainty Evolution')
        axes[1,0].set_xlabel('Update Step')
        axes[1,0].set_ylabel('Uncertainty')
        axes[1,0].legend()
        
        # Plot correlation between uncertainty and error
        corr_cbf = []
        corr_att = []
        for range_name in ['Short ATT', 'Medium ATT', 'Long ATT']:
            corr_cbf.append(np.mean([x for x in 
                self.metrics_history[f'{range_name}_uncertainty_correlation_cbf']
                if not np.isnan(x)]))
            corr_att.append(np.mean([x for x in 
                self.metrics_history[f'{range_name}_uncertainty_correlation_att']
                if not np.isnan(x)]))
        
        x = np.arange(3)
        width = 0.35
        axes[1,1].bar(x - width/2, corr_cbf, width, label='CBF')
        axes[1,1].bar(x + width/2, corr_att, width, label='ATT')
        axes[1,1].set_title('Uncertainty-Error Correlation')
        axes[1,1].set_xticks(x)
        axes[1,1].set_xticklabels(['Short', 'Medium', 'Long'])
        axes[1,1].legend()
        
        plt.tight_layout()
        plt.savefig(self.log_dir / 'training_progress.png')
        plt.close()
        
    def save_summary(self) -> None:
        """Save training summary and performance metrics"""
        
        summary = {
            'final_metrics': {
                metric: values[-1] 
                for metric, values in self.metrics_history.items()
            },
            'mean_grad_norms': {
                model_idx: np.mean(norms)
                for model_idx, norms in self.grad_norms_history.items()
            },
            'mean_uncertainties': {
                'cbf': np.mean(self.uncertainty_history['cbf']),
                'att': np.mean(self.uncertainty_history['att'])
            }
        }
        
        with open(self.log_dir / 'training_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
    def get_early_stopping_metrics(self) -> Dict[str, float]:
        """Get metrics used for early stopping decisions"""
        
        latest_metrics = {}
        
        # Get latest metrics for each ATT range
        for range_name in ['Short ATT', 'Medium ATT', 'Long ATT']:
            for metric in ['mae_cbf', 'mae_att']:
                key = f'{range_name}_{metric}'
                if self.metrics_history[key]:
                    latest_metrics[key] = self.metrics_history[key][-1]
        
        # Add gradient norm metrics
        latest_metrics['mean_grad_norm'] = np.mean([
            norms[-1] for norms in self.grad_norms_history.values()
            if norms
        ])
        
        return latest_metrics