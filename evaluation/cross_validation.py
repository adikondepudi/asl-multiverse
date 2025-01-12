import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import KFold
from pathlib import Path
import json
from dataclasses import dataclass

@dataclass
class EnsembleResult:
    """Container for ensemble predictions and uncertainty"""
    cbf_mean: np.ndarray
    att_mean: np.ndarray
    cbf_std: np.ndarray
    att_std: np.ndarray
    cbf_uncertainty: np.ndarray
    att_uncertainty: np.ndarray
    model_weights: np.ndarray

class CrossValidator:
    """K-fold cross-validation with specialized ATT range evaluation"""
    
    def __init__(self, 
                 n_folds: int = 5,
                 att_ranges: List[Tuple[int, int, str]] = None):
        self.n_folds = n_folds
        self.att_ranges = att_ranges or [
            (500, 1500, "Short ATT"),
            (1500, 2500, "Medium ATT"),
            (2500, 4000, "Long ATT")
        ]
    
    def cross_validate(self,
                      trainer_class,
                      data: Tuple[np.ndarray, np.ndarray],
                      config: Dict) -> Dict[str, Dict[str, float]]:
        """Perform k-fold cross-validation with ATT range analysis"""
        
        X, y = data
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        
        # Results for each fold and ATT range
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nTraining fold {fold + 1}/{self.n_folds}")
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model
            trainer = trainer_class(**config)
            train_loader = trainer._create_dataloader(X_train, y_train)
            val_loader = trainer._create_dataloader(X_val, y_val)
            
            trainer.train(train_loader, val_loader)
            
            # Evaluate on validation set
            cbf_pred, att_pred, cbf_uncertainty, att_uncertainty = trainer.predict(X_val)
            
            # Compute metrics for each ATT range
            range_results = {}
            for att_min, att_max, range_name in self.att_ranges:
                mask = (y_val[:, 1] >= att_min) & (y_val[:, 1] < att_max)
                
                if np.any(mask):
                    metrics = self._compute_range_metrics(
                        cbf_pred[mask], att_pred[mask],
                        y_val[mask, 0], y_val[mask, 1],
                        cbf_uncertainty[mask], att_uncertainty[mask]
                    )
                    range_results[range_name] = metrics
            
            fold_results.append(range_results)
        
        # Aggregate results across folds
        aggregated_results = self._aggregate_fold_results(fold_results)
        return aggregated_results
    
    def _compute_range_metrics(self,
                             cbf_pred: np.ndarray,
                             att_pred: np.ndarray,
                             cbf_true: np.ndarray,
                             att_true: np.ndarray,
                             cbf_uncertainty: np.ndarray,
                             att_uncertainty: np.ndarray) -> Dict[str, float]:
        """Compute comprehensive metrics for an ATT range"""
        
        metrics = {
            'mae_cbf': np.mean(np.abs(cbf_pred - cbf_true)),
            'mae_att': np.mean(np.abs(att_pred - att_true)),
            'rmse_cbf': np.sqrt(np.mean((cbf_pred - cbf_true)**2)),
            'rmse_att': np.sqrt(np.mean((att_pred - att_true)**2)),
            'rel_error_cbf': np.mean(np.abs(cbf_pred - cbf_true) / cbf_true),
            'rel_error_att': np.mean(np.abs(att_pred - att_true) / att_true)
        }
        
        # Uncertainty calibration metrics
        metrics.update(self._compute_uncertainty_metrics(
            cbf_pred, att_pred,
            cbf_true, att_true,
            cbf_uncertainty, att_uncertainty
        ))
        
        return metrics
    
    def _compute_uncertainty_metrics(self,
                                  cbf_pred: np.ndarray,
                                  att_pred: np.ndarray,
                                  cbf_true: np.ndarray,
                                  att_true: np.ndarray,
                                  cbf_uncertainty: np.ndarray,
                                  att_uncertainty: np.ndarray) -> Dict[str, float]:
        """Compute uncertainty-related metrics"""
        
        # Correlation between error and uncertainty
        cbf_error = np.abs(cbf_pred - cbf_true)
        att_error = np.abs(att_pred - att_true)
        
        metrics = {
            'uncertainty_correlation_cbf': np.corrcoef(cbf_error, cbf_uncertainty)[0,1],
            'uncertainty_correlation_att': np.corrcoef(att_error, att_uncertainty)[0,1]
        }
        
        # Calibration metrics
        for q in [0.68, 0.95]:  # 1 and 2 sigma
            cbf_calib = np.mean(cbf_error <= cbf_uncertainty * q)
            att_calib = np.mean(att_error <= att_uncertainty * q)
            
            metrics.update({
                f'calibration_{int(q*100)}_cbf': cbf_calib,
                f'calibration_{int(q*100)}_att': att_calib
            })
        
        return metrics
    
    def _aggregate_fold_results(self, 
                              fold_results: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Aggregate results across folds"""
        aggregated = {}
        
        for range_name in fold_results[0].keys():
            range_metrics = {}
            
            # Get all metrics for this range
            for metric in fold_results[0][range_name].keys():
                values = [fold[range_name][metric] for fold in fold_results]
                
                range_metrics.update({
                    f"{metric}_mean": np.mean(values),
                    f"{metric}_std": np.std(values)
                })
            
            aggregated[range_name] = range_metrics
        
        return aggregated

class ModelEnsemble:
    """Enhanced model ensemble with weighted prediction combining"""
    
    def __init__(self,
                 models: List[nn.Module],
                 att_ranges: List[Tuple[int, int, str]] = None):
        self.models = models
        self.att_ranges = att_ranges or [
            (500, 1500, "Short ATT"),
            (1500, 2500, "Medium ATT"),
            (2500, 4000, "Long ATT")
        ]
        
        # Initialize model weights (will be updated based on performance)
        self.model_weights = np.ones(len(models)) / len(models)
        
    def update_weights(self,
                      val_data: Tuple[np.ndarray, np.ndarray]) -> None:
        """Update model weights based on validation performance"""
        X_val, y_val = val_data
        
        # Get predictions from each model
        predictions = []
        uncertainties = []
        
        for model in self.models:
            cbf_pred, att_pred, cbf_uncert, att_uncert = self._predict_single(model, X_val)
            predictions.append(np.column_stack((cbf_pred, att_pred)))
            uncertainties.append(np.column_stack((cbf_uncert, att_uncert)))
        
        # Compute errors for each model and ATT range
        weights = np.zeros_like(self.model_weights)
        
        for i, (pred, uncert) in enumerate(zip(predictions, uncertainties)):
            total_score = 0
            
            for att_min, att_max, _ in self.att_ranges:
                mask = (y_val[:, 1] >= att_min) & (y_val[:, 1] < att_max)
                
                if np.any(mask):
                    # Compute weighted error based on uncertainty
                    cbf_error = np.mean(np.abs(pred[mask, 0] - y_val[mask, 0]) / uncert[mask, 0])
                    att_error = np.mean(np.abs(pred[mask, 1] - y_val[mask, 1]) / uncert[mask, 1])
                    
                    # Higher weight for short ATT range
                    range_weight = 2.0 if att_min < 1500 else 1.0
                    range_score = -(cbf_error + att_error) * range_weight
                    total_score += range_score
            
            weights[i] = np.exp(total_score)  # Using softmax-like weighting
        
        # Normalize weights
        self.model_weights = weights / np.sum(weights)
    
    def predict(self, X: np.ndarray) -> EnsembleResult:
        """Make predictions using the weighted ensemble"""
        predictions = []
        uncertainties = []
        
        # Get predictions from each model
        for model in self.models:
            cbf_pred, att_pred, cbf_uncert, att_uncert = self._predict_single(model, X)
            predictions.append(np.column_stack((cbf_pred, att_pred)))
            uncertainties.append(np.column_stack((cbf_uncert, att_uncert)))
        
        predictions = np.array(predictions)  # (n_models, n_samples, 2)
        uncertainties = np.array(uncertainties)  # (n_models, n_samples, 2)
        
        # Compute weighted means
        weighted_pred = np.sum(predictions * self.model_weights[:, None, None], axis=0)
        
        # Compute total uncertainty (model uncertainty + prediction uncertainty)
        pred_variance = np.sum(uncertainties**2 * self.model_weights[:, None, None], axis=0)
        model_variance = np.sum(self.model_weights[:, None, None] * 
                              (predictions - weighted_pred[None, :, :])**2, axis=0)
        total_uncertainty = np.sqrt(pred_variance + model_variance)
        
        return EnsembleResult(
            cbf_mean=weighted_pred[:, 0],
            att_mean=weighted_pred[:, 1],
            cbf_std=np.std(predictions[:, :, 0], axis=0),
            att_std=np.std(predictions[:, :, 1], axis=0),
            cbf_uncertainty=total_uncertainty[:, 0],
            att_uncertainty=total_uncertainty[:, 1],
            model_weights=self.model_weights
        )
    
    def _predict_single(self, 
                       model: nn.Module,
                       X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get predictions from a single model"""
        model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(next(model.parameters()).device)
            cbf_mean, att_mean, cbf_log_var, att_log_var = model(X_tensor)
            
            return (cbf_mean.cpu().numpy(),
                   att_mean.cpu().numpy(),
                   np.exp(cbf_log_var.cpu().numpy() / 2),  # Convert to std
                   np.exp(att_log_var.cpu().numpy() / 2))  # Convert to std