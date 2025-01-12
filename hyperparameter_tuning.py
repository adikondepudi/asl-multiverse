import itertools
from typing import Dict, List, Tuple
import numpy as np
from sklearn.model_selection import KFold
import torch
import json
from pathlib import Path

class HyperparameterTuner:
    """Hyperparameter optimization with grid search and cross-validation"""
    
    def __init__(self, base_dir: str = 'hyperparameter_tuning'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.param_grid = {
            'learning_rate': [0.0008, 0.0009, 0.001, 0.0011, 0.0012],
            'hidden_sizes': [
                [224, 128, 64],
                [256, 128, 64],
                [256, 160, 64]
            ],
            'dropout_rate': [0.05, 0.08, 0.1, 0.12, 0.15],
            'batch_size': [192, 224, 256, 288, 320]
        }
        
        self.results = []
    
    def generate_configs(self) -> List[Dict]:
        """Generate all possible hyperparameter combinations"""
        keys = self.param_grid.keys()
        values = self.param_grid.values()
        configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
        return configs
    
    def cross_validate_config(self,
                            config: Dict,
                            trainer_class,
                            train_data: Tuple[np.ndarray, np.ndarray],
                            n_folds: int = 5) -> Dict[str, float]:
        """Perform k-fold cross-validation for a configuration"""
        
        X, y = train_data
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_results = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Initialize trainer with current config
            trainer = trainer_class(
                input_size=X.shape[1],
                hidden_sizes=config['hidden_sizes'],
                learning_rate=config['learning_rate'],
                batch_size=config['batch_size']
            )
            
            # Train on this fold
            train_loader = trainer._create_dataloader(X_train, y_train)
            val_loader = trainer._create_dataloader(X_val, y_val)
            
            trainer.train(train_loader, val_loader, n_epochs=50)  # Reduced epochs for CV
            
            # Evaluate
            cbf_pred, att_pred, _, _ = trainer.predict(X_val)
            
            # Compute metrics
            metrics = {
                'mae_cbf': np.mean(np.abs(cbf_pred - y_val[:, 0])),
                'mae_att': np.mean(np.abs(att_pred - y_val[:, 1])),
                'rmse_cbf': np.sqrt(np.mean((cbf_pred - y_val[:, 0])**2)),
                'rmse_att': np.sqrt(np.mean((att_pred - y_val[:, 1])**2))
            }
            
            fold_results.append(metrics)
        
        # Average results across folds
        mean_results = {
            metric: np.mean([fold[metric] for fold in fold_results])
            for metric in fold_results[0].keys()
        }
        
        std_results = {
            f"{metric}_std": np.std([fold[metric] for fold in fold_results])
            for metric in fold_results[0].keys()
        }
        
        return {**mean_results, **std_results}
    
    def optimize(self,
                trainer_class,
                train_data: Tuple[np.ndarray, np.ndarray],
                n_folds: int = 5) -> Dict:
        """Perform full hyperparameter optimization"""
        
        configs = self.generate_configs()
        print(f"Testing {len(configs)} configurations...")
        
        for i, config in enumerate(configs):
            print(f"\nEvaluating configuration {i+1}/{len(configs)}")
            print("Config:", config)
            
            try:
                results = self.cross_validate_config(
                    config, trainer_class, train_data, n_folds)
                
                self.results.append({
                    'config': config,
                    'results': results,
                    'score': -(results['mae_cbf'] + results['mae_att'])  # Simple scoring
                })
                
                print("Results:", results)
                
            except Exception as e:
                print(f"Error evaluating configuration: {str(e)}")
                continue
        
        # Find best configuration
        best_result = max(self.results, key=lambda x: x['score'])
        
        # Save all results
        with open(self.base_dir / 'tuning_results.json', 'w') as f:
            json.dump({
                'all_results': self.results,
                'best_config': best_result['config'],
                'best_score': best_result['score']
            }, f, indent=4)
        
        return best_result['config']

class NormalizationTester:
    """Test different normalization strategies"""
    
    def __init__(self):
        self.norm_strategies = {
            'batch': lambda size: nn.BatchNorm1d(size),
            'layer': lambda size: nn.LayerNorm(size),
            'instance': lambda size: nn.InstanceNorm1d(size),
            'combined': lambda size: CombinedNorm(size)
        }
    
    def test_strategy(self,
                     strategy: str,
                     trainer_class,
                     train_data: Tuple[np.ndarray, np.ndarray],
                     val_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, float]:
        """Test a specific normalization strategy"""
        
        trainer = trainer_class(
            input_size=train_data[0].shape[1],
            norm_type=strategy
        )
        
        train_loader = trainer._create_dataloader(*train_data)
        val_loader = trainer._create_dataloader(*val_data)
        
        trainer.train(train_loader, val_loader)
        
        # Evaluate
        X_val, y_val = val_data
        cbf_pred, att_pred, _, _ = trainer.predict(X_val)
        
        return {
            'mae_cbf': np.mean(np.abs(cbf_pred - y_val[:, 0])),
            'mae_att': np.mean(np.abs(att_pred - y_val[:, 1]))
        }
    
    def compare_strategies(self,
                         trainer_class,
                         train_data: Tuple[np.ndarray, np.ndarray],
                         val_data: Tuple[np.ndarray, np.ndarray]) -> Dict[str, Dict[str, float]]:
        """Compare all normalization strategies"""
        
        results = {}
        for strategy in self.norm_strategies.keys():
            print(f"\nTesting {strategy} normalization...")
            results[strategy] = self.test_strategy(
                strategy, trainer_class, train_data, val_data)
        
        return results

class CombinedNorm(nn.Module):
    """Combined normalization approach"""
    
    def __init__(self, size: int):
        super().__init__()
        self.batch_norm = nn.BatchNorm1d(size)
        self.layer_norm = nn.LayerNorm(size)
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.batch_norm(x) + (1 - self.alpha) * self.layer_norm(x)