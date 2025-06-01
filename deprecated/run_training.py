import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, OneCycleLR
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pickle
from datetime import datetime
import pandas as pd
import time
from collections import defaultdict
import itertools
import os

from asl_simulation import ASLSimulator, ASLParameters
from asl_trainer import ASLTrainer

class ResultManager:
    """Manages saving and visualization of training results"""
    def __init__(self, base_dir="results"):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.results_dir = Path(base_dir) / timestamp
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.plots_dir = self.results_dir / "plots"
        self.models_dir = self.results_dir / "models"
        self.data_dir = self.results_dir / "data"
        
        for directory in [self.plots_dir, self.models_dir, self.data_dir]:
            directory.mkdir(exist_ok=True)

    def save_all(self, config, trainer, results, config_id):
        """Save all results for a given configuration"""
        self.save_config(config, config_id)
        self.save_model(trainer.model, config_id)
        self.save_training_history(trainer.train_losses, trainer.val_losses, config_id)
        
        # Save predictions and metrics for each ATT range
        for att_range in results:
            if att_range != 'overall_score' and att_range != 'config':
                metrics = results[att_range]
                self.save_metrics(metrics, config_id, att_range)
                
    def save_training_history(self, train_losses, val_losses, config_id):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title('Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.plots_dir / f'training_history_{config_id}.png')
        plt.close()
        
    def save_metrics(self, metrics, config_id, att_range):
        metrics_file = self.data_dir / f'metrics_{config_id}_{att_range}.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=4)
            
    def save_model(self, model, config_id):
        torch.save(model.state_dict(), self.models_dir / f'model_{config_id}.pt')
        
    def save_config(self, config, config_id):
        config_file = self.data_dir / f'config_{config_id}.json'
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=4)

def generate_configs():
    """Generate different model configurations to try"""
    base_configs = {
        'hidden_sizes': [
            [64, 32],         # Simple
            [128, 64, 32],    # Medium
            [256, 128, 64],   # Complex
        ],
        'activation': ['relu', 'leaky_relu'],  # Most common choices
        'use_batch_norm': [True],  # Usually helpful
        'dropout_rate': [0.1],     # Standard choice
        'learning_rate': [1e-3, 1e-4],
        'batch_size': [256],       # Good balance
        'n_samples': [5000],
        'epochs': [100],
        'scheduler': ['plateau'],   # Most reliable choice
        'use_curriculum': [True],   # Usually beneficial
    }
    
    # Generate combinations
    keys = base_configs.keys()
    values = base_configs.values()
    configs = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    return configs

def train_with_config(config, simulator, plds, result_manager):
    """Train model with given configuration and save results"""
    model = ASLTrainer(
        input_size=len(plds) * 2,
        hidden_sizes=config['hidden_sizes'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size']
    )
    
    # Prepare data
    train_loader, val_loader = model.prepare_data(
        simulator, 
        n_samples=config['n_samples'],
        val_split=0.2
    )
    
    # Train with curriculum if specified
    if config['use_curriculum']:
        # Phase 1: Middle range
        model.train(train_loader, val_loader, 
                   n_epochs=config['epochs']//2,
                   early_stopping_patience=10)
        
        # Phase 2: Full range
        model.train(train_loader, val_loader,
                   n_epochs=config['epochs'],
                   early_stopping_patience=15)
    else:
        model.train(train_loader, val_loader,
                   n_epochs=config['epochs'],
                   early_stopping_patience=15)
    
    # Evaluate across different ATT ranges
    results = evaluate_model(model, simulator, plds)
    
    return results, model

def evaluate_model(trainer, simulator, plds):
    """Evaluate model performance"""
    att_ranges = [
        (500, 1500, "Short ATT"),
        (1500, 2500, "Medium ATT"),
        (2500, 4000, "Long ATT")
    ]
    
    results = {}
    
    for att_min, att_max, range_name in att_ranges:
        att_values = np.linspace(att_min, att_max, 10)
        test_signals = simulator.generate_synthetic_data(plds, att_values, n_noise=50)
        
        X_test = np.zeros((50 * len(att_values), len(plds) * 2))
        for i in range(50):
            start_idx = i * len(att_values)
            end_idx = (i + 1) * len(att_values)
            X_test[start_idx:end_idx, :len(plds)] = test_signals['PCASL'][i]
            X_test[start_idx:end_idx, len(plds):] = test_signals['VSASL'][i]
        
        y_test = np.column_stack((
            np.full(50 * len(att_values), simulator.params.CBF),
            np.tile(att_values, 50)
        ))
        
        predictions = trainer.predict(X_test)
        metrics = calculate_metrics(predictions, y_test)
        results[range_name] = metrics
    
    results['overall_score'] = calculate_overall_score(results)
    return results

def calculate_metrics(predictions, true_values):
    """Calculate performance metrics"""
    metrics = {
        'MAE_CBF': np.mean(np.abs(predictions[:,0] - true_values[:,0])),
        'MAE_ATT': np.mean(np.abs(predictions[:,1] - true_values[:,1])),
        'RMSE_CBF': np.sqrt(np.mean((predictions[:,0] - true_values[:,0])**2)),
        'RMSE_ATT': np.sqrt(np.mean((predictions[:,1] - true_values[:,1])**2))
    }
    
    # Calculate relative errors with protection against division by zero
    mask = true_values[:,0] != 0
    metrics['RelError_CBF'] = np.mean(np.abs(predictions[mask,0] - true_values[mask,0]) / true_values[mask,0])
    
    mask = true_values[:,1] != 0
    metrics['RelError_ATT'] = np.mean(np.abs(predictions[mask,1] - true_values[mask,1]) / true_values[mask,1])
    
    return metrics

def calculate_overall_score(results):
    """Calculate overall score from all metrics"""
    score = 0
    for range_name in ["Short ATT", "Medium ATT", "Long ATT"]:
        metrics = results[range_name]
        # Normalize and combine metrics
        score += -metrics['MAE_CBF']/60 - metrics['MAE_ATT']/4000
    return score / 3

def main():
    print("Starting comprehensive ASL training exploration...")
    start_time = time.time()
    
    # Setup
    plds = np.arange(500, 3001, 500)
    simulator = ASLSimulator(ASLParameters())
    result_manager = ResultManager()
    
    # Generate configurations
    configs = generate_configs()
    print(f"Generated {len(configs)} configurations to test")
    
    # Store results
    all_results = []
    
    # Test each configuration
    for i, config in enumerate(configs, 1):
        print(f"\nTesting configuration {i}/{len(configs)}")
        print("Configuration:", config)
        
        try:
            config_id = f'config_{i:03d}'
            results, trainer = train_with_config(config, simulator, plds, result_manager)
            
            # Save results
            results['config'] = config
            result_manager.save_all(config, trainer, results, config_id)
            all_results.append(results)
            
            # Print current results
            print(f"\nResults for configuration {i}:")
            for att_range in ["Short ATT", "Medium ATT", "Long ATT"]:
                print(f"\n{att_range}:")
                for metric, value in results[att_range].items():
                    print(f"{metric}: {value:.4f}")
            print(f"Overall Score: {results['overall_score']:.4f}")
            
        except Exception as e:
            print(f"Error with configuration {i}:", str(e))
            continue
    
    # Find and print best configurations
    best_overall = max(all_results, key=lambda x: x['overall_score'])
    print("\nBest Overall Configuration:")
    print(best_overall['config'])
    print(f"Score: {best_overall['overall_score']:.4f}")
    
    print(f"\nTotal exploration time: {(time.time() - start_time)/3600:.2f} hours")

if __name__ == "__main__":
    main()