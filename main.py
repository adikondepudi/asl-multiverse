import torch
import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

from enhanced_asl_network import EnhancedASLNet, CustomLoss
from asl_simulation import ASLSimulator, ASLParameters
from asl_trainer import EnhancedASLTrainer  # Changed from enhanced_training
from evaluation import EnhancedEvaluator, PerformanceMonitor

def train_enhanced_asl_model(config: dict = None, output_dir: str = 'results'):
    """Main training function with all enhancements"""
    
    if config is None:
        config = {
            'hidden_sizes': [256, 128, 64],
            'learning_rate': 0.001,
            'batch_size': 256,
            'n_samples': 20000,
            'n_epochs': 200,
            'n_ensembles': 5,
            'dropout_rate': 0.1,
            'norm_type': 'batch'
        }
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(output_dir) / f'enhanced_training_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save configuration
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4)
    
    # Initialize components
    simulator = ASLSimulator(ASLParameters(
        T1_artery=config.get('T1_artery', 1850),
        T2_factor=config.get('T2_factor', 1.0),
        alpha_BS1=config.get('alpha_BS1', 1.0),
        alpha_PCASL=config.get('alpha_PCASL', 0.85),
        alpha_VSASL=config.get('alpha_VSASL', 0.56),
    ))
    
    # Get PLDs from config or use default
    pld_range = config.get('pld_range', [500, 3000, 500])
    plds = np.arange(*pld_range)
    input_size = len(plds) * 2  # PCASL + VSASL signals
    
    trainer = EnhancedASLTrainer(
        model_class=EnhancedASLNet,
        input_size=input_size,
        hidden_sizes=config['hidden_sizes'],
        learning_rate=config['learning_rate'],
        batch_size=config['batch_size'],
        n_ensembles=config['n_ensembles'],
        device=device
    )
    
    evaluator = EnhancedEvaluator()
    monitor = PerformanceMonitor(trainer, evaluator, log_dir=output_dir/'logs')
    
    # Prepare curriculum data
    print("Preparing curriculum datasets...")
    train_loaders, val_loader = trainer.prepare_curriculum_data(
        simulator,
        n_samples=config['n_samples']
    )
    
    # Generate test data for final evaluation
    print("Generating test data...")
    test_data = {}
    for att_min, att_max, range_name in evaluator.att_ranges:
        n_test = 1000
        att_values = np.random.uniform(att_min, att_max, n_test)
        signals = simulator.generate_synthetic_data(plds, att_values, n_noise=50)
        params = np.column_stack((np.full_like(att_values, simulator.params.CBF), att_values))
        test_data[range_name] = (signals, params)
    
    # Train ensemble
    print("\nStarting enhanced training...")
    start_time = time.time()
    
    histories = trainer.train_ensemble(
        train_loaders,
        val_loader,
        n_epochs=config['n_epochs']
    )
    
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    
    # Evaluate on test data
    print("\nEvaluating model performance...")
    all_results = {}
    
    for range_name, (test_signals, test_params) in test_data.items():
        results = evaluator.evaluate_model(trainer, test_signals, test_params)
        all_results[range_name] = results
        
        print(f"\nResults for {range_name}:")
        for metric, value in results.items():
            print(f"{metric}: {value:.4f}")
    
    # Save results
    print("\nSaving results...")
    monitor.save_summary()
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_results = {
            k: {
                metric: float(value) if isinstance(value, (np.float32, np.float64))
                else value.tolist() if isinstance(value, np.ndarray)
                else value
                for metric, value in results.items()
            }
            for k, results in all_results.items()
        }
        json.dump(serializable_results, f, indent=4)
    
    # Save models
    model_dir = output_dir / 'models'
    model_dir.mkdir(exist_ok=True)
    for i, model in enumerate(trainer.models):
        torch.save(model.state_dict(), model_dir / f'model_{i}.pt')
    
    print(f"\nTraining completed. Results saved in {output_dir}")
    return trainer, evaluator, all_results

if __name__ == "__main__":
    train_enhanced_asl_model()