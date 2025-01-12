import click
import yaml
import logging
from pathlib import Path
from typing import Dict

from main import train_enhanced_asl_model
from asl_simulation import ASLSimulator, ASLParameters
from evaluation import EnhancedEvaluator

def setup_logging(log_dir: str = 'logs') -> None:
    """Setup logging configuration"""
    log_dir = Path(log_dir)
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'asl_training.log'),
            logging.StreamHandler()
        ]
    )

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file"""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def flatten_config(config):
    """
    Flatten nested configuration dictionary to top-level keys.
    Example: {'training': {'batch_size': 256}} becomes {'batch_size': 256}
    """
    flattened = {}
    for section, params in config.items():
        if isinstance(params, dict):
            for key, value in params.items():
                flattened[key] = value
        else:
            flattened[section] = params
    return flattened

@click.group()
def cli():
    """ASL Analysis CLI"""
    pass

@cli.command()
@click.option('--config', type=str, default='config/default.yaml', 
              help='Path to configuration file')
@click.option('--output-dir', type=str, default='results',
              help='Directory for output files')
def train(config: str, output_dir: str):
    """Train ASL model"""
    setup_logging()
    config_dict = load_config(config)
    
    # Create flattened config for backward compatibility
    flattened_config = flatten_config(config_dict)
    
    logging.info(f"Starting training with config from {config}")
    trainer, evaluator, results = train_enhanced_asl_model(
        config=flattened_config,
        output_dir=output_dir
    )
    logging.info("Training completed")
@cli.command()
@click.option('--model-path', type=str, required=True,
              help='Path to trained model')
@click.option('--data-path', type=str, required=True,
              help='Path to test data')
@click.option('--output-dir', type=str, default='evaluation_results',
              help='Directory for evaluation results')
def evaluate(model_path: str, data_path: str, output_dir: str):
    """Evaluate trained model"""
    setup_logging()
    
    evaluator = EnhancedEvaluator()
    # Add evaluation logic here
    logging.info("Evaluation completed")

@cli.command()
@click.option('--config', type=str, default='config/default.yaml',
              help='Path to configuration file')
@click.option('--output-dir', type=str, default='test_results',
              help='Directory for test results')
def test(config: str, output_dir: str):
    """Run tests"""
    setup_logging()
    
    # Add test logic here
    logging.info("Tests completed")

if __name__ == '__main__':
    cli()