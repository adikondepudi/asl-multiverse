#!/usr/bin/env python3
"""
Spatial Model Validation Script
===============================
Evaluates SpatialASLNet (U-Net) models on 2D phantom data.

Usage:
    python validate_spatial.py --run_dir <path_to_trained_model> --data_dir <path_to_test_data>

This script:
1. Loads trained SpatialASLNet ensemble
2. Evaluates on held-out spatial test data
3. Computes metrics: MAE, RMSE, bias for CBF and ATT maps
4. Generates visualization plots
"""

import sys
import os
import json
import logging
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

# Suppress warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class SpatialValidator:
    """Validator for SpatialASLNet (U-Net) models."""

    def __init__(self, run_dir: str, data_dir: str, output_dir: str = "validation_results_spatial"):
        self.run_dir = Path(run_dir).resolve()
        self.data_dir = Path(data_dir).resolve()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('cpu')  # MPS can be unstable
        else:
            self.device = torch.device('cpu')

        logger.info(f"Using device: {self.device}")

        # Load config
        self._load_config()

        # Load models
        self.models = self._load_ensemble()

        # Load test data
        self.test_data = self._load_test_data()

    def _load_config(self):
        """Load research config from run directory."""
        # Try multiple possible locations
        config_paths = [
            self.run_dir / 'research_config.json',
            self.run_dir.parent / 'research_config.json',
        ]

        self.config = None
        for cp in config_paths:
            if cp.exists():
                with open(cp, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded config from {cp}")
                break

        if self.config is None:
            logger.warning("No research_config.json found. Using defaults.")
            self.config = {
                'pld_values': [500, 1000, 1500, 2000, 2500, 3000],
                'hidden_sizes': [32, 64, 128, 256],
            }

        self.plds = np.array(self.config.get('pld_values', [500, 1000, 1500, 2000, 2500, 3000]))
        logger.info(f"Using PLDs: {self.plds}")

    def _load_ensemble(self) -> List[torch.nn.Module]:
        """Load trained SpatialASLNet models."""
        from spatial_asl_network import SpatialASLNet

        # Find model files
        models_dir = self.run_dir / 'trained_models'
        if not models_dir.exists():
            models_dir = self.run_dir

        model_files = sorted(list(models_dir.glob('ensemble_model_*.pt')))

        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}")

        logger.info(f"Found {len(model_files)} model files")

        # Get model architecture from config
        hidden_sizes = self.config.get('hidden_sizes', [32, 64, 128, 256])
        features = sorted(hidden_sizes) if len(hidden_sizes) >= 4 else [32, 64, 128, 256]

        # Ensure 4 levels for U-Net
        while len(features) < 4:
            features.insert(0, max(1, features[0] // 2))

        loaded_models = []

        for mp in model_files:
            logger.info(f"Loading {mp.name}...")

            model = SpatialASLNet(n_plds=len(self.plds), features=features)

            try:
                state_dict = torch.load(mp, map_location=self.device)
                if 'model_state_dict' in state_dict:
                    model.load_state_dict(state_dict['model_state_dict'])
                else:
                    model.load_state_dict(state_dict)

                model.to(self.device)
                model.eval()
                loaded_models.append(model)
                logger.info(f"  Loaded successfully")
            except Exception as e:
                logger.error(f"  Failed to load: {e}")

        if not loaded_models:
            raise RuntimeError("All models failed to load!")

        return loaded_models

    def _load_test_data(self, max_samples: int = 100) -> Dict:
        """Load spatial test data from chunks."""
        chunk_files = sorted(list(self.data_dir.glob('spatial_chunk_*.npz')))

        if not chunk_files:
            raise FileNotFoundError(f"No spatial_chunk_*.npz files found in {self.data_dir}")

        logger.info(f"Found {len(chunk_files)} data chunks")

        # Load a subset for validation
        signals_list = []
        targets_list = []
        samples_loaded = 0

        for cf in chunk_files:
            if samples_loaded >= max_samples:
                break

            data = np.load(cf)
            signals_list.append(data['signals'])
            targets_list.append(data['targets'])
            samples_loaded += len(data['signals'])

        signals = np.concatenate(signals_list, axis=0)[:max_samples]
        targets = np.concatenate(targets_list, axis=0)[:max_samples]

        # Apply M0 normalization (same as SpatialDataset)
        M0_SCALE = 100.0
        signals = signals * M0_SCALE

        logger.info(f"Loaded {len(signals)} test samples")
        logger.info(f"  Signal shape: {signals.shape}")
        logger.info(f"  Target shape: {targets.shape}")

        return {
            'signals': torch.from_numpy(signals).float(),
            'targets': targets,  # Keep as numpy for metrics
        }

    @torch.no_grad()
    def run_inference(self, batch_size: int = 16) -> Dict:
        """Run inference on test data."""
        logger.info("Running inference...")

        signals = self.test_data['signals'].to(self.device)
        n_samples = len(signals)

        all_cbf_preds = []
        all_att_preds = []

        for start_idx in tqdm(range(0, n_samples, batch_size), desc="Inference"):
            end_idx = min(start_idx + batch_size, n_samples)
            batch = signals[start_idx:end_idx]

            # Ensemble prediction
            cbf_batch_preds = []
            att_batch_preds = []

            for model in self.models:
                cbf, att, _, _ = model(batch)
                cbf_batch_preds.append(cbf.cpu())
                att_batch_preds.append(att.cpu())

            # Average ensemble
            cbf_avg = torch.stack(cbf_batch_preds).mean(dim=0)
            att_avg = torch.stack(att_batch_preds).mean(dim=0)

            all_cbf_preds.append(cbf_avg)
            all_att_preds.append(att_avg)

        cbf_preds = torch.cat(all_cbf_preds, dim=0).numpy()
        att_preds = torch.cat(all_att_preds, dim=0).numpy()

        return {
            'cbf_pred': cbf_preds,  # (N, 1, H, W)
            'att_pred': att_preds,  # (N, 1, H, W)
        }

    def compute_metrics(self, predictions: Dict) -> Dict:
        """Compute validation metrics."""
        logger.info("Computing metrics...")

        targets = self.test_data['targets']  # (N, 2, H, W) - [CBF, ATT]
        cbf_true = targets[:, 0:1, :, :]  # (N, 1, H, W)
        att_true = targets[:, 1:2, :, :]  # (N, 1, H, W)

        cbf_pred = predictions['cbf_pred']
        att_pred = predictions['att_pred']

        # Create brain mask (non-zero regions)
        mean_signal = np.mean(np.abs(self.test_data['signals'].numpy()), axis=1, keepdims=True)
        brain_mask = (mean_signal > np.percentile(mean_signal, 5)).astype(np.float32)

        def masked_metrics(pred, true, mask):
            """Compute metrics only on brain pixels."""
            pred_flat = pred[mask > 0.5]
            true_flat = true[mask > 0.5]

            mae = np.mean(np.abs(pred_flat - true_flat))
            rmse = np.sqrt(np.mean((pred_flat - true_flat) ** 2))
            bias = np.mean(pred_flat - true_flat)

            # Correlation
            if len(pred_flat) > 10:
                corr = np.corrcoef(pred_flat, true_flat)[0, 1]
            else:
                corr = np.nan

            return {
                'MAE': float(mae),
                'RMSE': float(rmse),
                'Bias': float(bias),
                'Correlation': float(corr),
                'N_pixels': int(len(pred_flat)),
            }

        # Expand mask for broadcasting
        mask_expanded = np.broadcast_to(brain_mask, cbf_true.shape)

        metrics = {
            'CBF': masked_metrics(cbf_pred, cbf_true, mask_expanded),
            'ATT': masked_metrics(att_pred, att_true, mask_expanded),
        }

        # Per-sample metrics for detailed analysis
        per_sample_mae_cbf = []
        per_sample_mae_att = []

        for i in range(len(cbf_pred)):
            m = brain_mask[i, 0] > 0.5
            if m.sum() > 0:
                per_sample_mae_cbf.append(np.mean(np.abs(cbf_pred[i, 0][m] - cbf_true[i, 0][m])))
                per_sample_mae_att.append(np.mean(np.abs(att_pred[i, 0][m] - att_true[i, 0][m])))

        metrics['per_sample'] = {
            'CBF_MAE_mean': float(np.mean(per_sample_mae_cbf)),
            'CBF_MAE_std': float(np.std(per_sample_mae_cbf)),
            'ATT_MAE_mean': float(np.mean(per_sample_mae_att)),
            'ATT_MAE_std': float(np.std(per_sample_mae_att)),
        }

        return metrics

    def generate_plots(self, predictions: Dict, num_examples: int = 4):
        """Generate visualization plots."""
        logger.info("Generating plots...")

        targets = self.test_data['targets']
        cbf_true = targets[:, 0, :, :]
        att_true = targets[:, 1, :, :]
        cbf_pred = predictions['cbf_pred'][:, 0, :, :]
        att_pred = predictions['att_pred'][:, 0, :, :]

        # Select random examples
        n_samples = len(cbf_true)
        indices = np.random.choice(n_samples, min(num_examples, n_samples), replace=False)

        fig, axes = plt.subplots(num_examples, 6, figsize=(18, 3 * num_examples))
        if num_examples == 1:
            axes = axes.reshape(1, -1)

        for row, idx in enumerate(indices):
            # CBF True
            im = axes[row, 0].imshow(cbf_true[idx], cmap='hot', vmin=0, vmax=100)
            axes[row, 0].set_title(f'CBF True\n(Sample {idx})')
            axes[row, 0].axis('off')
            plt.colorbar(im, ax=axes[row, 0], fraction=0.046)

            # CBF Pred
            im = axes[row, 1].imshow(cbf_pred[idx], cmap='hot', vmin=0, vmax=100)
            axes[row, 1].set_title('CBF Predicted')
            axes[row, 1].axis('off')
            plt.colorbar(im, ax=axes[row, 1], fraction=0.046)

            # CBF Error
            cbf_err = cbf_pred[idx] - cbf_true[idx]
            im = axes[row, 2].imshow(cbf_err, cmap='RdBu_r', vmin=-20, vmax=20)
            axes[row, 2].set_title(f'CBF Error\n(MAE: {np.mean(np.abs(cbf_err)):.1f})')
            axes[row, 2].axis('off')
            plt.colorbar(im, ax=axes[row, 2], fraction=0.046)

            # ATT True
            im = axes[row, 3].imshow(att_true[idx], cmap='viridis', vmin=0, vmax=3000)
            axes[row, 3].set_title('ATT True')
            axes[row, 3].axis('off')
            plt.colorbar(im, ax=axes[row, 3], fraction=0.046)

            # ATT Pred
            im = axes[row, 4].imshow(att_pred[idx], cmap='viridis', vmin=0, vmax=3000)
            axes[row, 4].set_title('ATT Predicted')
            axes[row, 4].axis('off')
            plt.colorbar(im, ax=axes[row, 4], fraction=0.046)

            # ATT Error
            att_err = att_pred[idx] - att_true[idx]
            im = axes[row, 5].imshow(att_err, cmap='RdBu_r', vmin=-500, vmax=500)
            axes[row, 5].set_title(f'ATT Error\n(MAE: {np.mean(np.abs(att_err)):.0f} ms)')
            axes[row, 5].axis('off')
            plt.colorbar(im, ax=axes[row, 5], fraction=0.046)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'spatial_examples.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved example visualizations to {self.output_dir / 'spatial_examples.png'}")

        # Scatter plots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Flatten for scatter (subsample for speed)
        n_plot = min(50000, cbf_true.size)
        idx_flat = np.random.choice(cbf_true.size, n_plot, replace=False)

        # CBF scatter
        axes[0].scatter(cbf_true.flatten()[idx_flat], cbf_pred.flatten()[idx_flat],
                        alpha=0.1, s=1, c='blue')
        axes[0].plot([0, 150], [0, 150], 'r--', linewidth=2, label='Identity')
        axes[0].set_xlabel('True CBF (ml/100g/min)')
        axes[0].set_ylabel('Predicted CBF')
        axes[0].set_title('CBF: Predicted vs True')
        axes[0].set_xlim(0, 150)
        axes[0].set_ylim(0, 150)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # ATT scatter
        axes[1].scatter(att_true.flatten()[idx_flat], att_pred.flatten()[idx_flat],
                        alpha=0.1, s=1, c='green')
        axes[1].plot([0, 3500], [0, 3500], 'r--', linewidth=2, label='Identity')
        axes[1].set_xlabel('True ATT (ms)')
        axes[1].set_ylabel('Predicted ATT')
        axes[1].set_title('ATT: Predicted vs True')
        axes[1].set_xlim(0, 3500)
        axes[1].set_ylim(0, 3500)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.output_dir / 'spatial_scatter.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved scatter plots to {self.output_dir / 'spatial_scatter.png'}")

    def run_validation(self):
        """Run complete validation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting Spatial Model Validation")
        logger.info("=" * 60)

        # Run inference
        predictions = self.run_inference()

        # Compute metrics
        metrics = self.compute_metrics(predictions)

        # Generate plots
        self.generate_plots(predictions)

        # Save metrics
        metrics_path = self.output_dir / 'spatial_metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("VALIDATION RESULTS")
        logger.info("=" * 60)
        logger.info(f"\nCBF Metrics:")
        logger.info(f"  MAE:  {metrics['CBF']['MAE']:.2f} ml/100g/min")
        logger.info(f"  RMSE: {metrics['CBF']['RMSE']:.2f} ml/100g/min")
        logger.info(f"  Bias: {metrics['CBF']['Bias']:.2f} ml/100g/min")
        logger.info(f"  Corr: {metrics['CBF']['Correlation']:.4f}")

        logger.info(f"\nATT Metrics:")
        logger.info(f"  MAE:  {metrics['ATT']['MAE']:.0f} ms")
        logger.info(f"  RMSE: {metrics['ATT']['RMSE']:.0f} ms")
        logger.info(f"  Bias: {metrics['ATT']['Bias']:.0f} ms")
        logger.info(f"  Corr: {metrics['ATT']['Correlation']:.4f}")

        logger.info("\n" + "=" * 60)
        logger.info("Validation Complete!")
        logger.info("=" * 60)

        return metrics


def main():
    parser = argparse.ArgumentParser(description="Validate SpatialASLNet models")
    parser.add_argument('--run_dir', type=str, required=True,
                        help='Path to trained model directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to spatial test data directory')
    parser.add_argument('--output_dir', type=str, default='validation_results_spatial',
                        help='Output directory for results')
    parser.add_argument('--max_samples', type=int, default=100,
                        help='Maximum number of test samples to use')
    args = parser.parse_args()

    try:
        validator = SpatialValidator(
            run_dir=args.run_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        validator.test_data['signals'] = validator.test_data['signals'][:args.max_samples]
        validator.test_data['targets'] = validator.test_data['targets'][:args.max_samples]

        metrics = validator.run_validation()
        return 0

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
