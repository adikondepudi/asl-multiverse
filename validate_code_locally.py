# FILE: validate_code_locally.py
#
# PURPOSE:
#   A "pre-flight checklist" to run on your LOCAL computer before submitting a job to the HPC.
#   This script performs a rapid, small-scale run of the entire training and validation
#   pipeline to catch common errors like data loading issues, model architecture mismatches,
#   and data type conflicts (e.g., BFloat16).
#
# HOW TO USE:
#   1. Activate your local conda environment.
#   2. Run from your terminal: python validate_code_locally.py
#
# EXPECTED OUTPUT:
#   - If successful, it will print success messages for each test and end with
#     "✅ All local validation checks passed!".
#   - If it fails, it will print a detailed error message pointing to the bug.
#     Fix the bug and run this script again until it passes.

import torch
import numpy as np
import yaml
from dataclasses import asdict
import os

print("🚀 Starting Local Code Validation Script...")
print("-" * 50)

# --- Suppress known warnings for a cleaner output ---
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    # --- Import all necessary components from your project ---
    from asl_simulation import ASLParameters
    from enhanced_simulation import RealisticASLSimulator
    from asl_trainer import EnhancedASLTrainer, ASLIterableDataset
    from enhanced_asl_network import EnhancedASLNet
    from torch.utils.data import DataLoader
    print("[SUCCESS] All project modules imported correctly.")
except ImportError as e:
    print(f"[FATAL ERROR] Could not import project modules. Is your conda environment active?")
    print(f"Details: {e}")
    exit()


def run_validation_checks():
    # --- 1. Create a minimal "debug" configuration ---
    # We use a tiny config to make the tests run in seconds, not hours.
    print("\n--- Test 1: Configuring a minimal 'debug' environment ---")
    try:
        config_obj = {
            'batch_size': 2,
            'learning_rate': 0.001, 'weight_decay': 1e-5, 'hidden_sizes': [8, 4],
            'n_ensembles': 1, 'dropout_rate': 0.1, 'norm_type': "batch",
            'steps_per_epoch_stage1': 1, 'n_epochs_stage1': 1,
            'validation_steps_per_epoch': 1,
            'pld_values': [500, 1000, 1500, 2000, 2500, 3000],
            'T1_artery': 1850.0, 'T_tau': 1800.0, 'alpha_PCASL': 0.85, 'alpha_VSASL': 0.56,
            'alpha_BS1': 1.0, 'T2_factor': 1.0, 'training_noise_levels_stage1': [5.0],
            # Add other necessary keys for EnhancedASLNet and CustomLoss
            'loss_pinn_weight_stage1': 0.1, 'pre_estimator_loss_weight_stage1': 0.1,
            'use_transformer_temporal_model': True, 'm0_input_feature_model': False,
            'use_focused_transformer_model': True, 'transformer_d_model_focused': 8,
            'transformer_nhead_model': 2, 'transformer_nlayers_model': 1,
            'log_var_cbf_min': -6.0, 'log_var_cbf_max': 7.0, 'log_var_att_min': -2.0,
            'log_var_att_max': 14.0, 'loss_weight_cbf': 1.0, 'loss_weight_att': 1.0,
            'loss_log_var_reg_lambda': 0.0
        }

        # Create dummy norm_stats to simulate the real environment
        num_plds = len(config_obj['pld_values'])
        norm_stats = {
            'pcasl_mean': [0.0] * num_plds, 'pcasl_std': [1.0] * num_plds,
            'vsasl_mean': [0.0] * num_plds, 'vsasl_std': [1.0] * num_plds,
            'y_mean_cbf': 40.0, 'y_std_cbf': 20.0,
            'y_mean_att': 2000.0, 'y_std_att': 1000.0,
            'amplitude_mean': 0.1, 'amplitude_std': 0.1,
        }
        print("[SUCCESS] Minimal configuration and dummy norm_stats created.")
    except Exception as e:
        print(f"[FAIL] Could not create the test configuration. Error: {e}")
        return

    # --- 2. Test Data Loading and Generation ---
    # This will catch errors in the ASLIterableDataset __iter__ method.
    print("\n--- Test 2: Data Loading and Batch Generation ---")
    try:
        plds_np = np.array(config_obj['pld_values'])
        asl_params = ASLParameters(**{k: v for k, v in config_obj.items() if k in ASLParameters.__annotations__})
        simulator = RealisticASLSimulator(params=asl_params)
        train_dataset = ASLIterableDataset(simulator, plds_np, config_obj['training_noise_levels_stage1'], norm_stats)
        # Use num_workers=0 for simplicity and to avoid multiprocessing issues on some local setups
        train_loader = DataLoader(train_dataset, batch_size=config_obj['batch_size'], num_workers=0)
        
        # Try to pull one batch
        signals, params = next(iter(train_loader))
        
        print(f"[SUCCESS] Data batch generated successfully.")
        print(f"    Signal batch shape: {signals.shape}")
        print(f"    Params batch shape: {params.shape}")
    except Exception as e:
        print(f"[FAIL] Data loading failed. Check 'ASLIterableDataset' in 'asl_trainer.py'.")
        print(f"    Error: {e}")
        import traceback; traceback.print_exc()
        return

    # --- 3. Test Full Training and Validation Step ---
    # This tests the model, loss function, `autocast` (BFloat16), and validation logic.
    print("\n--- Test 3: Full Training & Validation Step ---")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Running test on device: '{device}'")
        
        # We need a validation loader for this test
        val_dataset = ASLIterableDataset(simulator, plds_np, config_obj['training_noise_levels_stage1'], norm_stats)
        val_loader = DataLoader(val_dataset, batch_size=config_obj['batch_size'], num_workers=0)

        # The base_input_size must match the data generation pipeline
        base_input_size_nn = num_plds * 2 + 4 # 2 modalities, 4 engineered features

        trainer = EnhancedASLTrainer(
            model_config=config_obj,
            model_class=lambda **kwargs: EnhancedASLNet(input_size=base_input_size_nn, **kwargs),
            input_size=base_input_size_nn,
            n_ensembles=1,
            device=device,
            batch_size=config_obj['batch_size']
        )
        trainer.norm_stats = norm_stats
        trainer.custom_loss_fn.norm_stats = norm_stats
        trainer.models[0].set_norm_stats(norm_stats)
        
        # Run a single training epoch with a single step
        trainer.train_ensemble(
            train_loaders=[train_loader],
            val_loaders=[val_loader],
            epoch_schedule=[1],
            steps_per_epoch_schedule=[1]
        )
        print(f"[SUCCESS] One training epoch and one validation epoch completed without errors.")
    except Exception as e:
        print(f"[FAIL] The training or validation step failed. This could be a model error, loss function error, or data type mismatch.")
        print(f"    Error: {e}")
        import traceback; traceback.print_exc()
        return

    print("-" * 50)
    print("✅ All local validation checks passed! Your code is likely ready for the HPC.")
    print("-" * 50)


if __name__ == "__main__":
    run_validation_checks()