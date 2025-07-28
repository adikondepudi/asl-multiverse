import torch
import numpy as np
import yaml
from dataclasses import asdict
import os

print("🚀 Starting Local Code Validation Script (Fast Dry-Run Mode)...")
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
    from enhanced_asl_network import EnhancedASLNet, CustomLoss
    from torch.utils.data import DataLoader
    print("[SUCCESS] All project modules imported correctly.")
except ImportError as e:
    print(f"[FATAL ERROR] Could not import project modules. Is your conda environment active?")
    print(f"Details: {e}")
    exit()


def run_validation_checks():
    # --- 1. Create a minimal "debug" configuration ---
    print("\n--- Test 1: Configuring a minimal 'debug' environment ---")
    try:
        config_obj = {
            'batch_size': 2,
            'learning_rate': 0.001, 'weight_decay': 1e-5, 'hidden_sizes': [8, 4],
            'n_ensembles': 1, 'dropout_rate': 0.1, 'norm_type': "batch",
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

    # --- 2. Test Data Generation ---
    print("\n--- Test 2: Data Loading and Batch Generation ---")
    try:
        plds_np = np.array(config_obj['pld_values'])
        asl_params = ASLParameters(**{k: v for k, v in config_obj.items() if k in ASLParameters.__annotations__})
        simulator = RealisticASLSimulator(params=asl_params)
        train_dataset = ASLIterableDataset(simulator, plds_np, config_obj['training_noise_levels_stage1'], norm_stats)
        train_loader = DataLoader(train_dataset, batch_size=config_obj['batch_size'], num_workers=0)
        
        # Pull one batch to test the generator
        signals, params_norm = next(iter(train_loader))
        
        print(f"[SUCCESS] Data batch generated successfully.")
        print(f"    Signal batch shape: {signals.shape}")
        print(f"    Params batch shape: {params_norm.shape}")
    except Exception as e:
        print(f"[FAIL] Data loading failed. Check 'ASLIterableDataset' in 'asl_trainer.py'.")
        print(f"    Error: {e}")
        import traceback; traceback.print_exc()
        return

    # --- 3. Test Forward Pass and Loss Calculation (The "Dry Run") ---
    print("\n--- Test 3: Model Forward Pass & Loss Calculation ---")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"    Running test on device: '{device}'")
        
        base_input_size_nn = num_plds * 2 + 4 # 2 modalities, 4 engineered features

        # Instantiate the model and loss function directly
        model = EnhancedASLNet(input_size=base_input_size_nn, **config_obj).to(device)
        model.set_norm_stats(norm_stats)
        model.train() # Set to train mode

        loss_fn = CustomLoss(
            pinn_weight=config_obj['loss_pinn_weight_stage1'],
            pre_estimator_loss_weight=config_obj['pre_estimator_loss_weight_stage1'],
            model_params=config_obj
        )
        loss_fn.norm_stats = norm_stats

        # Move the single batch to the correct device
        signals, params_norm = signals.to(device), params_norm.to(device)
        
        # Use autocast to simulate the exact training condition (catches dtype errors)
        with torch.amp.autocast(device_type=device, dtype=torch.bfloat16):
            # Perform the forward pass
            outputs = model(signals)
            cbf_mean_norm, att_mean_norm, cbf_log_var, att_log_var, cbf_rough, att_rough = outputs
            
            # Perform the loss calculation
            loss, loss_components = loss_fn(
                signals, cbf_mean_norm, att_mean_norm, 
                params_norm[:, 0:1], params_norm[:, 1:2], 
                cbf_log_var, att_log_var, 
                cbf_rough, att_rough, 
                global_epoch=0
            )

        print(f"[SUCCESS] Model forward pass and loss calculation completed.")
        print(f"    Calculated Loss (Tensor): {loss.item():.4f}")

    except Exception as e:
        print(f"[FAIL] The forward pass or loss calculation failed. This is a critical error.")
        print(f"    Check model architecture, loss function, and data shapes/types.")
        print(f"    Error: {e}")
        import traceback; traceback.print_exc()
        return

    print("-" * 50)
    print("✅ All local validation checks passed! Your code is ready for the HPC.")
    print("-" * 50)

if __name__ == "__main__":
    run_validation_checks()