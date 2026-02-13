import sys
import os
import time
import warnings

# --- 1. Immediate Debug Print ---
print("--- [DEBUG] Script process started... ---")

try:
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    import json
    import logging
    import argparse
    from tqdm import tqdm
    from pathlib import Path
    from scipy.stats import binned_statistic
    import glob
    print("--- [DEBUG] Standard libraries imported successfully. ---")
except ImportError as e:
    print(f"!!! [CRITICAL ERROR] Import failed: {e}")
    sys.exit(1)

# --- 2. Custom Imports ---
try:
    import yaml
    from asl_simulation import ASLParameters
    from enhanced_simulation import RealisticASLSimulator, SpatialPhantomGenerator
    from enhanced_asl_network import DisentangledASLNet
    from spatial_asl_network import SpatialASLNet, DualEncoderSpatialASLNet  # Spatial models (U-Net)
    from amplitude_aware_spatial_network import AmplitudeAwareSpatialASLNet  # Amplitude-aware model
    from utils import process_signals_dynamic, get_grid_search_initial_guess
    from multiverse_functions import fit_PCVSASL_misMatchPLD_vectInit_pep
    from feature_registry import FeatureRegistry, validate_signals, validate_norm_stats
    print("--- [DEBUG] Custom ASL modules imported successfully. ---")
except ImportError as e:
    print(f"!!! [CRITICAL ERROR] Could not import project files: {e}")
    sys.exit(1)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

class ASLValidator:
    def __init__(self, run_dir, output_dir="validation_results"):
        self.run_dir = Path(run_dir).resolve()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Check Directory
        if not self.run_dir.exists():
            logger.error(f"Directory not found: {self.run_dir}")
            sys.exit(1)

        # Device Setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('cpu') 
            print("--- [DEBUG] MacOS detected. Using CPU for stability.")
        else:
            self.device = torch.device('cpu')
        
        # 1. Load Research Config FIRST (needed for PLDs and features)
        config_path = self.run_dir / 'research_config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.config = json.load(f)
            logger.info(f"Loaded research_config.json")
        else:
            logger.error(f"CRITICAL: research_config.json not found in {self.run_dir}. Cannot proceed.")
            sys.exit(1)
        
        # 2. Extract PLDs from config (NO HARDCODING)
        pld_values = self.config.get('pld_values')
        if pld_values is None:
            logger.error("Config missing 'pld_values'. Cannot proceed.")
            sys.exit(1)
        self.plds = np.array(pld_values, dtype=np.float64)
        logger.info(f"Using PLDs from config: {self.plds}")
        
        # 3. Validate active_features exists (NO DEFAULTS - must match training)
        self.active_features = self.config.get('active_features')
        if self.active_features is None:
            logger.error("Config missing 'active_features'. Cannot proceed.")
            sys.exit(1)
        FeatureRegistry.validate_active_features(self.active_features)
        logger.info(f"Using active_features from config: {self.active_features}")
        
        # 4. Load physics params from config with defaults
        self.params = ASLParameters(
            T1_artery=self.config.get('T1_artery', 1650.0),  # 3T consensus (Alsop 2015)
            T_tau=self.config.get('T_tau', 1800.0), 
            alpha_PCASL=self.config.get('alpha_PCASL', 0.85),
            alpha_VSASL=self.config.get('alpha_VSASL', 0.56),
            TR_PCASL=4000.0, TR_VSASL=3936.0
        )
        self.simulator = RealisticASLSimulator(params=self.params)
        
        # 5. Load Normalization Stats
        norm_stats_path = self.run_dir / 'norm_stats.json'
        if norm_stats_path.exists():
            logger.info(f"Loading normalization stats from {norm_stats_path.name}")
            with open(norm_stats_path, 'r') as f:
                self.norm_stats = json.load(f)
            # Validate norm_stats
            validate_norm_stats(self.norm_stats, context="ASLValidator init")
        else:
            logger.error(f"Could not find norm_stats.json in {self.run_dir}")
            sys.exit(1)
        
        # 5. Initialize plot data storage for interactive dashboard
        self.plot_data_storage = {}

        # 6. Load Ensemble Models
        self.models = self._load_ensemble()
    
    def _log_llm_metrics(self, scenario_name, nn_preds, ls_preds, ground_truth, label):
        """
        Calculates high-level statistics optimized for LLM interpretation.
        """
        def calc_stats(preds, truth):
            # Handle NaNs (common in LS)
            mask = ~np.isnan(preds)
            if np.sum(mask) == 0:
                return {k: None for k in ["MAE", "RMSE", "Bias", "R2", "Failure_Rate"]}
            
            p, t = preds[mask], truth[mask]
            err = p - t
            
            mae = np.mean(np.abs(err))
            rmse = np.sqrt(np.mean(err**2))
            bias = np.mean(err)
            
            # R2 Score
            ss_res = np.sum(err**2)
            ss_tot = np.sum((t - np.mean(t))**2)
            r2 = 1 - (ss_res / (ss_tot + 1e-6))
            
            return {
                "MAE": float(mae),
                "RMSE": float(rmse),
                "Bias": float(bias),
                "R2": float(r2),
                "Failure_Rate": float(1.0 - (len(p) / len(preds)))
            }

        nn_stats = calc_stats(nn_preds, ground_truth)
        ls_stats = calc_stats(ls_preds, ground_truth)
        
        # Calculate "Win Rate" (How often NN is closer to truth than LS)
        valid_idx = (~np.isnan(nn_preds)) & (~np.isnan(ls_preds))
        if np.sum(valid_idx) > 0:
            nn_err = np.abs(nn_preds[valid_idx] - ground_truth[valid_idx])
            ls_err = np.abs(ls_preds[valid_idx] - ground_truth[valid_idx])
            nn_wins = np.sum(nn_err < ls_err)
            win_rate = float(nn_wins / len(nn_err))
        else:
            win_rate = None

        # Store in dictionary
        if not hasattr(self, 'llm_report'):
            self.llm_report = {}
            
        if scenario_name not in self.llm_report:
            self.llm_report[scenario_name] = {}
            
        self.llm_report[scenario_name][label] = {
            "Neural_Net": nn_stats,
            "Least_Squares": ls_stats,
            "NN_vs_LS_Win_Rate": win_rate
        }

    def _bootstrap_ci(self, data, n_bootstrap=1000, ci=0.95):
        """
        Compute bootstrap confidence interval for the mean of an array.

        Args:
            data: 1D array of per-sample values (e.g., per-voxel absolute errors)
            n_bootstrap: Number of bootstrap resamples (default 1000)
            ci: Confidence level (default 0.95 for 95% CI)

        Returns:
            (mean, ci_lower, ci_upper) tuple
        """
        data = np.asarray(data)
        # Remove NaNs
        data = data[~np.isnan(data)]
        if len(data) == 0:
            return (np.nan, np.nan, np.nan)

        rng = np.random.RandomState(42)  # Reproducible
        n = len(data)
        boot_means = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sample = data[rng.randint(0, n, size=n)]
            boot_means[b] = np.mean(sample)

        alpha = 1.0 - ci
        ci_lower = float(np.percentile(boot_means, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
        mean_val = float(np.mean(data))
        return (mean_val, ci_lower, ci_upper)

    def _bootstrap_ci_winrate(self, nn_errors, ls_errors, n_bootstrap=1000, ci=0.95):
        """
        Compute bootstrap confidence interval for win rate (fraction where NN < LS).

        Args:
            nn_errors: 1D array of NN absolute errors per voxel
            ls_errors: 1D array of LS absolute errors per voxel (same locations)
            n_bootstrap: Number of bootstrap resamples
            ci: Confidence level

        Returns:
            (win_rate, ci_lower, ci_upper) tuple
        """
        nn_errors = np.asarray(nn_errors)
        ls_errors = np.asarray(ls_errors)
        # Remove pairs where either is NaN
        valid = ~(np.isnan(nn_errors) | np.isnan(ls_errors))
        nn_errors = nn_errors[valid]
        ls_errors = ls_errors[valid]
        if len(nn_errors) == 0:
            return (np.nan, np.nan, np.nan)

        nn_wins = (nn_errors < ls_errors).astype(float)
        rng = np.random.RandomState(42)
        n = len(nn_wins)
        boot_rates = np.empty(n_bootstrap)
        for b in range(n_bootstrap):
            sample = nn_wins[rng.randint(0, n, size=n)]
            boot_rates[b] = np.mean(sample)

        alpha = 1.0 - ci
        ci_lower = float(np.percentile(boot_rates, 100 * alpha / 2))
        ci_upper = float(np.percentile(boot_rates, 100 * (1 - alpha / 2)))
        win_rate = float(np.mean(nn_wins))
        return (win_rate, ci_lower, ci_upper)

    def save_llm_report(self):
        """Saves the stats to JSON and Markdown for easy LLM pasting."""
        # 1. Save JSON (Machine Readable)
        json_path = self.output_dir / "llm_analysis_report.json"
        with open(json_path, 'w') as f:
            json.dump(self.llm_report, f, indent=4)
            
        # 2. Save Markdown (Human/LLM Readable)
        md_path = self.output_dir / "llm_analysis_report.md"
        with open(md_path, 'w') as f:
            f.write("# ASL Multiverse Validation Report (LLM Optimized)\n\n")
            for scenario, metrics in self.llm_report.items():
                f.write(f"## Scenario: {scenario}\n")
                for param, data in metrics.items():
                    f.write(f"### Parameter: {param}\n")
                    f.write(f"- **NN vs LS Win Rate**: {data['NN_vs_LS_Win_Rate']:.2%}\n" if data['NN_vs_LS_Win_Rate'] is not None else "- **NN vs LS Win Rate**: N/A\n")
                    
                    nn = data['Neural_Net']
                    ls = data['Least_Squares']
                    
                    f.write("| Metric | Neural Net | Least Squares |\n")
                    f.write("| :--- | :--- | :--- |\n")
                    f.write(f"| MAE | {nn['MAE']:.4f} | {ls['MAE']:.4f} |\n")
                    f.write(f"| RMSE | {nn['RMSE']:.4f} | {ls['RMSE']:.4f} |\n")
                    f.write(f"| Bias | {nn['Bias']:.4f} | {ls['Bias']:.4f} |\n")
                    f.write(f"| R² | {nn['R2']:.4f} | {ls['R2']:.4f} |\n")
                    f.write(f"| Fail Rate | {nn['Failure_Rate']:.1%} | {ls['Failure_Rate']:.1%} |\n\n")
        
        print(f"\n--- [LLM REPORT SAVED] ---\nJSON: {json_path}\nMarkdown: {md_path}")

    def save_plot_data_json(self):
        """Saves raw plot coordinates for the interactive Streamlit dashboard."""
        json_path = self.output_dir / "interactive_plot_data.json"
        
        # Helper to handle NaN values for JSON serialization
        def nan_to_none(obj):
            if isinstance(obj, float) and np.isnan(obj):
                return None
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(json_path, 'w') as f:
            json.dump(self.plot_data_storage, f, indent=2, default=nan_to_none)
        
        print(f"   > Saved interactive dashboard data to {json_path}")

    def _detect_model_architecture(self, state_dict_keys):
        """
        Auto-detect model architecture from checkpoint keys.

        Returns:
            str: 'spatial' for SpatialASLNet (U-Net), 'voxel' for DisentangledASLNet
        """
        keys_str = ' '.join(state_dict_keys)

        # SpatialASLNet has encoder1.double_conv, encoder2, etc.
        if 'encoder1.double_conv' in keys_str or 'encoder1.' in keys_str:
            return 'spatial'

        # DisentangledASLNet has encoder.pcasl_tower or encoder.encoder_mlp
        if 'encoder.pcasl_tower' in keys_str or 'encoder.encoder_mlp' in keys_str:
            return 'voxel'

        # Default to voxel if unclear
        logger.warning("Could not auto-detect architecture from checkpoint keys. Defaulting to 'voxel'.")
        return 'voxel'

    def _load_ensemble(self):
        models_dir = self.run_dir / 'trained_models'
        if not models_dir.exists():
             logger.error(f"trained_models/ folder not found at: {models_dir}")
             sys.exit(1)

        model_files = sorted(list(models_dir.glob('ensemble_model_*.pt')))

        if not model_files:
            logger.error(f"No .pt files found in {models_dir}")
            sys.exit(1)

        logger.info(f"Found {len(model_files)} models. Loading onto {self.device}...")

        loaded_models = []

        # --- AUTO-DETECT MODEL ARCHITECTURE FROM CHECKPOINT ---
        first_state = torch.load(model_files[0], map_location='cpu')
        sd = first_state['model_state_dict'] if 'model_state_dict' in first_state else first_state
        self.model_architecture = self._detect_model_architecture(list(sd.keys()))
        logger.info(f"Auto-detected model architecture: {self.model_architecture}")

        if self.model_architecture == 'spatial':
            # --- SPATIAL MODEL LOADING ---
            self.is_spatial = True

            # Load config.yaml to determine exact model class
            config_yaml_path = self.run_dir / 'config.yaml'
            training_config = {}
            if config_yaml_path.exists():
                with open(config_yaml_path, 'r') as f:
                    full_config = yaml.safe_load(f)
                    training_config = full_config.get('training', {})

            model_class_name = training_config.get('model_class_name', 'SpatialASLNet')
            logger.info(f"Loading {model_class_name} models for spatial validation...")

            for mp in model_files:
                print(f"   ... Loading {mp.name}")

                # Instantiate correct model class based on config
                if model_class_name == 'AmplitudeAwareSpatialASLNet':
                    # Read architecture flags from config.yaml (not hardcoded defaults)
                    use_film_at_bottleneck = training_config.get('use_film_at_bottleneck', True)
                    use_film_at_decoder = training_config.get('use_film_at_decoder', True)
                    use_amplitude_output_modulation = training_config.get('use_amplitude_output_modulation', True)

                    logger.info(f"AmplitudeAware config: film_bottleneck={use_film_at_bottleneck}, "
                               f"film_decoder={use_film_at_decoder}, output_mod={use_amplitude_output_modulation}")

                    # Instantiate with config settings
                    model = AmplitudeAwareSpatialASLNet(
                        n_plds=len(self.plds),
                        features=training_config.get('hidden_sizes', [32, 64, 128, 256]),
                        use_film_at_bottleneck=use_film_at_bottleneck,
                        use_film_at_decoder=use_film_at_decoder,
                        use_amplitude_output_modulation=use_amplitude_output_modulation,
                    )
                elif model_class_name == 'DualEncoderSpatialASLNet':
                    # DualEncoderSpatialASLNet: Y-Net with separate PCASL/VSASL streams
                    model = DualEncoderSpatialASLNet(
                        n_plds=len(self.plds),
                        features=training_config.get('hidden_sizes', [32, 64, 128, 256])
                    )
                else:
                    # Default to SpatialASLNet
                    model = SpatialASLNet(
                        n_plds=len(self.plds),
                        features=training_config.get('hidden_sizes', [32, 64, 128, 256])
                    )

                try:
                    state_dict = torch.load(mp, map_location=self.device)
                    if 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        model.load_state_dict(state_dict)

                    model.to(self.device)
                    model.eval()
                    loaded_models.append(model)
                except Exception as e:
                    logger.error(f"Failed to load {mp.name}: {e}")
        else:
            # --- VOXEL MODEL LOADING (DisentangledASLNet) ---
            self.is_spatial = False

            # Extract Architecture Params
            hidden_sizes = self.config.get('hidden_sizes', [128, 64, 32])
            d_model = self.config.get('transformer_d_model_focused', 32)
            nhead = self.config.get('transformer_nhead_model', 4)
            moe_config = self.config.get('moe', None)
            encoder_type = self.config.get('encoder_type', 'physics_processor')

            # --- AUTO-DETECT NUM_SCALAR_FEATURES ---
            # Peek at the first model to find the input dimension of the FiLM layer
            try:
                # Shape is [out, in]. index 1 is input dimension.
                # For MLP-only encoder, use a different key
                if encoder_type.lower() == 'mlp_only':
                    self.detected_scalar_features = sd['encoder.encoder_mlp.0.weight'].shape[1] - (len(self.plds) * 2)
                else:
                    self.detected_scalar_features = sd['encoder.pcasl_film.generator.0.weight'].shape[1]
                logger.info(f"Auto-detected scalar features from checkpoint: {self.detected_scalar_features}")
            except Exception as e:
                logger.warning(f"Could not auto-detect scalar features: {e}. Defaulting to norm_stats + 1.")
                self.detected_scalar_features = len(self.norm_stats['scalar_features_mean']) + 1

            # Input size is dynamically determined from checkpoint - scalars are auto-detected
            input_size = len(self.plds) * 2 + self.detected_scalar_features

            for mp in model_files:
                print(f"   ... Loading {mp.name}")

                # Reconstruct exact architecture
                model = DisentangledASLNet(
                    mode='regression',
                    input_size=input_size,
                    n_plds=len(self.plds),
                    num_scalar_features=self.detected_scalar_features,
                    hidden_sizes=hidden_sizes,
                    transformer_d_model_focused=d_model,
                    transformer_nhead_model=nhead,
                    dropout_rate=0.0,
                    moe=moe_config,
                    encoder_type=encoder_type
                )

                try:
                    state_dict = torch.load(mp, map_location=self.device)
                    if 'model_state_dict' in state_dict:
                        model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        model.load_state_dict(state_dict)

                    model.to(self.device)
                    model.eval()
                    loaded_models.append(model)
                except Exception as e:
                    logger.error(f"Failed to load {mp.name}: {e}")

        if not loaded_models:
            logger.error("All models failed to load.")
            sys.exit(1)

        return loaded_models

    def generate_noise(self, clean_signal, tsnr):
        ref_signal = self.simulator._compute_reference_signal()
        noise_sd = ref_signal / tsnr
        noise_scaling = self.simulator.compute_tr_noise_scaling(self.plds)
        n_plds = len(self.plds)
        pcasl_noise = noise_sd * noise_scaling['PCASL'] * np.random.randn(n_plds)
        vsasl_noise = noise_sd * noise_scaling['VSASL'] * np.random.randn(n_plds)
        return clean_signal + np.concatenate([pcasl_noise, vsasl_noise])

    def run_nn_inference(self, signals, t1_values):
        # Guard: Spatial models require different inference pipeline
        if hasattr(self, 'is_spatial') and self.is_spatial:
            raise RuntimeError(
                "run_nn_inference is for voxel-wise models only. "
                "Spatial (U-Net) models require 2D image input, not 1D voxel signals."
            )

        n_plds = len(self.plds)
        
        # Use active_features from config (validated in __init__)
        processing_config = {
            'pld_values': list(self.plds.astype(int)),
            'active_features': self.active_features  # Use validated config, no defaults
        }
        
        t1_input = t1_values.reshape(-1, 1).astype(np.float32)
        
        # Use dynamic feature processing - pass raw signals only
        processed = process_signals_dynamic(signals, self.norm_stats, processing_config, t1_values=t1_input)
        
        inputs_tensor = torch.from_numpy(processed.astype(np.float32)).to(self.device)
        
        cbf_preds_accum = []
        att_preds_accum = []

        with torch.no_grad():
            for model in self.models:
                # MOE HEAD returns 4 values: cbf_mean, att_mean, cbf_log, att_log
                # STANDARD HEAD returns 4 values (via ModuleDict logic in forward)
                cbf_norm, att_norm, _, _, _, _ = model(inputs_tensor)
                cbf_preds_accum.append(cbf_norm)
                att_preds_accum.append(att_norm)

        cbf_norm_avg = torch.stack(cbf_preds_accum).mean(dim=0).cpu().numpy().flatten()
        att_norm_avg = torch.stack(att_preds_accum).mean(dim=0).cpu().numpy().flatten()

        cbf_pred = cbf_norm_avg * self.norm_stats['y_std_cbf'] + self.norm_stats['y_mean_cbf']
        att_pred = att_norm_avg * self.norm_stats['y_std_att'] + self.norm_stats['y_mean_att']
        
        return cbf_pred, att_pred

    def run_ls_inference(self, signals, t1_artery_val):
        logger.info("Running Least Squares (Grid Search + Optimizer)...")
        n_samples = signals.shape[0]
        cbf_preds = []
        att_preds = []
        
        ls_params = {
            'T1_artery': t1_artery_val, 
            'T_tau': self.params.T_tau,
            'alpha_PCASL': self.params.alpha_PCASL,
            'alpha_VSASL': self.params.alpha_VSASL,
            'T2_factor': self.params.T2_factor,
            'alpha_BS1': self.params.alpha_BS1
        }
        
        # Prepare PLD time input for the optimizer
        pldti = np.column_stack([self.plds, self.plds])
        
        iter_wrapper = tqdm(range(n_samples), desc="LS Fitting") if n_samples > 10 else range(n_samples)

        for i in iter_wrapper:
            try:
                if isinstance(t1_artery_val, np.ndarray):
                    ls_params['T1_artery'] = t1_artery_val[i]
                
                # 1. Get Initial Guess via Grid Search
                init_guess = get_grid_search_initial_guess(signals[i], self.plds, ls_params)
                
                # 2. Refine using Least Squares Optimizer
                signal_reshaped = signals[i].reshape((len(self.plds), 2), order='F')
                
                beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                    pldti, 
                    signal_reshaped, 
                    init_guess, 
                    **ls_params
                )
                
                cbf_preds.append(beta[0] * 6000.0)
                att_preds.append(beta[1])
            except Exception as e:
                # FIX: Print the error for the first failure so we can debug
                if i == 0: 
                    print(f"\n!!! LS GRID SEARCH FAILED: {e}")
                    import traceback
                    traceback.print_exc()
                cbf_preds.append(np.nan)
                att_preds.append(np.nan)
                
        return np.array(cbf_preds), np.array(att_preds)

    def run_phase_1(self):
        logger.info("--- Phase 1: Simulation Checks ---")
        scenarios = [
            {'cbf': 60.0, 'att': 1200.0, 'label': 'Standard'},
            {'cbf': 30.0, 'att': 3500.0, 'label': 'Low Flow - Delayed'}
        ]
        
        for i, sc in enumerate(scenarios):
            cbf, att = sc['cbf'], sc['att']
            vsasl = self.simulator._generate_vsasl_signal(self.plds, att, cbf, self.params.T1_artery, self.params.alpha_VSASL)
            pcasl = self.simulator._generate_pcasl_signal(self.plds, att, cbf, self.params.T1_artery, self.params.T_tau, self.params.alpha_PCASL)
            
            plt.figure(figsize=(8, 6))
            plt.plot(self.plds, pcasl, 'o-', label='PCASL', linewidth=2, color='blue')
            plt.plot(self.plds, vsasl, 's-', label='VSASL', linewidth=2, color='red')
            plt.title(f"Phase 1 - Plot {i+1}: {sc['label']}\nCBF={cbf}, ATT={att}")
            plt.xlabel("PLD (ms)"); plt.ylabel("Signal"); plt.legend(); plt.grid(True, alpha=0.3)
            plt.savefig(self.output_dir / f"Phase1_Plot{i+1}_{sc['label'].replace(' ', '_')}.png", dpi=150)
            plt.close()
            print(f"   > Saved Phase1 Plot {i+1}")

    def _plot_scenario(self, scenario_name, x_values, x_label, nn_cbf, nn_att, ls_cbf, ls_att, true_cbf, true_att):
        print(f"   > Generating plots and caching data for {scenario_name}")
        x_values = np.array(x_values)
        
        # --- 1. Curve Binning Logic ---
        def get_binned_stats(preds, truths, x_vals):
            bins = 15
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
                
                # Bias = Pred - Truth
                bin_bias = binned_statistic(x_vals, preds - truths, statistic=np.nanmean, bins=bins)
                # CoV = Std / Mean_Truth
                bin_pred_std = binned_statistic(x_vals, preds, statistic=np.nanstd, bins=bins)
                bin_truth_mean = binned_statistic(x_vals, truths, statistic=np.nanmean, bins=bins).statistic
                cov_stat = (bin_pred_std.statistic / (bin_truth_mean + 1e-6)) * 100
                
                bin_centers = 0.5 * (bin_bias.bin_edges[1:] + bin_bias.bin_edges[:-1])
                # Convert to list for JSON serialization
                return bin_centers.tolist(), bin_bias.statistic.tolist(), cov_stat.tolist()
        
        # --- 2. Scalar Metric Logic (Global MAE for leaderboard) ---
        def get_scalar_mae(preds, truths):
            mask = ~np.isnan(preds)
            if mask.sum() == 0:
                return None
            return float(np.mean(np.abs(preds[mask] - truths[mask])))
        
        # Calculate Binned Curves
        nn_x, nn_bias, nn_cov = get_binned_stats(nn_cbf, true_cbf, x_values)
        ls_x, ls_bias, ls_cov = get_binned_stats(ls_cbf, true_cbf, x_values)
        _, nn_bias_att, nn_cov_att = get_binned_stats(nn_att, true_att, x_values)
        _, ls_bias_att, ls_cov_att = get_binned_stats(ls_att, true_att, x_values)
        
        # Calculate Scalar Metrics for Leaderboard
        metrics = {
            "CBF_MAE_NN": get_scalar_mae(nn_cbf, true_cbf),
            "CBF_MAE_LS": get_scalar_mae(ls_cbf, true_cbf),
            "ATT_MAE_NN": get_scalar_mae(nn_att, true_att),
            "ATT_MAE_LS": get_scalar_mae(ls_att, true_att),
        }
        
        # --- 3. Store Everything for Interactive Dashboard ---
        self.plot_data_storage[scenario_name] = {
            "x_axis": nn_x,
            "x_label": x_label,
            "metrics": metrics,
            "curves": {
                "CBF_Bias": {"nn": nn_bias, "ls": ls_bias},
                "CBF_CoV": {"nn": nn_cov, "ls": ls_cov},
                "ATT_Bias": {"nn": nn_bias_att, "ls": ls_bias_att},
                "ATT_CoV": {"nn": nn_cov_att, "ls": ls_cov_att}
            }
        }
        
        # --- 4. Static PNG Plotting (keep existing behavior) ---
        def make_plot(metric_name, nn_y, ls_y, ylabel):
            plt.figure(figsize=(6, 4))
            nn_y_arr = np.array(nn_y)
            ls_y_arr = np.array(ls_y)
            nn_x_arr = np.array(nn_x)
            ls_x_arr = np.array(ls_x)
            
            plt.plot(nn_x_arr, nn_y_arr, 'o-', color='purple', label='Neural Net', linewidth=2, markersize=5)
            
            valid_ls = ~np.isnan(ls_y_arr)
            if np.sum(valid_ls) > 0:
                plt.plot(ls_x_arr[valid_ls], ls_y_arr[valid_ls], 'x--', color='gray', label='LS', linewidth=1.5, markersize=5)
            
            plt.xlabel(x_label); plt.ylabel(ylabel); plt.title(f"{scenario_name}: {metric_name}")
            plt.legend(); plt.grid(True, alpha=0.3); plt.axhline(0, color='black', linewidth=0.5)
            plt.savefig(self.output_dir / f"{scenario_name}_{metric_name.replace(' ', '')}.png", dpi=150)
            plt.close()

        make_plot("CBF Bias", nn_bias, ls_bias, "Error")
        make_plot("CBF CoV", nn_cov, ls_cov, "CoV (%)")
        make_plot("ATT Bias", nn_bias_att, ls_bias_att, "Error")
        make_plot("ATT CoV", nn_cov_att, ls_cov_att, "CoV (%)")

    def _run_spatial_at_snr(self, snr_value, n_phantoms, phantom_size=64):
        """
        Run spatial validation at a single SNR level.

        Args:
            snr_value: SNR level (e.g. 3, 5, 10, 15, 25)
            n_phantoms: Number of test phantoms to generate
            phantom_size: Size of each phantom (default 64)

        Returns:
            dict with keys: all_nn_cbf, all_nn_att, all_true_cbf, all_true_att,
                           all_ls_cbf, all_ls_att, all_ls_true_cbf, all_ls_true_att,
                           all_nn_at_ls_cbf, all_nn_at_ls_att,
                           all_smoothed_ls_cbf, all_smoothed_ls_att,
                           all_smoothed_ls_true_cbf, all_smoothed_ls_true_att,
                           all_nn_at_sls_cbf, all_nn_at_sls_att
        """
        from scipy.ndimage import gaussian_filter

        all_nn_cbf, all_ls_cbf = [], []
        all_nn_att, all_ls_att = [], []
        all_true_cbf, all_true_att = [], []
        all_ls_true_cbf, all_ls_true_att = [], []
        all_nn_at_ls_cbf, all_nn_at_ls_att = [], []
        all_smoothed_ls_cbf, all_smoothed_ls_att = [], []
        all_smoothed_ls_true_cbf, all_smoothed_ls_true_att = [], []
        all_nn_at_sls_cbf, all_nn_at_sls_att = [], []

        phantom_gen = SpatialPhantomGenerator(size=phantom_size, pve_sigma=1.0)

        for phantom_idx in range(n_phantoms):
            # Generate tissue-structured phantom (matches training data)
            np.random.seed(phantom_idx)
            true_cbf_map, true_att_map, metadata = phantom_gen.generate_phantom(include_pathology=True)

            # Create brain mask from non-zero CBF regions
            mask = (true_cbf_map > 1.0).astype(np.float32)

            # Generate clean signals for each voxel
            signals = np.zeros((len(self.plds) * 2, phantom_size, phantom_size), dtype=np.float32)

            for i in range(phantom_size):
                for j in range(phantom_size):
                    if mask[i, j] > 0:
                        cbf, att = true_cbf_map[i, j], true_att_map[i, j]
                        pcasl = self.simulator._generate_pcasl_signal(
                            self.plds, att, cbf, self.params.T1_artery,
                            self.params.T_tau, self.params.alpha_PCASL
                        )
                        vsasl = self.simulator._generate_vsasl_signal(
                            self.plds, att, cbf, self.params.T1_artery, self.params.alpha_VSASL
                        )
                        signals[:len(self.plds), i, j] = pcasl
                        signals[len(self.plds):, i, j] = vsasl

            # Add noise at the specified SNR level
            ref_signal = self.simulator._compute_reference_signal()
            noise_sd = ref_signal / snr_value
            # Use a different random seed per SNR to get independent noise realizations
            # but reproducible across runs: seed = phantom_idx * 1000 + int(snr_value * 10)
            noise_rng = np.random.RandomState(phantom_idx * 1000 + int(snr_value * 10))
            noise = noise_sd * noise_rng.randn(*signals.shape).astype(np.float32)
            noisy_signals = signals + noise

            # Scale signals (matching SpatialDataset M0 normalization)
            noisy_signals_scaled = noisy_signals * 100.0

            # Apply normalization matching training config
            normalization_mode = self.config.get('normalization_mode', 'per_curve')
            global_scale_factor = self.config.get('global_scale_factor', 1.0)

            if normalization_mode == 'global_scale':
                noisy_signals_normalized = noisy_signals_scaled * global_scale_factor
            else:
                temporal_mean = np.mean(noisy_signals_scaled, axis=0, keepdims=True)
                temporal_std = np.std(noisy_signals_scaled, axis=0, keepdims=True) + 1e-6
                noisy_signals_normalized = (noisy_signals_scaled - temporal_mean) / temporal_std

            # --- NN Inference ---
            input_tensor = torch.from_numpy(noisy_signals_normalized[np.newaxis, ...]).float().to(self.device)

            with torch.no_grad():
                nn_cbf_maps, nn_att_maps = [], []
                for model in self.models:
                    cbf_pred, att_pred, _, _ = model(input_tensor)
                    cbf_denorm = cbf_pred * self.norm_stats['y_std_cbf'] + self.norm_stats['y_mean_cbf']
                    att_denorm = att_pred * self.norm_stats['y_std_att'] + self.norm_stats['y_mean_att']
                    cbf_denorm = torch.clamp(cbf_denorm, min=0.0, max=200.0)
                    att_denorm = torch.clamp(att_denorm, min=0.0, max=5000.0)
                    nn_cbf_maps.append(cbf_denorm.cpu().numpy())
                    nn_att_maps.append(att_denorm.cpu().numpy())

                nn_cbf = np.mean(nn_cbf_maps, axis=0)[0, 0]  # (H, W)
                nn_att = np.mean(nn_att_maps, axis=0)[0, 0]

            # --- LS Inference (voxel-by-voxel) ---
            ls_cbf = np.full((phantom_size, phantom_size), np.nan)
            ls_att = np.full((phantom_size, phantom_size), np.nan)

            brain_indices = np.argwhere(mask > 0)
            sample_indices = brain_indices[::10]

            ls_params = {
                'T1_artery': self.params.T1_artery,
                'T_tau': self.params.T_tau,
                'alpha_PCASL': self.params.alpha_PCASL,
                'alpha_VSASL': self.params.alpha_VSASL,
                'T2_factor': self.params.T2_factor,
                'alpha_BS1': self.params.alpha_BS1
            }
            pldti = np.column_stack([self.plds, self.plds])

            for idx in sample_indices:
                i, j = idx
                voxel_signal = noisy_signals[:, i, j]

                try:
                    init_guess = get_grid_search_initial_guess(voxel_signal, self.plds, ls_params)
                    signal_reshaped = voxel_signal.reshape((len(self.plds), 2), order='F')
                    beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, signal_reshaped, init_guess, **ls_params
                    )
                    ls_cbf[i, j] = beta[0] * 6000.0
                    ls_att[i, j] = beta[1]
                except Exception:
                    pass

            # --- Smoothed-LS Inference ---
            smoothed_signals = np.zeros_like(noisy_signals)
            smooth_sigma = 2.0
            for ch in range(noisy_signals.shape[0]):
                smoothed_signals[ch] = gaussian_filter(noisy_signals[ch], sigma=smooth_sigma)

            sls_cbf = np.full((phantom_size, phantom_size), np.nan)
            sls_att = np.full((phantom_size, phantom_size), np.nan)

            for idx in sample_indices:
                i, j = idx
                voxel_signal = smoothed_signals[:, i, j]
                try:
                    init_guess = get_grid_search_initial_guess(voxel_signal, self.plds, ls_params)
                    signal_reshaped = voxel_signal.reshape((len(self.plds), 2), order='F')
                    beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                        pldti, signal_reshaped, init_guess, **ls_params
                    )
                    sls_cbf[i, j] = beta[0] * 6000.0
                    sls_att[i, j] = beta[1]
                except Exception:
                    pass

            # Collect smoothed-LS values at sampled voxels
            for idx in sample_indices:
                i, j = idx
                if not np.isnan(sls_cbf[i, j]):
                    all_smoothed_ls_cbf.append(sls_cbf[i, j])
                    all_smoothed_ls_att.append(sls_att[i, j])
                    all_smoothed_ls_true_cbf.append(true_cbf_map[i, j])
                    all_smoothed_ls_true_att.append(true_att_map[i, j])
                    all_nn_at_sls_cbf.append(nn_cbf[i, j])
                    all_nn_at_sls_att.append(nn_att[i, j])

            # Collect brain voxel values for statistics
            brain_mask_bool = mask > 0
            all_nn_cbf.extend(nn_cbf[brain_mask_bool].flatten())
            all_nn_att.extend(nn_att[brain_mask_bool].flatten())
            all_true_cbf.extend(true_cbf_map[brain_mask_bool].flatten())
            all_true_att.extend(true_att_map[brain_mask_bool].flatten())

            # For LS, only use sampled voxels
            for idx in sample_indices:
                i, j = idx
                if not np.isnan(ls_cbf[i, j]):
                    all_ls_cbf.append(ls_cbf[i, j])
                    all_ls_att.append(ls_att[i, j])
                    all_ls_true_cbf.append(true_cbf_map[i, j])
                    all_ls_true_att.append(true_att_map[i, j])
                    all_nn_at_ls_cbf.append(nn_cbf[i, j])
                    all_nn_at_ls_att.append(nn_att[i, j])

            if phantom_idx == 0:
                logger.info(f"  Phantom 0 (SNR={snr_value}) - NN CBF: mean={nn_cbf[brain_mask_bool].mean():.2f}, "
                           f"std={nn_cbf[brain_mask_bool].std():.2f}")
                logger.info(f"  Phantom 0 (SNR={snr_value}) - True CBF: mean={true_cbf_map[brain_mask_bool].mean():.2f}, "
                           f"std={true_cbf_map[brain_mask_bool].std():.2f}")
                logger.info(f"  Phantom 0 (SNR={snr_value}) - NN ATT: mean={nn_att[brain_mask_bool].mean():.2f}, "
                           f"std={nn_att[brain_mask_bool].std():.2f}")
                logger.info(f"  Phantom 0 (SNR={snr_value}) - True ATT: mean={true_att_map[brain_mask_bool].mean():.2f}, "
                           f"std={true_att_map[brain_mask_bool].std():.2f}")

        return {
            'all_nn_cbf': np.array(all_nn_cbf),
            'all_nn_att': np.array(all_nn_att),
            'all_true_cbf': np.array(all_true_cbf),
            'all_true_att': np.array(all_true_att),
            'all_ls_cbf': np.array(all_ls_cbf),
            'all_ls_att': np.array(all_ls_att),
            'all_ls_true_cbf': np.array(all_ls_true_cbf),
            'all_ls_true_att': np.array(all_ls_true_att),
            'all_nn_at_ls_cbf': np.array(all_nn_at_ls_cbf),
            'all_nn_at_ls_att': np.array(all_nn_at_ls_att),
            'all_smoothed_ls_cbf': np.array(all_smoothed_ls_cbf),
            'all_smoothed_ls_att': np.array(all_smoothed_ls_att),
            'all_smoothed_ls_true_cbf': np.array(all_smoothed_ls_true_cbf),
            'all_smoothed_ls_true_att': np.array(all_smoothed_ls_true_att),
            'all_nn_at_sls_cbf': np.array(all_nn_at_sls_cbf),
            'all_nn_at_sls_att': np.array(all_nn_at_sls_att),
        }

    def _compute_snr_metrics_with_ci(self, snr_value, results):
        """
        Compute metrics with bootstrap CIs for a single SNR level's results.

        Args:
            snr_value: The SNR value (for labeling)
            results: Dict returned by _run_spatial_at_snr

        Returns:
            Dict of metrics with CIs
        """
        all_nn_cbf = results['all_nn_cbf']
        all_nn_att = results['all_nn_att']
        all_true_cbf = results['all_true_cbf']
        all_true_att = results['all_true_att']
        all_nn_at_ls_cbf = results['all_nn_at_ls_cbf']
        all_nn_at_ls_att = results['all_nn_at_ls_att']
        all_ls_cbf = results['all_ls_cbf']
        all_ls_att = results['all_ls_att']
        all_ls_true_cbf = results['all_ls_true_cbf']
        all_ls_true_att = results['all_ls_true_att']

        # Per-voxel absolute errors for bootstrap
        nn_cbf_errors = np.abs(all_nn_cbf - all_true_cbf)
        nn_att_errors = np.abs(all_nn_att - all_true_att)

        # NN metrics with CIs (full brain)
        nn_cbf_mae_mean, nn_cbf_mae_lo, nn_cbf_mae_hi = self._bootstrap_ci(nn_cbf_errors)
        nn_att_mae_mean, nn_att_mae_lo, nn_att_mae_hi = self._bootstrap_ci(nn_att_errors)
        nn_cbf_bias = float(np.mean(all_nn_cbf - all_true_cbf))
        nn_att_bias = float(np.mean(all_nn_att - all_true_att))

        metrics = {
            'snr': snr_value,
            'nn_cbf_mae': nn_cbf_mae_mean,
            'nn_cbf_mae_ci': [nn_cbf_mae_lo, nn_cbf_mae_hi],
            'nn_cbf_bias': nn_cbf_bias,
            'nn_att_mae': nn_att_mae_mean,
            'nn_att_mae_ci': [nn_att_mae_lo, nn_att_mae_hi],
            'nn_att_bias': nn_att_bias,
        }

        # LS metrics with CIs (at spatially matched voxels)
        if len(all_ls_cbf) > 0:
            ls_cbf_errors = np.abs(all_ls_cbf - all_ls_true_cbf)
            ls_att_errors = np.abs(all_ls_att - all_ls_true_att)
            ls_cbf_mae_mean, ls_cbf_mae_lo, ls_cbf_mae_hi = self._bootstrap_ci(ls_cbf_errors)
            ls_att_mae_mean, ls_att_mae_lo, ls_att_mae_hi = self._bootstrap_ci(ls_att_errors)

            metrics['ls_cbf_mae'] = ls_cbf_mae_mean
            metrics['ls_cbf_mae_ci'] = [ls_cbf_mae_lo, ls_cbf_mae_hi]
            metrics['ls_att_mae'] = ls_att_mae_mean
            metrics['ls_att_mae_ci'] = [ls_att_mae_lo, ls_att_mae_hi]

            # Win rate with CIs (at matched voxels)
            nn_matched_cbf_errors = np.abs(all_nn_at_ls_cbf - all_ls_true_cbf)
            nn_matched_att_errors = np.abs(all_nn_at_ls_att - all_ls_true_att)

            cbf_wr, cbf_wr_lo, cbf_wr_hi = self._bootstrap_ci_winrate(
                nn_matched_cbf_errors, ls_cbf_errors)
            att_wr, att_wr_lo, att_wr_hi = self._bootstrap_ci_winrate(
                nn_matched_att_errors, ls_att_errors)

            metrics['cbf_win_rate'] = cbf_wr
            metrics['cbf_win_rate_ci'] = [cbf_wr_lo, cbf_wr_hi]
            metrics['att_win_rate'] = att_wr
            metrics['att_win_rate_ci'] = [att_wr_lo, att_wr_hi]
            metrics['n_ls_samples'] = len(all_ls_cbf)

        metrics['n_nn_voxels'] = len(all_nn_cbf)

        return metrics

    def run_spatial_validation(self, multi_snr=True):
        """
        Validate spatial (U-Net) models by generating synthetic 2D phantoms
        and comparing NN predictions to Least Squares fitting.

        Args:
            multi_snr: If True (default), run validation at SNR = [3, 5, 10, 15, 25]
                      and produce a multi-SNR summary. If False, run only at SNR=10
                      (backwards-compatible behavior).
        """
        logger.info("=" * 60)
        logger.info("SPATIAL MODEL VALIDATION")
        logger.info("=" * 60)

        phantom_size = 64

        if multi_snr:
            snr_values = [3, 5, 10, 15, 25]
            n_phantoms_multi = 20  # Reduced for multi-SNR to keep runtime reasonable
        else:
            snr_values = [10]
            n_phantoms_multi = 50  # Full count for single-SNR

        multi_snr_results = {}

        for snr_val in snr_values:
            n_phantoms = n_phantoms_multi if multi_snr else 50
            # For SNR=10 in multi-SNR mode, use full 50 phantoms for backward-compatible report
            if multi_snr and snr_val == 10:
                n_phantoms = 50

            logger.info("-" * 40)
            logger.info(f"Running spatial validation at SNR={snr_val} with {n_phantoms} phantoms...")
            logger.info("-" * 40)

            results = self._run_spatial_at_snr(snr_val, n_phantoms, phantom_size)
            metrics = self._compute_snr_metrics_with_ci(snr_val, results)
            multi_snr_results[snr_val] = metrics

            # --- Log detailed results for this SNR ---
            logger.info(f"--- SNR={snr_val} RESULTS ---")

            # NN metrics with CIs
            logger.info(f"  NN CBF - MAE: {metrics['nn_cbf_mae']:.2f} "
                        f"[{metrics['nn_cbf_mae_ci'][0]:.2f}, {metrics['nn_cbf_mae_ci'][1]:.2f}], "
                        f"Bias: {metrics['nn_cbf_bias']:.2f}")
            logger.info(f"  NN ATT - MAE: {metrics['nn_att_mae']:.2f} "
                        f"[{metrics['nn_att_mae_ci'][0]:.2f}, {metrics['nn_att_mae_ci'][1]:.2f}], "
                        f"Bias: {metrics['nn_att_bias']:.2f}")

            if 'ls_cbf_mae' in metrics:
                logger.info(f"  LS CBF - MAE: {metrics['ls_cbf_mae']:.2f} "
                            f"[{metrics['ls_cbf_mae_ci'][0]:.2f}, {metrics['ls_cbf_mae_ci'][1]:.2f}]")
                logger.info(f"  LS ATT - MAE: {metrics['ls_att_mae']:.2f} "
                            f"[{metrics['ls_att_mae_ci'][0]:.2f}, {metrics['ls_att_mae_ci'][1]:.2f}]")
                logger.info(f"  CBF Win Rate: {metrics['cbf_win_rate']:.1%} "
                            f"[{metrics['cbf_win_rate_ci'][0]:.1%}, {metrics['cbf_win_rate_ci'][1]:.1%}]")
                logger.info(f"  ATT Win Rate: {metrics['att_win_rate']:.1%} "
                            f"[{metrics['att_win_rate_ci'][0]:.1%}, {metrics['att_win_rate_ci'][1]:.1%}]")
                logger.info(f"  LS samples: {metrics['n_ls_samples']}, NN voxels: {metrics['n_nn_voxels']}")

            # --- For SNR=10, log to LLM report for backwards compatibility ---
            if snr_val == 10:
                all_nn_at_ls_cbf = results['all_nn_at_ls_cbf']
                all_nn_at_ls_att = results['all_nn_at_ls_att']
                all_ls_cbf = results['all_ls_cbf']
                all_ls_att = results['all_ls_att']
                all_ls_true_cbf = results['all_ls_true_cbf']
                all_ls_true_att = results['all_ls_true_att']

                self._log_llm_metrics("Spatial_SNR10", all_nn_at_ls_cbf,
                                      all_ls_cbf, all_ls_true_cbf, "CBF")
                self._log_llm_metrics("Spatial_SNR10", all_nn_at_ls_att,
                                      all_ls_att, all_ls_true_att, "ATT")

                # Smoothed-LS comparison
                all_smoothed_ls_cbf = results['all_smoothed_ls_cbf']
                all_smoothed_ls_att = results['all_smoothed_ls_att']
                all_smoothed_ls_true_cbf = results['all_smoothed_ls_true_cbf']
                all_smoothed_ls_true_att = results['all_smoothed_ls_true_att']
                all_nn_at_sls_cbf = results['all_nn_at_sls_cbf']
                all_nn_at_sls_att = results['all_nn_at_sls_att']

                if len(all_smoothed_ls_cbf) > 0:
                    self._log_llm_metrics("Spatial_SNR10_SmoothedLS", all_nn_at_sls_cbf,
                                          all_smoothed_ls_cbf, all_smoothed_ls_true_cbf, "CBF")
                    self._log_llm_metrics("Spatial_SNR10_SmoothedLS", all_nn_at_sls_att,
                                          all_smoothed_ls_att, all_smoothed_ls_true_att, "ATT")

        # --- Multi-SNR Summary Table ---
        if multi_snr and len(snr_values) > 1:
            logger.info("")
            logger.info("=" * 80)
            logger.info("MULTI-SNR SUMMARY TABLE")
            logger.info("=" * 80)
            header = (f"{'SNR':>5} | {'NN CBF MAE':>20} | {'LS CBF MAE':>20} | "
                      f"{'CBF WinRate':>20} | {'NN ATT MAE':>20} | {'LS ATT MAE':>20} | "
                      f"{'ATT WinRate':>20}")
            logger.info(header)
            logger.info("-" * len(header))

            for snr_val in snr_values:
                m = multi_snr_results[snr_val]
                nn_cbf_str = f"{m['nn_cbf_mae']:.2f} [{m['nn_cbf_mae_ci'][0]:.2f},{m['nn_cbf_mae_ci'][1]:.2f}]"
                nn_att_str = f"{m['nn_att_mae']:.2f} [{m['nn_att_mae_ci'][0]:.2f},{m['nn_att_mae_ci'][1]:.2f}]"

                if 'ls_cbf_mae' in m:
                    ls_cbf_str = f"{m['ls_cbf_mae']:.2f} [{m['ls_cbf_mae_ci'][0]:.2f},{m['ls_cbf_mae_ci'][1]:.2f}]"
                    ls_att_str = f"{m['ls_att_mae']:.2f} [{m['ls_att_mae_ci'][0]:.2f},{m['ls_att_mae_ci'][1]:.2f}]"
                    cbf_wr_str = f"{m['cbf_win_rate']:.1%} [{m['cbf_win_rate_ci'][0]:.1%},{m['cbf_win_rate_ci'][1]:.1%}]"
                    att_wr_str = f"{m['att_win_rate']:.1%} [{m['att_win_rate_ci'][0]:.1%},{m['att_win_rate_ci'][1]:.1%}]"
                else:
                    ls_cbf_str = "N/A"
                    ls_att_str = "N/A"
                    cbf_wr_str = "N/A"
                    att_wr_str = "N/A"

                logger.info(f"{snr_val:>5} | {nn_cbf_str:>20} | {ls_cbf_str:>20} | "
                            f"{cbf_wr_str:>20} | {nn_att_str:>20} | {ls_att_str:>20} | "
                            f"{att_wr_str:>20}")

            logger.info("=" * 80)

            # --- Save multi-SNR results to JSON ---
            multi_snr_json = {}
            for snr_val, m in multi_snr_results.items():
                # Convert all values to JSON-safe types
                snr_key = str(snr_val)
                multi_snr_json[snr_key] = {}
                for k, v in m.items():
                    if isinstance(v, (np.floating, np.integer)):
                        multi_snr_json[snr_key][k] = float(v)
                    elif isinstance(v, np.ndarray):
                        multi_snr_json[snr_key][k] = v.tolist()
                    elif isinstance(v, list):
                        multi_snr_json[snr_key][k] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in v]
                    else:
                        multi_snr_json[snr_key][k] = v

            json_path = self.output_dir / "multi_snr_results.json"
            with open(json_path, 'w') as f:
                json.dump(multi_snr_json, f, indent=4)
            logger.info(f"Saved multi-SNR results to {json_path}")

        logger.info("=" * 60)

    def run_phase_3(self, multi_snr=True):
        logger.info("--- Phase 3: Comprehensive Stats ---")

        # Handle spatial models with dedicated validation
        if hasattr(self, 'is_spatial') and self.is_spatial:
            self.run_spatial_validation(multi_snr=multi_snr)
            self.save_llm_report()
            self.save_plot_data_json()
            return
        n = 500
        
        # --- DIAGNOSTIC: Print model output statistics ---
        logger.info("=" * 60)
        logger.info("DIAGNOSTIC: Checking model predictions...")
        logger.info("=" * 60)

        # A: Fixed CBF, Varying ATT
        # NOTE: ATT constrained to max PLD (3000ms) to ensure signals are measurable
        t_cbf = np.full(n, 60.0); t_att = np.linspace(500, 3000, n)
        sigs = [self.generate_noise(np.concatenate([
            self.simulator._generate_pcasl_signal(self.plds, a, c, self.params.T1_artery, self.params.T_tau, self.params.alpha_PCASL),
            self.simulator._generate_vsasl_signal(self.plds, a, c, self.params.T1_artery, self.params.alpha_VSASL)
        ]), 10.0) for c, a in zip(t_cbf, t_att)]
        sigs = np.array(sigs)
        nn_c, nn_a = self.run_nn_inference(sigs, np.full(n, self.params.T1_artery))  # 3T consensus (Alsop 2015)
        ls_c, ls_a = self.run_ls_inference(sigs, self.params.T1_artery)

        # Diagnostic output
        logger.info(f"Scenario A (Fixed CBF=60, Varying ATT):")
        logger.info(f"  NN CBF predictions: mean={nn_c.mean():.2f}, std={nn_c.std():.2f}, min={nn_c.min():.2f}, max={nn_c.max():.2f}")
        logger.info(f"  NN ATT predictions: mean={nn_a.mean():.2f}, std={nn_a.std():.2f}, min={nn_a.min():.2f}, max={nn_a.max():.2f}")
        logger.info(f"  True CBF: {t_cbf[0]:.2f} (constant)")
        logger.info(f"  True ATT: range [{t_att.min():.0f}, {t_att.max():.0f}]")
        logger.info(f"  LS CBF: mean={np.nanmean(ls_c):.2f}, valid={np.sum(~np.isnan(ls_c))}/{n}")
        
        # --- NEW: Log Stats ---
        self._log_llm_metrics("A_FixedCBF_VarATT", nn_c, ls_c, t_cbf, "CBF")
        self._log_llm_metrics("A_FixedCBF_VarATT", nn_a, ls_a, t_att, "ATT")
        # ----------------------
        
        self._plot_scenario("A_FixedCBF_VarATT", t_att, "ATT (ms)", nn_c, nn_a, ls_c, ls_a, t_cbf, t_att)

        # B: Fixed ATT, Varying CBF
        t_cbf = np.linspace(20, 120, n); t_att = np.full(n, 1500.0)
        sigs = [self.generate_noise(np.concatenate([
            self.simulator._generate_pcasl_signal(self.plds, a, c, self.params.T1_artery, self.params.T_tau, self.params.alpha_PCASL),
            self.simulator._generate_vsasl_signal(self.plds, a, c, self.params.T1_artery, self.params.alpha_VSASL)
        ]), 10.0) for c, a in zip(t_cbf, t_att)]
        sigs = np.array(sigs)
        nn_c, nn_a = self.run_nn_inference(sigs, np.full(n, self.params.T1_artery))  # 3T consensus (Alsop 2015)
        ls_c, ls_a = self.run_ls_inference(sigs, self.params.T1_artery)

        # --- NEW: Log Stats ---
        self._log_llm_metrics("B_FixedATT_VarCBF", nn_c, ls_c, t_cbf, "CBF")
        self._log_llm_metrics("B_FixedATT_VarCBF", nn_a, ls_a, t_att, "ATT")
        # ----------------------

        self._plot_scenario("B_FixedATT_VarCBF", t_cbf, "CBF", nn_c, nn_a, ls_c, ls_a, t_cbf, t_att)
        
        # C & D
        # NOTE: ATT constrained to max PLD (3000ms) to ensure signals are measurable
        t_cbf = np.random.uniform(20, 100, n); t_att = np.random.uniform(500, 3000, n)
        for snr, label in [(10.0, "C_VarBoth_SNR10"), (3.0, "D_VarBoth_SNR3")]:
            sigs = [self.generate_noise(np.concatenate([
                self.simulator._generate_pcasl_signal(self.plds, a, c, self.params.T1_artery, self.params.T_tau, self.params.alpha_PCASL),
                self.simulator._generate_vsasl_signal(self.plds, a, c, self.params.T1_artery, self.params.alpha_VSASL)
            ]), snr) for c, a in zip(t_cbf, t_att)]
            sigs = np.array(sigs)
            nn_c, nn_a = self.run_nn_inference(sigs, np.full(n, self.params.T1_artery))  # 3T consensus (Alsop 2015)
            ls_c, ls_a = self.run_ls_inference(sigs, self.params.T1_artery)

            # --- NEW: Log Stats ---
            self._log_llm_metrics(label, nn_c, ls_c, t_cbf, "CBF")
            self._log_llm_metrics(label, nn_a, ls_a, t_att, "ATT")
            # ----------------------

            self._plot_scenario(label, t_att, "ATT (ms)", nn_c, nn_a, ls_c, ls_a, t_cbf, t_att)
            
        # --- Save Reports ---
        self.save_llm_report()
        self.save_plot_data_json()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="validation_results")
    parser.add_argument("--no-multi-snr", action="store_true", default=False,
                        help="Disable multi-SNR validation curve (only run SNR=10)")
    args = parser.parse_args()

    try:
        val = ASLValidator(args.run_dir, args.output_dir)
        val.run_phase_1()
        val.run_phase_3(multi_snr=not args.no_multi_snr)
        print("\n--- [SUCCESS] Validation Finished. Check output dir. ---")
    except KeyboardInterrupt:
        print("\n--- [CANCELLED] User stopped the script. ---")
    except Exception as e:
        print(f"\n!!! [FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()