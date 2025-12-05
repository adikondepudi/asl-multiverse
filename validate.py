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
    from asl_simulation import ASLParameters
    from enhanced_simulation import RealisticASLSimulator
    from enhanced_asl_network import DisentangledASLNet
    from utils import engineer_signal_features, process_signals_dynamic, get_grid_search_initial_guess
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
            T1_artery=self.config.get('T1_artery', 1850.0),
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

        # 4. Load Ensemble Models
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
        
        # Extract Architecture Params
        hidden_sizes = self.config.get('hidden_sizes', [128, 64, 32])
        d_model = self.config.get('transformer_d_model_focused', 32)
        nhead = self.config.get('transformer_nhead_model', 4)
        moe_config = self.config.get('moe', None)
        encoder_type = self.config.get('encoder_type', 'physics_processor')  # <-- NEW
        
        # --- AUTO-DETECT NUM_SCALAR_FEATURES ---
        # Peek at the first model to find the input dimension of the FiLM layer
        try:
            first_state = torch.load(model_files[0], map_location='cpu')
            sd = first_state['model_state_dict'] if 'model_state_dict' in first_state else first_state
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
                encoder_type=encoder_type  # <-- NEW
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
        print(f"   > Generating plots for {scenario_name}")
        x_values = np.array(x_values)
        
        def get_stats(preds, truths, x_vals):
            bins = 15
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='Mean of empty slice', category=RuntimeWarning)
                
                bin_bias = binned_statistic(x_vals, preds - truths, statistic=np.nanmean, bins=bins)
                bin_std = binned_statistic(x_vals, preds, statistic=np.nanstd, bins=bins)
                bin_truth_mean = binned_statistic(x_vals, truths, statistic=np.nanmean, bins=bins).statistic
                
                cov_stat = (bin_std.statistic / (bin_truth_mean + 1e-6)) * 100
                bin_centers = 0.5 * (bin_bias.bin_edges[1:] + bin_bias.bin_edges[:-1])
                return bin_centers, bin_bias.statistic, cov_stat

        nn_x, nn_bias, nn_cov = get_stats(nn_cbf, true_cbf, x_values)
        ls_x, ls_bias, ls_cov = get_stats(ls_cbf, true_cbf, x_values)
        _, nn_bias_att, nn_cov_att = get_stats(nn_att, true_att, x_values)
        _, ls_bias_att, ls_cov_att = get_stats(ls_att, true_att, x_values)

        def make_plot(metric_name, nn_y, ls_y, ylabel):
            plt.figure(figsize=(6, 4))
            plt.plot(nn_x, nn_y, 'o-', color='purple', label=f'Neural Net', linewidth=2, markersize=5)
            
            valid_ls = ~np.isnan(ls_y)
            if np.sum(valid_ls) > 0:
                plt.plot(ls_x[valid_ls], ls_y[valid_ls], 'x--', color='gray', label='LS', linewidth=1.5, markersize=5)
            
            plt.xlabel(x_label); plt.ylabel(ylabel); plt.title(f"{scenario_name}: {metric_name}")
            plt.legend(); plt.grid(True, alpha=0.3); plt.axhline(0, color='black', linewidth=0.5)
            plt.savefig(self.output_dir / f"{scenario_name}_{metric_name.replace(' ', '')}.png", dpi=150)
            plt.close()

        make_plot("CBF Bias", nn_bias, ls_bias, "Error")
        make_plot("CBF CoV", nn_cov, ls_cov, "CoV (%)")
        make_plot("ATT Bias", nn_bias_att, ls_bias_att, "Error")
        make_plot("ATT CoV", nn_cov_att, ls_cov_att, "CoV (%)")

    def run_phase_3(self):
        logger.info("--- Phase 3: Comprehensive Stats ---")
        n = 500
        
        # A: Fixed CBF, Varying ATT
        t_cbf = np.full(n, 60.0); t_att = np.linspace(500, 4000, n)
        sigs = [self.generate_noise(np.concatenate([
            self.simulator._generate_pcasl_signal(self.plds, a, c, 1850, 1800, 0.85),
            self.simulator._generate_vsasl_signal(self.plds, a, c, 1850, 0.56)
        ]), 10.0) for c, a in zip(t_cbf, t_att)]
        sigs = np.array(sigs)
        nn_c, nn_a = self.run_nn_inference(sigs, np.full(n, 1850.0))
        ls_c, ls_a = self.run_ls_inference(sigs, 1850.0)
        
        # --- NEW: Log Stats ---
        self._log_llm_metrics("A_FixedCBF_VarATT", nn_c, ls_c, t_cbf, "CBF")
        self._log_llm_metrics("A_FixedCBF_VarATT", nn_a, ls_a, t_att, "ATT")
        # ----------------------
        
        self._plot_scenario("A_FixedCBF_VarATT", t_att, "ATT (ms)", nn_c, nn_a, ls_c, ls_a, t_cbf, t_att)

        # B: Fixed ATT, Varying CBF
        t_cbf = np.linspace(20, 120, n); t_att = np.full(n, 1500.0)
        sigs = [self.generate_noise(np.concatenate([
            self.simulator._generate_pcasl_signal(self.plds, a, c, 1850, 1800, 0.85),
            self.simulator._generate_vsasl_signal(self.plds, a, c, 1850, 0.56)
        ]), 10.0) for c, a in zip(t_cbf, t_att)]
        sigs = np.array(sigs)
        nn_c, nn_a = self.run_nn_inference(sigs, np.full(n, 1850.0))
        ls_c, ls_a = self.run_ls_inference(sigs, 1850.0)

        # --- NEW: Log Stats ---
        self._log_llm_metrics("B_FixedATT_VarCBF", nn_c, ls_c, t_cbf, "CBF")
        self._log_llm_metrics("B_FixedATT_VarCBF", nn_a, ls_a, t_att, "ATT")
        # ----------------------

        self._plot_scenario("B_FixedATT_VarCBF", t_cbf, "CBF", nn_c, nn_a, ls_c, ls_a, t_cbf, t_att)
        
        # C & D
        t_cbf = np.random.uniform(20, 100, n); t_att = np.random.uniform(500, 4000, n)
        for snr, label in [(10.0, "C_VarBoth_SNR10"), (3.0, "D_VarBoth_SNR3")]:
            sigs = [self.generate_noise(np.concatenate([
                self.simulator._generate_pcasl_signal(self.plds, a, c, 1850, 1800, 0.85),
                self.simulator._generate_vsasl_signal(self.plds, a, c, 1850, 0.56)
            ]), snr) for c, a in zip(t_cbf, t_att)]
            sigs = np.array(sigs)
            nn_c, nn_a = self.run_nn_inference(sigs, np.full(n, 1850.0))
            ls_c, ls_a = self.run_ls_inference(sigs, 1850.0)
            
            # --- NEW: Log Stats ---
            self._log_llm_metrics(label, nn_c, ls_c, t_cbf, "CBF")
            self._log_llm_metrics(label, nn_a, ls_a, t_att, "ATT")
            # ----------------------

            self._plot_scenario(label, t_att, "ATT (ms)", nn_c, nn_a, ls_c, ls_a, t_cbf, t_att)
            
        # --- NEW: Save the Report ---
        self.save_llm_report()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="validation_results")
    args = parser.parse_args()
    
    try:
        val = ASLValidator(args.run_dir, args.output_dir)
        val.run_phase_1()
        val.run_phase_3()
        print("\n--- [SUCCESS] Validation Finished. Check output dir. ---")
    except KeyboardInterrupt:
        print("\n--- [CANCELLED] User stopped the script. ---")
    except Exception as e:
        print(f"\n!!! [FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()