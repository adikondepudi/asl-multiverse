#!/usr/bin/env python3
"""
Run validation on all 10 amplitude ablation experiments and generate consolidated report.
"""

import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

def run_validation(exp_dir):
    """Run validation on a single experiment."""
    exp_name = exp_dir.name
    output_dir = Path(f"validation_results/{exp_name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"Running validation: {exp_name}")
    print(f"{'='*70}")

    try:
        result = subprocess.run(
            [
                "python3", "validate.py",
                "--run_dir", str(exp_dir),
                "--output_dir", str(output_dir)
            ],
            capture_output=True,
            text=True,
            timeout=600
        )

        if result.returncode == 0:
            print(f"✅ {exp_name} - SUCCESS")
            return {
                "status": "success",
                "output_dir": str(output_dir),
                "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout
            }
        else:
            print(f"❌ {exp_name} - FAILED")
            print(f"Error output:\n{result.stderr[-500:]}")
            return {
                "status": "failed",
                "error": result.stderr[-500:] if len(result.stderr) > 500 else result.stderr
            }
    except subprocess.TimeoutExpired:
        print(f"⏱️  {exp_name} - TIMEOUT")
        return {
            "status": "timeout",
            "error": "Validation exceeded 10 minute timeout"
        }
    except Exception as e:
        print(f"❌ {exp_name} - ERROR: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }

def load_validation_metrics(output_dir):
    """Load validation metrics from llm_analysis_report.json."""
    metrics_file = Path(output_dir) / "llm_analysis_report.json"

    if not metrics_file.exists():
        return None

    try:
        with open(metrics_file, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"  Warning: Could not load metrics from {metrics_file}: {e}")
        return None

def extract_cbf_att_metrics(metrics_data):
    """Extract CBF and ATT metrics from validation report."""
    if not metrics_data:
        return None

    results = {}

    # Metrics are organized as: scenario -> parameter -> metrics
    for scenario, params in metrics_data.items():
        if 'CBF' in params:
            cbf_data = params['CBF']['Neural_Net']
            att_data = params['ATT']['Neural_Net']
            cbf_win_rate = params['CBF'].get('NN_vs_LS_Win_Rate', None)
            att_win_rate = params['ATT'].get('NN_vs_LS_Win_Rate', None)

            results[scenario] = {
                'cbf_mae': cbf_data.get('MAE', None),
                'cbf_bias': cbf_data.get('Bias', None),
                'att_mae': att_data.get('MAE', None),
                'att_bias': att_data.get('Bias', None),
                'cbf_win_rate': cbf_win_rate,
                'att_win_rate': att_win_rate,
            }

    return results

def main():
    base_dir = Path("amplitude_ablation_v1")
    exp_dirs = sorted([d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(('0', '1'))])

    print(f"\nFound {len(exp_dirs)} experiments to validate:")
    for exp_dir in exp_dirs:
        print(f"  - {exp_dir.name}")

    # Run validations
    results = {}
    for exp_dir in exp_dirs:
        exp_name = exp_dir.name
        validation_result = run_validation(exp_dir)
        results[exp_name] = validation_result

        # Try to load metrics if successful
        if validation_result["status"] == "success":
            metrics = load_validation_metrics(validation_result["output_dir"])
            metrics_summary = extract_cbf_att_metrics(metrics)
            if metrics_summary:
                validation_result["metrics"] = metrics_summary

    # Generate summary report
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print(f"{'='*70}\n")

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_experiments": len(exp_dirs),
        "successful": sum(1 for r in results.values() if r["status"] == "success"),
        "failed": sum(1 for r in results.values() if r["status"] != "success"),
        "experiments": results
    }

    # Print summary table
    print(f"{'Experiment':<40} {'Status':<15} {'CBF MAE':<12} {'ATT MAE':<12}")
    print("-" * 80)

    for exp_name in sorted(results.keys()):
        result = results[exp_name]
        status = result["status"].upper()

        cbf_mae = "N/A"
        att_mae = "N/A"

        if "metrics" in result and result["metrics"]:
            # Get first scenario metrics
            first_scenario = list(result["metrics"].values())[0]
            cbf_mae = f"{first_scenario['cbf_mae']:.2f}" if first_scenario['cbf_mae'] is not None else "N/A"
            att_mae = f"{first_scenario['att_mae']:.2f}" if first_scenario['att_mae'] is not None else "N/A"

        print(f"{exp_name:<40} {status:<15} {cbf_mae:<12} {att_mae:<12}")

    # Save summary to file
    summary_file = Path("validation_results/VALIDATION_SUMMARY.json")
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Summary saved to {summary_file}")

    # Print final status
    print(f"\n{'='*70}")
    print(f"TOTAL: {summary['successful']} successful, {summary['failed']} failed")
    print(f"{'='*70}\n")

    return 0 if summary['failed'] == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
