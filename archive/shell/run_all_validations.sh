#!/bin/bash
# =============================================================================
# run_all_validations.sh
# =============================================================================
# Runs validate.py on ALL experiments in hpc_ablation_jobs folder.
# This generates the interactive_plot_data.json files needed for the dashboard.
#
# Usage:
#   chmod +x run_all_validations.sh
#   ./run_all_validations.sh
# =============================================================================

set -e  # Exit on any error

JOBS_DIR="hpc_ablation_jobs"

echo "=============================================="
echo "  ASL Multiverse - Batch Validation Runner"
echo "=============================================="
echo ""

# Check if directory exists
if [ ! -d "$JOBS_DIR" ]; then
    echo "ERROR: Directory '$JOBS_DIR' not found!"
    exit 1
fi

# Count experiments
NUM_EXPS=$(ls -d ${JOBS_DIR}/*/ 2>/dev/null | wc -l | tr -d ' ')
echo "Found $NUM_EXPS experiments to validate."
echo ""

# Counter
CURRENT=0

# Loop through all experiment directories
for EXP_DIR in ${JOBS_DIR}/*/; do
    CURRENT=$((CURRENT + 1))
    EXP_NAME=$(basename "$EXP_DIR")
    OUTPUT_DIR="${EXP_DIR}validation_results"
    
    echo "[$CURRENT/$NUM_EXPS] Processing: $EXP_NAME"
    echo "  Input:  $EXP_DIR"
    echo "  Output: $OUTPUT_DIR"
    
    # Run validation
    python3 validate.py \
        --run_dir "$EXP_DIR" \
        --output_dir "$OUTPUT_DIR"
    
    # Check if JSON was created
    if [ -f "${OUTPUT_DIR}/interactive_plot_data.json" ]; then
        echo "  ✅ Success: interactive_plot_data.json created"
    else
        echo "  ⚠️  Warning: JSON file not found"
    fi
    
    echo ""
done

echo "=============================================="
echo "  DONE! Validated $NUM_EXPS experiments."
echo "=============================================="
echo ""
echo "Next step: Launch the interactive dashboard:"
echo "  streamlit run asl_interactive_dashboard.py"
echo ""
