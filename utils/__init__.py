"""Utility functions and feature registry."""

from utils.helpers import (
    process_signals_dynamic,
    ParallelStreamingStatsCalculator,
    process_signals_cpu,
    get_grid_search_initial_guess,
    get_multi_start_initial_guesses,
    fit_multi_start_ls,
)
from utils.feature_registry import FeatureRegistry, FeatureConfigError, validate_signals, validate_norm_stats
