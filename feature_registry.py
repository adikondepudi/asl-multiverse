# FILE: feature_registry.py
"""
Centralized Feature Registry for ASL Pipeline.

This module is THE SINGLE SOURCE OF TRUTH for:
- Feature dimension calculations
- Config validation
- Default parameter values

All other modules MUST import from here instead of hardcoding values.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FeatureConfigError(Exception):
    """Raised when feature configuration is invalid."""
    pass


class FeatureRegistry:
    """
    Centralized registry for ASL feature configurations.
    
    Usage:
        scalar_dim = FeatureRegistry.compute_scalar_dim(['mean', 'std', 'peak'])
        FeatureRegistry.validate_active_features(['mean', 'std'])
    """
    
    # Canonical feature dimensions - NEVER change these without updating all code
    FEATURE_DIMS: Dict[str, int] = {
        'mean': 2,      # pcasl_mean, vsasl_mean
        'std': 2,       # pcasl_std, vsasl_std
        'ttp': 2,       # pcasl_ttp, vsasl_ttp (time to peak)
        'com': 2,       # pcasl_com, vsasl_com (center of mass)
        'peak': 2,      # pcasl_peak, vsasl_peak (peak height)
        't1_artery': 1, # Single T1 value
        'z_coord': 1,   # Single slice index
    }
    
    # Canonical feature indices in scalar_features_mean/std arrays
    # Layout: [mu_p(0), sig_p(1), mu_v(2), sig_v(3), ttp_p(4), ttp_v(5), com_p(6), com_v(7), peak_p(8), peak_v(9)]
    NORM_STATS_INDICES: Dict[str, List[int]] = {
        'mean': [0, 2],   # mu_p, mu_v
        'std': [1, 3],    # sig_p, sig_v
        'ttp': [4, 5],    # ttp_p, ttp_v
        'com': [6, 7],    # com_p, com_v
        'peak': [8, 9],   # peak_p, peak_v
    }
    
    # Default PLD values (in ms) - should match standard 6-PLD acquisition
    DEFAULT_PLDS: List[int] = [500, 1000, 1500, 2000, 2500, 3000]
    
    # Default physics parameters
    DEFAULT_PHYSICS: Dict[str, float] = {
        'T1_artery': 1850.0,
        'T_tau': 1800.0,
        'alpha_PCASL': 0.85,
        'alpha_VSASL': 0.56,
        'T2_factor': 1.0,
        'alpha_BS1': 1.0,
    }
    
    # Valid noise component names
    VALID_NOISE_COMPONENTS = {'thermal', 'physio', 'drift', 'spikes'}
    
    # Valid encoder types
    VALID_ENCODER_TYPES = {'physics_processor', 'mlp_only'}
    
    @classmethod
    def compute_scalar_dim(cls, active_features: List[str]) -> int:
        """
        Compute the total number of scalar features for a given feature list.
        
        Args:
            active_features: List of feature names, e.g. ['mean', 'std', 'peak']
            
        Returns:
            Total scalar dimension (int)
            
        Raises:
            FeatureConfigError: If unknown features are requested
        """
        cls.validate_active_features(active_features)
        return sum(cls.FEATURE_DIMS[f] for f in active_features)
    
    @classmethod
    def validate_active_features(cls, active_features: List[str]) -> None:
        """
        Validate that all requested features are known.
        
        Raises:
            FeatureConfigError: If unknown features are present
        """
        if not active_features:
            raise FeatureConfigError("active_features cannot be empty")
        
        unknown = set(active_features) - set(cls.FEATURE_DIMS.keys())
        if unknown:
            raise FeatureConfigError(
                f"Unknown features: {unknown}. "
                f"Valid features: {list(cls.FEATURE_DIMS.keys())}"
            )
    
    @classmethod
    def validate_noise_components(cls, components: List[str]) -> None:
        """
        Validate noise component names.
        
        Raises:
            FeatureConfigError: If unknown components are present
        """
        if not components:
            raise FeatureConfigError("data_noise_components cannot be empty")
        
        unknown = set(components) - cls.VALID_NOISE_COMPONENTS
        if unknown:
            raise FeatureConfigError(
                f"Unknown noise components: {unknown}. "
                f"Valid components: {cls.VALID_NOISE_COMPONENTS}"
            )
    
    @classmethod
    def validate_encoder_type(cls, encoder_type: str) -> None:
        """
        Validate encoder type string.
        
        Raises:
            FeatureConfigError: If unknown encoder type
        """
        if encoder_type.lower() not in cls.VALID_ENCODER_TYPES:
            raise FeatureConfigError(
                f"Unknown encoder_type: '{encoder_type}'. "
                f"Valid types: {cls.VALID_ENCODER_TYPES}"
            )
    
    @classmethod
    def validate_plds(cls, pld_values: List[int]) -> None:
        """
        Validate PLD values are reasonable.
        
        Raises:
            FeatureConfigError: If PLDs are invalid
        """
        if not pld_values:
            raise FeatureConfigError("pld_values cannot be empty")
        
        if len(pld_values) < 2:
            raise FeatureConfigError("Need at least 2 PLDs for meaningful ASL analysis")
        
        pld_array = np.array(pld_values)
        
        if not np.all(pld_array > 0):
            raise FeatureConfigError("All PLDs must be positive")
        
        if not np.all(np.diff(pld_array) > 0):
            logger.warning("PLDs are not strictly increasing - this may cause issues")
        
        if pld_array.min() < 100 or pld_array.max() > 5000:
            logger.warning(f"PLDs {pld_values} are outside typical range [100, 5000] ms")
    
    @classmethod
    def validate_config(cls, config: Dict[str, Any]) -> None:
        """
        Comprehensive config validation.
        
        Args:
            config: Full research config dictionary
            
        Raises:
            FeatureConfigError: If any validation fails
        """
        # Validate active features
        active_features = config.get('active_features')
        if active_features is None:
            raise FeatureConfigError("Config missing 'active_features'")
        cls.validate_active_features(active_features)
        
        # Validate noise components
        noise_components = config.get('data_noise_components')
        if noise_components is None:
            raise FeatureConfigError("Config missing 'data_noise_components'")
        cls.validate_noise_components(noise_components)
        
        # Validate PLDs
        pld_values = config.get('pld_values')
        if pld_values is None:
            raise FeatureConfigError("Config missing 'pld_values'")
        cls.validate_plds(pld_values)
        
        # Validate encoder type if present
        encoder_type = config.get('encoder_type')
        if encoder_type:
            cls.validate_encoder_type(encoder_type)
        
        # Validate numeric ranges
        n_ensembles = config.get('n_ensembles', 1)
        if n_ensembles < 1:
            raise FeatureConfigError(f"n_ensembles must be >= 1, got {n_ensembles}")
        
        batch_size = config.get('batch_size', 256)
        if batch_size < 1:
            raise FeatureConfigError(f"batch_size must be >= 1, got {batch_size}")
        
        logger.info(f"Config validated: {len(active_features)} features, "
                    f"{len(pld_values)} PLDs, {noise_components} noise")
    
    @classmethod
    def get_input_size(cls, n_plds: int, active_features: List[str]) -> int:
        """
        Calculate the full input size for the network.
        
        Args:
            n_plds: Number of PLDs
            active_features: List of active feature names
            
        Returns:
            Total input dimension (curves + scalars)
        """
        curve_dim = n_plds * 2  # PCASL + VSASL shape vectors
        scalar_dim = cls.compute_scalar_dim(active_features)
        return curve_dim + scalar_dim


def validate_signals(signals: np.ndarray, context: str = "unknown") -> None:
    """
    Universal signal validation with defensive checks.
    
    Args:
        signals: Array of ASL signals
        context: Description for error messages
        
    Raises:
        ValueError: If signals are invalid
    """
    if signals is None:
        raise ValueError(f"[{context}] Signals are None")
    
    if signals.ndim < 1:
        raise ValueError(f"[{context}] Signals must have at least 1 dimension")
    
    if not np.isfinite(signals).all():
        nan_count = np.isnan(signals).sum()
        inf_count = np.isinf(signals).sum()
        raise ValueError(
            f"[{context}] Signals contain {nan_count} NaN and {inf_count} Inf values"
        )
    
    if signals.ndim >= 2 and signals.shape[-1] % 2 != 0:
        logger.warning(
            f"[{context}] Signal last dimension {signals.shape[-1]} is odd - "
            "expected even (PCASL + VSASL)"
        )


def validate_norm_stats(norm_stats: Dict[str, Any], context: str = "unknown") -> None:
    """
    Validate normalization statistics dictionary.
    
    Args:
        norm_stats: Dictionary containing normalization statistics
        context: Description for error messages
        
    Raises:
        ValueError: If norm_stats is invalid
    """
    required_keys = ['y_mean_cbf', 'y_std_cbf', 'y_mean_att', 'y_std_att',
                     'scalar_features_mean', 'scalar_features_std']
    
    missing = set(required_keys) - set(norm_stats.keys())
    if missing:
        raise ValueError(f"[{context}] norm_stats missing keys: {missing}")
    
    # Check for zero std (would cause division by zero)
    for key in ['y_std_cbf', 'y_std_att']:
        if norm_stats[key] <= 0:
            raise ValueError(f"[{context}] norm_stats['{key}'] must be positive")
    
    scalar_std = np.array(norm_stats['scalar_features_std'])
    if np.any(scalar_std <= 0):
        logger.warning(f"[{context}] Some scalar_features_std values are <= 0")
