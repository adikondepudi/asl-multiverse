# enhanced_simulation.py
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from scipy.ndimage import gaussian_filter
from asl_simulation import ASLSimulator, ASLParameters
from noise_engine import SpatialNoiseEngine
import multiprocessing as mp
from tqdm import tqdm
import logging
import time

logger = logging.getLogger(__name__)


class SpatialPhantomGenerator:
    """
    Generates realistic 2D spatial phantoms for ASL training.

    Creates tissue segmentation with gray matter, white matter, CSF,
    and pathological regions (tumor/stroke) with proper partial volume effects.

    Domain Randomization:
    --------------------
    Supports per-sample variation of physics parameters to prevent overfitting:
    - Blood T1: Varies with hematocrit (1550-2150 ms)
    - Labeling efficiencies: α_PCASL (0.75-0.95), α_VSASL (0.40-0.70)
    - Label duration (τ): ±10% variation
    - M0 scaling: Receiver gain variations

    This is critical for generalization to real patient data where these
    parameters vary significantly between subjects and scanners.
    """

    # Tissue CBF values (ml/100g/min)
    TISSUE_CBF = {
        'gray_matter': (50.0, 70.0),
        'white_matter': (18.0, 28.0),
        'csf': (0.0, 5.0),
        'tumor_hyper': (90.0, 150.0),  # Hypervascular tumor
        'tumor_hypo': (5.0, 20.0),     # Hypoperfused tumor core
        'stroke_core': (2.0, 10.0),    # Ischemic core
        'stroke_penumbra': (15.0, 35.0),  # Penumbra (salvageable)
    }

    # Tissue ATT values (ms)
    # NOTE: ATT values constrained to max PLD (3000ms) for detectability
    TISSUE_ATT = {
        'gray_matter': (1000.0, 1600.0),
        'white_matter': (1200.0, 1800.0),
        'csf': (100.0, 500.0),
        'tumor_hyper': (500.0, 1000.0),   # Fast transit (neovascularization)
        'tumor_hypo': (1800.0, 2500.0),   # Slow transit
        'stroke_core': (2500.0, 3000.0),  # Very delayed (constrained to max PLD)
        'stroke_penumbra': (1800.0, 2500.0),
    }

    # Domain randomization default ranges
    DEFAULT_DOMAIN_RAND = {
        'T1_artery_range': (1550.0, 2150.0),    # Hematocrit variations
        'alpha_PCASL_range': (0.75, 0.95),      # Labeling efficiency
        'alpha_VSASL_range': (0.40, 0.70),      # VSS efficiency
        'T_tau_perturb': 0.10,                  # ±10% label duration
        'M0_scale_range': (0.9, 1.1),           # Receiver gain variations
    }

    def __init__(self, size: int = 64, pve_sigma: float = 1.0,
                 domain_randomization: dict = None):
        """
        Args:
            size: Image size (size x size)
            pve_sigma: Gaussian blur sigma for partial volume effect
            domain_randomization: Dict with physics parameter ranges for training
        """
        self.size = size
        self.pve_sigma = pve_sigma

        # Domain randomization config
        self.domain_rand = domain_randomization or {}
        self.use_domain_rand = self.domain_rand.get('enabled', True)

        # Physics parameter ranges (with defaults)
        if self.use_domain_rand:
            self.T1_range = self.domain_rand.get('T1_artery_range', self.DEFAULT_DOMAIN_RAND['T1_artery_range'])
            self.alpha_PCASL_range = self.domain_rand.get('alpha_PCASL_range', self.DEFAULT_DOMAIN_RAND['alpha_PCASL_range'])
            self.alpha_VSASL_range = self.domain_rand.get('alpha_VSASL_range', self.DEFAULT_DOMAIN_RAND['alpha_VSASL_range'])
            self.T_tau_perturb = self.domain_rand.get('T_tau_perturb', self.DEFAULT_DOMAIN_RAND['T_tau_perturb'])
            self.M0_scale_range = self.domain_rand.get('M0_scale_range', self.DEFAULT_DOMAIN_RAND['M0_scale_range'])

    def sample_physics_params(self, base_params: 'ASLParameters' = None) -> Dict:
        """
        Sample physics parameters with domain randomization.

        This prevents the network from overfitting to fixed parameter values,
        which is critical for generalization to real patient data.

        Args:
            base_params: Base ASLParameters to use as defaults

        Returns:
            Dict with sampled physics parameters:
            - T1_artery: Blood T1 in ms
            - T_tau: Label duration in ms
            - alpha_PCASL: PCASL labeling efficiency
            - alpha_VSASL: VSASL labeling efficiency
            - M0_scale: M0 scaling factor
        """
        if base_params is None:
            base_params = ASLParameters()

        if self.use_domain_rand:
            # Sample from ranges
            T1_artery = np.random.uniform(*self.T1_range)
            alpha_PCASL = np.random.uniform(*self.alpha_PCASL_range)
            alpha_VSASL = np.random.uniform(*self.alpha_VSASL_range)
            T_tau = base_params.T_tau * (1 + np.random.uniform(-self.T_tau_perturb, self.T_tau_perturb))
            M0_scale = np.random.uniform(*self.M0_scale_range)
        else:
            # Use defaults
            T1_artery = base_params.T1_artery
            alpha_PCASL = base_params.alpha_PCASL
            alpha_VSASL = base_params.alpha_VSASL
            T_tau = base_params.T_tau
            M0_scale = 1.0

        return {
            'T1_artery': T1_artery,
            'T_tau': T_tau,
            'alpha_PCASL': alpha_PCASL,
            'alpha_VSASL': alpha_VSASL,
            'M0_scale': M0_scale,
        }
        
    def _generate_blob_mask(self, center: Tuple[int, int], 
                            radius: int, 
                            irregularity: float = 0.3) -> np.ndarray:
        """Generate an irregular blob mask (roughly circular with noise)."""
        y, x = np.ogrid[:self.size, :self.size]
        cy, cx = center
        
        # Base circular distance
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        # Add angular irregularity for more realistic shapes
        theta = np.arctan2(y - cy, x - cx)
        noise = irregularity * radius * np.sin(3*theta + np.random.rand()*np.pi)
        noise += irregularity * radius * 0.5 * np.sin(5*theta + np.random.rand()*np.pi)
        
        effective_radius = radius + noise
        mask = dist <= effective_radius
        
        return mask
    
    def _generate_voronoi_tissue(self) -> np.ndarray:
        """
        Generate tissue segmentation using simplified Voronoi-like regions.
        
        Returns:
            tissue_map: (H, W) with values 0=background, 1=GM, 2=WM, 3=CSF
        """
        tissue_map = np.zeros((self.size, self.size), dtype=np.int32)
        
        # Generate random seed points for regions
        n_seeds = np.random.randint(15, 30)
        seeds = np.random.randint(5, self.size-5, size=(n_seeds, 2))
        
        # Assign random tissue types (weighted toward GM and WM)
        tissue_types = np.random.choice(
            [1, 2, 3],  # GM, WM, CSF
            size=n_seeds,
            p=[0.5, 0.4, 0.1]
        )
        
        # Simple Voronoi: each pixel assigned to nearest seed
        y, x = np.ogrid[:self.size, :self.size]
        for i, (seed, tissue) in enumerate(zip(seeds, tissue_types)):
            dist = np.sqrt((x - seed[1])**2 + (y - seed[0])**2)
            if i == 0:
                min_dist = dist
                tissue_map = np.full((self.size, self.size), tissue)
            else:
                closer = dist < min_dist
                tissue_map[closer] = tissue
                min_dist = np.minimum(min_dist, dist)
        
        return tissue_map
    
    def generate_phantom(self, 
                        include_pathology: bool = True,
                        pathology_type: str = None) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a complete spatial phantom with CBF and ATT maps.
        
        Args:
            include_pathology: Whether to add pathological regions
            pathology_type: 'tumor', 'stroke', or None (random)
            
        Returns:
            cbf_map: (H, W) CBF values in ml/100g/min
            att_map: (H, W) ATT values in ms
            metadata: Dict with tissue labels and pathology info
        """
        # Generate tissue segmentation
        tissue_map = self._generate_voronoi_tissue()
        
        # Initialize parameter maps
        cbf_map = np.zeros((self.size, self.size), dtype=np.float32)
        att_map = np.zeros((self.size, self.size), dtype=np.float32)
        
        # Fill based on tissue type
        tissue_names = ['background', 'gray_matter', 'white_matter', 'csf']
        for tissue_id, tissue_name in enumerate(tissue_names):
            if tissue_id == 0:
                continue
            mask = tissue_map == tissue_id
            cbf_range = self.TISSUE_CBF[tissue_name]
            att_range = self.TISSUE_ATT[tissue_name]
            
            # Add some spatial variation within tissue type
            cbf_map[mask] = np.random.uniform(*cbf_range) + np.random.randn(np.sum(mask)) * 3
            att_map[mask] = np.random.uniform(*att_range) + np.random.randn(np.sum(mask)) * 50
        
        metadata = {'tissue_map': tissue_map, 'pathologies': []}
        
        # Add pathological regions
        if include_pathology:
            if pathology_type is None:
                pathology_type = np.random.choice(['tumor', 'stroke', 'none'], p=[0.35, 0.35, 0.3])
            
            if pathology_type != 'none':
                n_lesions = np.random.randint(1, 3)
                
                for _ in range(n_lesions):
                    # Random lesion location (avoid edges)
                    cx = np.random.randint(15, self.size - 15)
                    cy = np.random.randint(15, self.size - 15)
                    radius = np.random.randint(5, 15)
                    
                    lesion_mask = self._generate_blob_mask((cy, cx), radius)
                    
                    if pathology_type == 'tumor':
                        # Hypervascular rim with hypoperfused core
                        core_mask = self._generate_blob_mask((cy, cx), max(3, radius - 4))
                        rim_mask = lesion_mask & ~core_mask
                        
                        cbf_map[rim_mask] = np.random.uniform(*self.TISSUE_CBF['tumor_hyper'])
                        att_map[rim_mask] = np.random.uniform(*self.TISSUE_ATT['tumor_hyper'])
                        cbf_map[core_mask] = np.random.uniform(*self.TISSUE_CBF['tumor_hypo'])
                        att_map[core_mask] = np.random.uniform(*self.TISSUE_ATT['tumor_hypo'])
                        
                    elif pathology_type == 'stroke':
                        # Ischemic core with penumbra
                        core_mask = self._generate_blob_mask((cy, cx), max(3, radius - 5))
                        penumbra_mask = lesion_mask & ~core_mask
                        
                        cbf_map[core_mask] = np.random.uniform(*self.TISSUE_CBF['stroke_core'])
                        att_map[core_mask] = np.random.uniform(*self.TISSUE_ATT['stroke_core'])
                        cbf_map[penumbra_mask] = np.random.uniform(*self.TISSUE_CBF['stroke_penumbra'])
                        att_map[penumbra_mask] = np.random.uniform(*self.TISSUE_ATT['stroke_penumbra'])
                    
                    metadata['pathologies'].append({
                        'type': pathology_type,
                        'center': (cy, cx),
                        'radius': radius
                    })
        
        # Apply Partial Volume Effect (Gaussian blur)
        # This is CRUCIAL: creates soft edges that break simple LS fitting
        cbf_map = gaussian_filter(cbf_map, sigma=self.pve_sigma)
        att_map = gaussian_filter(att_map, sigma=self.pve_sigma)
        
        # Clamp to valid ranges
        cbf_map = np.clip(cbf_map, 0, 200).astype(np.float32)
        att_map = np.clip(att_map, 100, 5000).astype(np.float32)

        return cbf_map, att_map, metadata

    def generate_hard_phantom(self, difficulty: str = 'hard') -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate a more challenging spatial phantom for validation that differentiates
        model configurations better than standard phantoms.

        Args:
            difficulty: 'hard' or 'extreme'
                - 'hard': More pathology, wider ATT range, more PVE, watershed zones,
                  venous contamination
                - 'extreme': All of 'hard' plus negative CBF artifacts, highly
                  heterogeneous tissue with sharp boundaries, small embedded lesions

        Returns:
            cbf_map: (H, W) CBF values in ml/100g/min
            att_map: (H, W) ATT values in ms
            metadata: Dict with tissue labels, pathology info, and difficulty level
        """
        if difficulty not in ('hard', 'extreme'):
            raise ValueError(f"difficulty must be 'hard' or 'extreme', got '{difficulty}'")

        # --- Extended parameter ranges for hard phantoms ---
        hard_tissue_cbf = {
            'gray_matter': (50.0, 70.0),
            'white_matter': (18.0, 28.0),
            'csf': (0.0, 5.0),
        }
        hard_tissue_att = {
            'gray_matter': (500.0, 2200.0),      # Wider than standard (was 1000-1600)
            'white_matter': (800.0, 2800.0),      # Wider than standard (was 1200-1800)
            'csf': (100.0, 500.0),
        }

        if difficulty == 'extreme':
            hard_tissue_att['gray_matter'] = (400.0, 2800.0)
            hard_tissue_att['white_matter'] = (600.0, 3200.0)

        # Generate tissue segmentation
        tissue_map = self._generate_voronoi_tissue()

        # Initialize parameter maps
        cbf_map = np.zeros((self.size, self.size), dtype=np.float32)
        att_map = np.zeros((self.size, self.size), dtype=np.float32)

        # Fill based on tissue type with higher intra-tissue variability
        tissue_names = ['background', 'gray_matter', 'white_matter', 'csf']
        for tissue_id, tissue_name in enumerate(tissue_names):
            if tissue_id == 0:
                continue
            mask = tissue_map == tissue_id
            cbf_range = hard_tissue_cbf[tissue_name]
            att_range = hard_tissue_att[tissue_name]

            # Higher spatial variation within tissue (sigma 5 vs 3 in standard)
            intra_sigma = 5.0 if difficulty == 'hard' else 8.0
            cbf_map[mask] = np.random.uniform(*cbf_range) + np.random.randn(np.sum(mask)) * intra_sigma
            att_map[mask] = np.random.uniform(*att_range) + np.random.randn(np.sum(mask)) * 100

        metadata = {'tissue_map': tissue_map, 'pathologies': [], 'difficulty': difficulty}

        # --- Pathology: More frequent and more extreme ---
        # 30% pathology rate in hard mode (vs ~10% implicit in standard)
        pathology_prob = 0.70 if difficulty == 'hard' else 0.85
        if np.random.rand() < pathology_prob:
            pathology_type = np.random.choice(['tumor', 'stroke'])
            n_lesions = np.random.randint(2, 5)  # More lesions than standard (was 1-2)

            for _ in range(n_lesions):
                cx = np.random.randint(12, self.size - 12)
                cy = np.random.randint(12, self.size - 12)
                radius = np.random.randint(5, 18)  # Larger range (was 5-15)

                lesion_mask = self._generate_blob_mask((cy, cx), radius, irregularity=0.4)

                if pathology_type == 'tumor':
                    core_mask = self._generate_blob_mask((cy, cx), max(3, radius - 4))
                    rim_mask = lesion_mask & ~core_mask

                    # Higher CBF for hypervascular tumors (up to 150)
                    cbf_map[rim_mask] = np.random.uniform(90.0, 150.0)
                    att_map[rim_mask] = np.random.uniform(400.0, 1000.0)
                    cbf_map[core_mask] = np.random.uniform(3.0, 15.0)
                    att_map[core_mask] = np.random.uniform(2000.0, 3500.0)

                elif pathology_type == 'stroke':
                    core_mask = self._generate_blob_mask((cy, cx), max(3, radius - 5))
                    penumbra_mask = lesion_mask & ~core_mask

                    cbf_map[core_mask] = np.random.uniform(1.0, 8.0)
                    att_map[core_mask] = np.random.uniform(2800.0, 3500.0)
                    cbf_map[penumbra_mask] = np.random.uniform(10.0, 30.0)
                    att_map[penumbra_mask] = np.random.uniform(2000.0, 3000.0)

                metadata['pathologies'].append({
                    'type': pathology_type,
                    'center': (cy, cx),
                    'radius': radius
                })

        # --- Watershed zones: thin strips of very low CBF between tissue types ---
        # Find boundaries between tissue types
        from scipy.ndimage import sobel
        edges_y = sobel(tissue_map.astype(np.float32), axis=0)
        edges_x = sobel(tissue_map.astype(np.float32), axis=1)
        edge_magnitude = np.sqrt(edges_y**2 + edges_x**2)
        # Dilate edges slightly to create thin strips
        watershed_mask = gaussian_filter(edge_magnitude, sigma=0.8) > 0.3
        # Apply watershed zones
        cbf_map[watershed_mask] = np.random.uniform(5.0, 10.0)
        att_map[watershed_mask] = np.random.uniform(2200.0, 3000.0)
        metadata['has_watershed_zones'] = True

        # --- Venous contamination: scattered high-CBF spots with very high ATT ---
        n_venous_spots = np.random.randint(3, 8)
        for _ in range(n_venous_spots):
            vx = np.random.randint(5, self.size - 5)
            vy = np.random.randint(5, self.size - 5)
            vr = np.random.randint(2, 5)
            venous_mask = self._generate_blob_mask((vy, vx), vr, irregularity=0.5)
            cbf_map[venous_mask] = np.random.uniform(100.0, 180.0)
            att_map[venous_mask] = np.random.uniform(3000.0, 3500.0)
        metadata['n_venous_spots'] = n_venous_spots

        # --- Extreme-only features ---
        if difficulty == 'extreme':
            # Negative CBF artifacts (motion-corrupted regions)
            n_motion_artifacts = np.random.randint(1, 4)
            for _ in range(n_motion_artifacts):
                mx = np.random.randint(10, self.size - 10)
                my = np.random.randint(10, self.size - 10)
                mr = np.random.randint(3, 8)
                motion_mask = self._generate_blob_mask((my, mx), mr, irregularity=0.6)
                cbf_map[motion_mask] = np.random.uniform(-5.0, 0.0)
                att_map[motion_mask] = np.random.uniform(500.0, 2000.0)
            metadata['n_motion_artifacts'] = n_motion_artifacts

            # Small embedded lesions (3-5 pixel radius) in normal tissue
            n_small_lesions = np.random.randint(5, 12)
            for _ in range(n_small_lesions):
                lx = np.random.randint(8, self.size - 8)
                ly = np.random.randint(8, self.size - 8)
                lr = np.random.randint(3, 5)
                small_mask = self._generate_blob_mask((ly, lx), lr, irregularity=0.2)
                # Random lesion type
                if np.random.rand() < 0.5:
                    cbf_map[small_mask] = np.random.uniform(80.0, 130.0)
                    att_map[small_mask] = np.random.uniform(500.0, 900.0)
                else:
                    cbf_map[small_mask] = np.random.uniform(2.0, 10.0)
                    att_map[small_mask] = np.random.uniform(2500.0, 4000.0)
            metadata['n_small_lesions'] = n_small_lesions

            # Sharp boundaries: reduce PVE sigma for some regions to create
            # highly heterogeneous tissue with abrupt transitions
            # We'll apply PVE with reduced sigma and mix with sharp version
            sharp_region_mask = np.random.rand(self.size, self.size) > 0.5
            cbf_sharp = cbf_map.copy()
            att_sharp = att_map.copy()

        # Apply Partial Volume Effect (increased sigma for hard phantoms)
        pve_sigma = self.pve_sigma * 1.5 if difficulty == 'hard' else self.pve_sigma * 0.5
        cbf_map = gaussian_filter(cbf_map, sigma=pve_sigma)
        att_map = gaussian_filter(att_map, sigma=pve_sigma)

        if difficulty == 'extreme':
            # Mix sharp and blurred regions for heterogeneous boundaries
            cbf_map = np.where(sharp_region_mask, cbf_sharp, cbf_map)
            att_map = np.where(sharp_region_mask, att_sharp, att_map)
            # Light smoothing to avoid fully pixelated boundaries
            cbf_map = gaussian_filter(cbf_map, sigma=0.3)
            att_map = gaussian_filter(att_map, sigma=0.3)

        # Clamp to valid ranges (allow negative CBF in extreme mode)
        if difficulty == 'extreme':
            cbf_map = np.clip(cbf_map, -10, 200).astype(np.float32)
            att_map = np.clip(att_map, 100, 5000).astype(np.float32)
        else:
            cbf_map = np.clip(cbf_map, 0, 200).astype(np.float32)
            att_map = np.clip(att_map, 100, 5000).astype(np.float32)

        return cbf_map, att_map, metadata

    def generate_validation_suite(self, n_per_difficulty: int = 20) -> Dict[str, List[Tuple[np.ndarray, np.ndarray, Dict]]]:
        """
        Generate a balanced validation suite with easy/standard/hard/extreme phantoms.

        This produces a set of phantoms across a range of difficulty levels for
        comprehensive model evaluation. Harder phantoms stress-test the model with
        wider parameter ranges, more pathology, and challenging artifacts.

        Args:
            n_per_difficulty: Number of phantoms per difficulty level (default 20)

        Returns:
            Dict mapping difficulty level to list of (cbf_map, att_map, metadata) tuples.
            Keys: 'easy', 'standard', 'hard', 'extreme'
        """
        suite = {
            'easy': [],
            'standard': [],
            'hard': [],
            'extreme': [],
        }

        logger.info(f"Generating validation suite: {n_per_difficulty} phantoms per difficulty...")

        # Easy: no pathology, standard tissue ranges
        for i in range(n_per_difficulty):
            np.random.seed(1000 + i)  # Reproducible
            cbf, att, meta = self.generate_phantom(include_pathology=False)
            meta['difficulty'] = 'easy'
            suite['easy'].append((cbf, att, meta))

        # Standard: pathology possible, standard ranges (existing behavior)
        for i in range(n_per_difficulty):
            np.random.seed(2000 + i)
            cbf, att, meta = self.generate_phantom(include_pathology=True)
            meta['difficulty'] = 'standard'
            suite['standard'].append((cbf, att, meta))

        # Hard
        for i in range(n_per_difficulty):
            np.random.seed(3000 + i)
            cbf, att, meta = self.generate_hard_phantom(difficulty='hard')
            suite['hard'].append((cbf, att, meta))

        # Extreme
        for i in range(n_per_difficulty):
            np.random.seed(4000 + i)
            cbf, att, meta = self.generate_hard_phantom(difficulty='extreme')
            suite['extreme'].append((cbf, att, meta))

        total = sum(len(v) for v in suite.values())
        logger.info(f"Validation suite generated: {total} total phantoms "
                    f"({n_per_difficulty} per difficulty x 4 levels)")

        return suite

@dataclass
class PhysiologicalVariation:
    # NOTE: ATT ranges are constrained to max PLD (3000ms) to ensure signals are measurable.
    # ATT values > max PLD result in zero/minimal signal, making estimation ill-posed.
    cbf_range: Tuple[float, float] = (20.0, 100.0)
    att_range: Tuple[float, float] = (500.0, 3000.0)  # Constrained to match PLD range
    t1_artery_range: Tuple[float, float] = (1650.0, 2050.0)
    stroke_cbf_range: Tuple[float, float] = (5.0, 30.0)
    stroke_att_range: Tuple[float, float] = (1500.0, 3000.0)  # Constrained
    tumor_cbf_range: Tuple[float, float] = (10.0, 150.0)
    tumor_att_range: Tuple[float, float] = (700.0, 3000.0)  # Constrained
    young_cbf_range: Tuple[float, float] = (60.0, 120.0)
    young_att_range: Tuple[float, float] = (500.0, 1500.0)
    elderly_cbf_range: Tuple[float, float] = (30.0, 70.0)
    elderly_att_range: Tuple[float, float] = (1500.0, 3000.0)  # Constrained
    t_tau_perturb_range: Tuple[float, float] = (-0.05, 0.05)
    alpha_perturb_range: Tuple[float, float] = (-0.10, 0.10)
    arterial_blood_volume_range: Tuple[float, float] = (0.00, 0.015) # 0 to 1.5%

class RealisticASLSimulator(ASLSimulator):
    def __init__(self, params: ASLParameters = ASLParameters()):
        super().__init__(params)
        self.physio_var = PhysiologicalVariation()

    def add_modular_noise(self, signal, snr, noise_components=['thermal']):
        """
        Applies noise components based on a list string.
        options: 'thermal', 'physio', 'drift', 'spikes'
        """
        noisy_signal = signal.copy()
        sig_len = signal.shape[-1]
        t = np.arange(sig_len)
        
        # 1. Base Thermal Level
        mean_sig = np.mean(np.abs(signal))
        noise_sd = (mean_sig / snr) if snr > 0 else 0
        
        # 2. Additive Layers
        if 'physio' in noise_components:
            # Cardiac (Fast) + Respiratory (Slow)
            cardiac = (noise_sd * 0.5) * np.sin(2 * np.pi * 1.0 * t / sig_len * 5 + np.random.rand())
            resp = (noise_sd * 0.3) * np.sin(2 * np.pi * 0.3 * t / sig_len * 5 + np.random.rand())
            noisy_signal += (cardiac + resp)

        if 'drift' in noise_components:
            # Low freq baseline shift
            drift = (noise_sd * 0.4) * np.linspace(-1, 1, sig_len)
            noisy_signal += drift

        if 'spikes' in noise_components:
            # Random outliers
            if np.random.rand() < 0.2: # 20% of samples get a spike
                idx = np.random.randint(0, sig_len)
                noisy_signal[idx] += 5 * noise_sd * np.random.choice([-1, 1])

        # 3. Final Rician/Thermal Noise (Correct MRI Physics)
        if 'thermal' in noise_components:
            n_real = np.random.normal(0, noise_sd, signal.shape)
            n_imag = np.random.normal(0, noise_sd, signal.shape)
            noisy_signal = np.sqrt((noisy_signal + n_real)**2 + n_imag**2)
            
        return noisy_signal

    def add_realistic_noise(self, signal: np.ndarray, snr: float = 5.0,
                            temporal_correlation: float = 0.3, include_spike_artifacts: bool = True,
                            spike_probability: float = 0.01, spike_magnitude_factor: float = 5.0,
                            include_baseline_drift: bool = True, drift_magnitude_factor: float = 0.1,
                            include_physiological: bool = True) -> np.ndarray:
        """
        Applies a comprehensive, physically layered noise model to a clean ASL signal.
        This models the sequence of real-world signal corruption:
        1. Physiological fluctuations and drift are added to the clean signal.
        2. Rician noise, the correct model for MR magnitude data, is applied.
        3. Sporadic spike artifacts corrupt the final noisy signal.
        """
        signal_with_phys = signal.copy()
        
        mean_abs_signal = np.mean(np.abs(signal))
        base_noise_level = mean_abs_signal / snr if snr > 0 and mean_abs_signal > 0 else 1e-5

        if include_physiological and signal.ndim > 0 and signal.shape[-1] > 1:
            t = np.arange(signal.shape[-1])
            cardiac_freq = np.random.uniform(0.8, 1.2)
            cardiac = (base_noise_level * 0.5) * np.sin(2 * np.pi * cardiac_freq * t / signal.shape[-1] * 5 + np.random.rand() * np.pi)
            respiratory_freq = np.random.uniform(0.2, 0.4)
            respiratory = (base_noise_level * 0.3) * np.sin(2 * np.pi * respiratory_freq * t / signal.shape[-1] * 5 + np.random.rand() * np.pi)
            signal_with_phys += cardiac + respiratory

        if include_baseline_drift and signal.ndim > 0 and signal.shape[-1] > 1:
            drift_freq = np.random.uniform(0.05, 0.2)
            drift_amp = drift_magnitude_factor * (mean_abs_signal if mean_abs_signal > 0 else base_noise_level)
            drift = drift_amp * np.sin(2 * np.pi * drift_freq * t / signal.shape[-1] + np.random.rand() * np.pi)
            signal_with_phys += drift

        sigma_rician = base_noise_level / np.sqrt(2)
        noise_real = np.random.normal(0, sigma_rician, signal.shape)
        noise_imag = np.random.normal(0, sigma_rician, signal.shape)
        noisy_signal = np.sqrt((signal_with_phys + noise_real)**2 + noise_imag**2)

        if include_spike_artifacts and signal.ndim > 0:
            num_spikes = np.random.poisson(signal.shape[-1] * spike_probability)
            spike_indices = np.random.choice(signal.shape[-1], num_spikes, replace=False)
            for i in spike_indices:
                spike = (np.random.choice([-1, 1])) * spike_magnitude_factor * base_noise_level
                noisy_signal[i] += spike
        
        return noisy_signal

    def generate_spatial_batch(self, plds: np.ndarray, batch_size: int = 4, size: int = 64,
                               domain_randomization: dict = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generates a batch of 2D ASL training images with blobs and artifacts.

        Supports domain randomization: physics parameters (T1, alpha, tau) are
        sampled per-batch to prevent overfitting to fixed values.

        Args:
            plds: Post-labeling delays in ms
            batch_size: Number of images per batch
            size: Image dimensions (size x size)
            domain_randomization: Dict with physics parameter ranges

        Returns:
            signals: (Batch, 2*n_plds, Size, Size)
            targets: (Batch, 2, Size, Size) -> [CBF, ATT]
        """
        # Prepare outputs
        n_plds = len(plds)
        signals = np.zeros((batch_size, n_plds * 2, size, size), dtype=np.float32)
        targets = np.zeros((batch_size, 2, size, size), dtype=np.float32)

        # Shape: (n_plds, 1, 1)
        plds_bc = plds[:, np.newaxis, np.newaxis]
        lambda_b = 0.90
        t2_f = self.params.T2_factor

        # Domain randomization config
        dr = domain_randomization or {}
        use_dr = dr.get('enabled', True)
        T1_range = dr.get('T1_artery_range', (1550.0, 2150.0))
        alpha_PCASL_range = dr.get('alpha_PCASL_range', (0.75, 0.95))
        alpha_VSASL_range = dr.get('alpha_VSASL_range', (0.40, 0.70))
        T_tau_perturb = dr.get('T_tau_perturb', 0.10)
        M0_scale_range = dr.get('M0_scale_range', (0.9, 1.1))
        # Background suppression: 1.0 = no BS, 0.85-0.95 = typical in-vivo BS
        # PCASL uses alpha_BS1^4 (4 BS pulses), VSASL uses alpha_BS1^3 (3 BS pulses)
        alpha_BS1_range = dr.get('alpha_BS1_range', (0.85, 1.0))

        for b in range(batch_size):
            # --- Domain Randomization: Sample physics params per batch ---
            if use_dr:
                t1_b = np.random.uniform(*T1_range)
                tau = self.params.T_tau * (1 + np.random.uniform(-T_tau_perturb, T_tau_perturb))
                alpha_bs1 = np.random.uniform(*alpha_BS1_range)
                alpha_p = np.random.uniform(*alpha_PCASL_range) * (alpha_bs1**4)
                alpha_v = np.random.uniform(*alpha_VSASL_range) * (alpha_bs1**3)
                m0_scale = np.random.uniform(*M0_scale_range)
            else:
                t1_b = self.params.T1_artery
                tau = self.params.T_tau
                alpha_p = self.params.alpha_PCASL * (self.params.alpha_BS1**4)
                alpha_v = self.params.alpha_VSASL * (self.params.alpha_BS1**3)
                m0_scale = 1.0
            # --- 1. Generate Parameter Maps (The "Phantom") ---
            # Background (Gray Matter-ish)
            cbf_map = np.random.normal(50, 5, (size, size))
            att_map = np.random.normal(1200, 100, (size, size))
            
            # Add Pathology Blobs
            num_blobs = np.random.randint(1, 4)
            for _ in range(num_blobs):
                cx, cy = np.random.randint(10, size-10, 2)
                r = np.random.randint(5, 15)
                
                y_grid, x_grid = np.ogrid[:size, :size]
                mask = ((x_grid - cx)**2 + (y_grid - cy)**2) <= r**2
                
                # Flip coin: Tumor (High Flow) or Stroke (Low Flow)
                if np.random.rand() > 0.5:
                    cbf_map[mask] = np.random.uniform(90, 140)  # Hyperperfusion
                    att_map[mask] = np.random.uniform(500, 1000)  # Fast transit
                else:
                    cbf_map[mask] = np.random.uniform(5, 15)    # Hypoperfusion
                    att_map[mask] = np.random.uniform(2000, 3000)  # Delayed arrival

            # Smooth edges to simulate Partial Volume Effect
            cbf_map = gaussian_filter(cbf_map, sigma=1.0)
            att_map = gaussian_filter(att_map, sigma=1.0)
            
            # Save targets
            targets[b, 0] = cbf_map
            targets[b, 1] = att_map

            # --- 2. Vectorized Signal Generation (NumPy Broadcasting) ---
            # Input maps: (Size, Size) -> Broadcast to (n_plds, Size, Size)

            att_bc = att_map[np.newaxis, :, :]
            # CRITICAL: Convert CBF from ml/100g/min to ml/g/s (divide by 6000)
            # This matches asl_simulation.py physics equations
            cbf_bc = (cbf_map / 6000.0)[np.newaxis, :, :]
            
            # --- PCASL Logic ---
            # Mask 1: Bolus Arrived (PLD >= ATT)
            mask_arrived = (plds_bc >= att_bc)
            
            # Mask 2: Bolus in Transit (ATT - tau <= PLD < ATT)
            mask_transit = (plds_bc < att_bc) & (plds_bc >= (att_bc - tau))
            
            # Equation 1: Arrived
            sig_p_arrived = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                             np.exp(-plds_bc / t1_b) *
                             (1 - np.exp(-tau / t1_b)) * t2_f) / lambda_b
                             
            # Equation 2: Transit
            sig_p_transit = (2 * alpha_p * cbf_bc * t1_b / 1000.0 *
                             (np.exp(-att_bc / t1_b) - np.exp(-(tau + plds_bc) / t1_b)) *
                             t2_f) / lambda_b
            
            pcasl_sig = np.zeros_like(plds_bc * cbf_bc)
            pcasl_sig[mask_arrived] = sig_p_arrived[mask_arrived]
            pcasl_sig[mask_transit] = sig_p_transit[mask_transit]

            # --- VSASL Logic ---
            mask_vs_arrived = (plds_bc > att_bc)
            
            # Equation 1: PLD <= ATT
            sig_v_early = (2 * alpha_v * cbf_bc * (plds_bc / 1000.0) *
                           np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
                           
            # Equation 2: PLD > ATT
            sig_v_late = (2 * alpha_v * cbf_bc * (att_bc / 1000.0) *
                          np.exp(-plds_bc / t1_b) * t2_f) / lambda_b
            
            vsasl_sig = np.where(mask_vs_arrived, sig_v_late, sig_v_early)

            # --- 3. Stack and Add Noise ---
            # Current shape: (n_plds, Size, Size)
            # We need to stack into (2*n_plds, Size, Size)
            clean_stack = np.concatenate([pcasl_sig, vsasl_sig], axis=0)

            # Apply M0 scaling (domain randomization for receiver gain variations)
            clean_stack = clean_stack * m0_scale

            # Add Complex Rician Noise (Spatial)
            # Rician noise: S_noisy = sqrt((S + N_real)^2 + N_imag^2)
            # This correctly simulates the positive bias at low SNR seen in real MRI
            mean_sig = np.mean(clean_stack)
            # Random SNR per image
            snr = np.random.uniform(5.0, 15.0)
            sigma = mean_sig / snr

            noise_r = np.random.normal(0, sigma, clean_stack.shape)
            noise_i = np.random.normal(0, sigma, clean_stack.shape)
            noisy_stack = np.sqrt((clean_stack + noise_r)**2 + noise_i**2)
            
            signals[b] = noisy_stack

        return signals, targets

    def generate_diverse_dataset(self, plds: np.ndarray, n_subjects: int = 100,
                               conditions: List[str] = ['healthy', 'stroke', 'tumor', 'elderly'],
                               noise_levels: List[float] = [3.0, 5.0, 10.0]) -> Dict:
        """
        Generates a fixed-size dataset with maximal realism for validation or testing.
        This function now correctly uses the unified, layered noise model for every generated sample,
        ensuring the validation data is as realistic as the training data.
        """
        dataset = {'signals': [], 'parameters': [], 'conditions': [], 'noise_levels': [], 'perturbed_params': []}
        base_params = self.params

        condition_map = {
            'healthy': (self.physio_var.cbf_range, self.physio_var.att_range, self.physio_var.t1_artery_range),
            'stroke': (self.physio_var.stroke_cbf_range, self.physio_var.stroke_att_range, (self.physio_var.t1_artery_range[0]-100, self.physio_var.t1_artery_range[1]+100)),
            'tumor': (self.physio_var.tumor_cbf_range, self.physio_var.tumor_att_range, (self.physio_var.t1_artery_range[0]-150, self.physio_var.t1_artery_range[1]+150)),
            'elderly': (self.physio_var.elderly_cbf_range, self.physio_var.elderly_att_range, (self.physio_var.t1_artery_range[0]+50, self.physio_var.t1_artery_range[1]+150))
        }

        for _ in tqdm(range(n_subjects), desc="Generating Fixed Diverse Dataset"):
            condition = np.random.choice(conditions)
            cbf_range, att_range, t1_range = condition_map.get(condition, (self.physio_var.cbf_range, self.physio_var.att_range, self.physio_var.t1_artery_range))

            cbf = np.random.uniform(*cbf_range)
            att = np.random.uniform(*att_range)
            t1_a = np.random.uniform(*t1_range)
            abv = np.random.uniform(*self.physio_var.arterial_blood_volume_range) if np.random.rand() > 0.5 else 0.0
            slice_idx = np.random.randint(0, 20) # Simulating 20 slices
            slice_delay_factor = np.exp(-(slice_idx * 45.0)/1000.0)
            
            perturbed_t_tau = base_params.T_tau * (1 + np.random.uniform(*self.physio_var.t_tau_perturb_range))
            perturbed_alpha_pcasl = np.clip(base_params.alpha_PCASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range)), 0.1, 1.1)
            perturbed_alpha_vsasl = np.clip(base_params.alpha_VSASL * (1 + np.random.uniform(*self.physio_var.alpha_perturb_range)), 0.1, 1.0)
            
            # Apply slice timing to alphas
            eff_alpha_pcasl = perturbed_alpha_pcasl * slice_delay_factor
            eff_alpha_vsasl = perturbed_alpha_vsasl * slice_delay_factor

            vsasl_clean = self._generate_vsasl_signal(plds, att, cbf, t1_a, eff_alpha_vsasl)
            pcasl_clean = self._generate_pcasl_signal(plds, att, cbf, t1_a, perturbed_t_tau, eff_alpha_pcasl)
            art_sig = self._generate_arterial_signal(plds, att, abv, t1_a, eff_alpha_pcasl)
            
            pcasl_clean += art_sig # Add macrovascular component
            
            for snr in noise_levels:
                vsasl_noisy = self.add_realistic_noise(vsasl_clean, snr=snr)
                pcasl_noisy = self.add_realistic_noise(pcasl_clean, snr=snr)
                
                multiverse_signal_flat = np.concatenate([pcasl_noisy, vsasl_noisy])
                
                dataset['signals'].append(multiverse_signal_flat)
                dataset['parameters'].append([cbf, att, t1_a, float(slice_idx)]) # Store slice index
                dataset['conditions'].append(condition)
                dataset['noise_levels'].append(snr)
                dataset['perturbed_params'].append({
                    't1_artery': t1_a, 't_tau': perturbed_t_tau, 
                    'alpha_pcasl': perturbed_alpha_pcasl, 'alpha_vsasl': perturbed_alpha_vsasl
                })

        dataset['signals'] = np.array(dataset['signals'])
        dataset['parameters'] = np.array(dataset['parameters'])
        return dataset

if __name__ == "__main__":
    simulator = RealisticASLSimulator()
    logger.info("Enhanced ASL Simulator initialized. `generate_balanced_dataset` has been removed.")
    logger.info("For training, please use the `generate_offline_dataset.py` script.")
    logger.info("\nTesting `generate_diverse_dataset` for fixed validation/test sets...")
    test_data = simulator.generate_diverse_dataset(plds=np.arange(500, 3001, 500), n_subjects=10)
    if test_data['signals'].size > 0:
        logger.info(f"\nGenerated test dataset shape: {test_data['signals'].shape}")
    else: logger.info("No test data generated.")