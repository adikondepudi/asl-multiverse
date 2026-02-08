# Proposed Changes to validate.py: Multi-Sigma Smoothed-LS & Tissue Stratification

**Status**: PROPOSED ONLY -- do NOT apply until coordinated with foundation-lead P1 fixes.

---

## 1. Current Smoothed-LS Implementation (How It Works)

### Location
`validate.py`, method `_run_spatial_at_snr()`, lines 788-820

### Exact Algorithm
```
For each phantom:
  1. Take noisy_signals (n_plds*2, H, W) -- the RAW noisy signals (not scaled/normalized)
  2. Apply gaussian_filter per channel with sigma=2.0:
     for ch in range(12):  # 6 PCASL + 6 VSASL channels
         smoothed_signals[ch] = gaussian_filter(noisy_signals[ch], sigma=2.0)
  3. For each subsampled voxel (1:10 ratio):
     - Extract smoothed voxel signal
     - Run grid_search + LS optimizer (same as raw LS)
     - Store result
  4. Collect across phantoms into flat arrays
```

### Key Details

- **Smoothing is applied to RAW noisy signals** (before M0 scaling or normalization). This is correct -- smoothing should happen in signal space, not normalized space.
- **Same subsampled voxels** as raw LS (`brain_indices[::10]`)
- **Same LS parameters** as raw LS (inherited from config -- currently broken T1_artery=1850)
- **Single sigma=2.0** hardcoded on line 790
- **Smoothed-LS results only reported at SNR=10**: Gated by `if snr_val == 10:` on line 1004 of `run_spatial_validation()`. At other SNR levels, smoothed-LS IS computed (the code runs for all SNRs) but the results are NOT logged to the LLM report and NOT included in the multi-SNR summary table.
- **Smoothed-LS metrics NOT in `_compute_snr_metrics_with_ci()`**: That method only processes raw LS. Smoothed-LS is only reported via `_log_llm_metrics()` at SNR=10.

### Data Flow Summary
```
noisy_signals --> gaussian_filter(sigma=2.0) --> smoothed_signals
smoothed_signals[:, i, j] --> reshape --> grid_search --> LS_optimizer --> sls_cbf, sls_att
```

### Limitations
1. Only sigma=2.0 -- no way to compare different smoothing levels
2. Reported only at SNR=10 despite being computed at all SNRs
3. No bootstrap CIs or win rate CIs for smoothed-LS
4. No Wilcoxon significance test for NN vs smoothed-LS

---

## 2. Phantom Tissue Map Structure (For Tissue Stratification)

### `SpatialPhantomGenerator.generate_phantom()` returns:
- `cbf_map`: (H, W) float32, in ml/100g/min, range [0, 200]
- `att_map`: (H, W) float32, in ms, range [100, 5000]
- `metadata`: dict with:
  - `tissue_map`: (H, W) int32, values: 0=background, 1=GM, 2=WM, 3=CSF
  - `pathologies`: list of dicts, each with `{type, center: (cy, cx), radius}`

### Critical Detail: Pathology Does NOT Update tissue_map
When pathology is applied (lines 241-259), the CBF/ATT values are overwritten at pathology voxels, but `tissue_map` still shows the ORIGINAL tissue type. For example, a tumor voxel that was originally GM will have `tissue_map[i,j] = 1` (GM) but `cbf_map[i,j] = 120.0` (tumor_hyper range).

### Partial Volume Effect Complication
After tissue assignment and pathology, a Gaussian blur is applied (line 269):
```python
cbf_map = gaussian_filter(cbf_map, sigma=self.pve_sigma)  # pve_sigma=1.0
att_map = gaussian_filter(att_map, sigma=self.pve_sigma)
```
This blurs boundaries, so the tissue_map (which has sharp boundaries) does NOT perfectly correspond to the blurred CBF/ATT values at tissue boundaries.

### Tissue Stratification Approach
Given these complexities, the stratification should:
1. Use `tissue_map` for GM/WM/CSF classification
2. Reconstruct pathology mask from `metadata['pathologies']` center/radius
3. Detect boundary voxels where tissue_map differs from neighbors
4. Exclude boundary voxels from pure-tissue metrics (contaminated by PVE blur)

---

## 3. Proposed Code Changes to validate.py

### Change 1: Add `smooth_sigmas` parameter to `_run_spatial_at_snr()`

**Current signature** (line 661):
```python
def _run_spatial_at_snr(self, snr_value, n_phantoms, phantom_size=64):
```

**Proposed signature**:
```python
def _run_spatial_at_snr(self, snr_value, n_phantoms, phantom_size=64,
                        smooth_sigmas=None):
```

**What changes inside the method**:

Replace the hardcoded sigma=2.0 section (lines 685-687, 788-820) with a loop over multiple sigmas.

#### Current code to replace (lines 685-687):
```python
all_smoothed_ls_cbf, all_smoothed_ls_att = [], []
all_smoothed_ls_true_cbf, all_smoothed_ls_true_att = [], []
all_nn_at_sls_cbf, all_nn_at_sls_att = [], []
```

#### Proposed replacement:
```python
# Multi-sigma smoothed-LS collectors
if smooth_sigmas is None:
    smooth_sigmas = [2.0]  # Backwards compatible default
sls_collectors = {}
for sigma in smooth_sigmas:
    sls_collectors[sigma] = {
        'cbf': [], 'att': [],
        'true_cbf': [], 'true_att': [],
        'nn_cbf': [], 'nn_att': [],
    }
```

#### Current code to replace (lines 788-820):
```python
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
```

#### Proposed replacement:
```python
# --- Multi-Sigma Smoothed-LS Inference ---
for sigma in smooth_sigmas:
    smoothed_signals = np.zeros_like(noisy_signals)
    for ch in range(noisy_signals.shape[0]):
        smoothed_signals[ch] = gaussian_filter(noisy_signals[ch], sigma=sigma)

    for idx in sample_indices:
        i, j = idx
        voxel_signal = smoothed_signals[:, i, j]
        try:
            init_guess = get_grid_search_initial_guess(voxel_signal, self.plds, ls_params)
            signal_reshaped = voxel_signal.reshape((len(self.plds), 2), order='F')
            beta, _, _, _ = fit_PCVSASL_misMatchPLD_vectInit_pep(
                pldti, signal_reshaped, init_guess, **ls_params
            )
            sls_cbf_val = beta[0] * 6000.0
            sls_att_val = beta[1]
        except Exception:
            sls_cbf_val = np.nan
            sls_att_val = np.nan

        if not np.isnan(sls_cbf_val):
            sls_collectors[sigma]['cbf'].append(sls_cbf_val)
            sls_collectors[sigma]['att'].append(sls_att_val)
            sls_collectors[sigma]['true_cbf'].append(true_cbf_map[i, j])
            sls_collectors[sigma]['true_att'].append(true_att_map[i, j])
            sls_collectors[sigma]['nn_cbf'].append(nn_cbf[i, j])
            sls_collectors[sigma]['nn_att'].append(nn_att[i, j])
```

#### Update the return dict (lines 850-867):

Replace the single smoothed-LS arrays with the multi-sigma dict:
```python
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
    # Multi-sigma smoothed-LS results
    'smoothed_ls': {
        sigma: {k: np.array(v) for k, v in coll.items()}
        for sigma, coll in sls_collectors.items()
    },
    # Tissue metadata for stratification
    'all_tissue_labels': all_tissue_labels,
}
```

### Change 2: Add tissue label collection

In the phantom loop, after line 694 where `metadata` is returned, add tissue classification and per-voxel label collection.

**Add at top of method** (near the other collector lists):
```python
all_tissue_labels = []  # per-voxel tissue category string
```

**Add after NN inference (after line 754), before LS fitting**:
```python
# Classify voxels by tissue type for stratified metrics
tissue_map = metadata['tissue_map']
pathologies = metadata.get('pathologies', [])

# Build pathology mask from metadata
pathology_mask = np.zeros((phantom_size, phantom_size), dtype=bool)
for p in pathologies:
    cy, cx = p['center']
    radius = p['radius']
    y, x = np.ogrid[:phantom_size, :phantom_size]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    pathology_mask |= (dist <= radius * 1.2)

# Build boundary mask (adjacent pixels differ in tissue type)
boundary_mask = np.zeros((phantom_size, phantom_size), dtype=bool)
for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
    shifted = np.roll(np.roll(tissue_map, di, axis=0), dj, axis=1)
    boundary_mask |= (tissue_map != shifted)
from scipy.ndimage import binary_dilation
boundary_mask = binary_dilation(boundary_mask, iterations=1)
```

**In the brain voxel collection loop (lines 822-827)**, add tissue labels:
```python
# Collect brain voxel values with tissue labels
brain_mask_bool = mask > 0
all_nn_cbf.extend(nn_cbf[brain_mask_bool].flatten())
all_nn_att.extend(nn_att[brain_mask_bool].flatten())
all_true_cbf.extend(true_cbf_map[brain_mask_bool].flatten())
all_true_att.extend(true_att_map[brain_mask_bool].flatten())

# Tissue labels for each brain voxel
for i in range(phantom_size):
    for j in range(phantom_size):
        if brain_mask_bool[i, j]:
            if pathology_mask[i, j]:
                all_tissue_labels.append('pathology')
            elif boundary_mask[i, j]:
                all_tissue_labels.append('boundary')
            elif tissue_map[i, j] == 1:
                all_tissue_labels.append('gray_matter')
            elif tissue_map[i, j] == 2:
                all_tissue_labels.append('white_matter')
            elif tissue_map[i, j] == 3:
                all_tissue_labels.append('csf')
            else:
                all_tissue_labels.append('other')
```

### Change 3: Add `_compute_smoothed_ls_metrics()` method

New method to compute metrics for each sigma level:
```python
def _compute_smoothed_ls_metrics(self, results, sigma):
    """Compute metrics for smoothed-LS at a specific sigma."""
    sls_data = results['smoothed_ls'].get(sigma, {})
    if len(sls_data.get('cbf', [])) == 0:
        return None

    sls_cbf_err = np.abs(sls_data['cbf'] - sls_data['true_cbf'])
    sls_att_err = np.abs(sls_data['att'] - sls_data['true_att'])
    nn_at_sls_cbf_err = np.abs(sls_data['nn_cbf'] - sls_data['true_cbf'])
    nn_at_sls_att_err = np.abs(sls_data['nn_att'] - sls_data['true_att'])

    sls_cbf_mae, lo, hi = self._bootstrap_ci(sls_cbf_err)
    sls_att_mae, att_lo, att_hi = self._bootstrap_ci(sls_att_err)
    cbf_wr, wr_lo, wr_hi = self._bootstrap_ci_winrate(nn_at_sls_cbf_err, sls_cbf_err)
    att_wr, awr_lo, awr_hi = self._bootstrap_ci_winrate(nn_at_sls_att_err, sls_att_err)

    return {
        'sigma': sigma,
        'cbf_mae': sls_cbf_mae, 'cbf_mae_ci': [lo, hi],
        'att_mae': sls_att_mae, 'att_mae_ci': [att_lo, att_hi],
        'cbf_win_rate': cbf_wr, 'cbf_win_rate_ci': [wr_lo, wr_hi],
        'att_win_rate': att_wr, 'att_win_rate_ci': [awr_lo, awr_hi],
        'n_samples': len(sls_data['cbf']),
    }
```

### Change 4: Add `_compute_tissue_metrics()` method

```python
def _compute_tissue_metrics(self, results):
    """Compute per-tissue-type metrics from collected voxel data."""
    labels = np.array(results['all_tissue_labels'])
    nn_cbf = results['all_nn_cbf']
    nn_att = results['all_nn_att']
    true_cbf = results['all_true_cbf']
    true_att = results['all_true_att']

    tissue_metrics = {}
    for tissue_name in ['gray_matter', 'white_matter', 'pathology', 'boundary']:
        mask = labels == tissue_name
        n = np.sum(mask)
        if n < 10:
            continue

        cbf_err = np.abs(nn_cbf[mask] - true_cbf[mask])
        att_err = np.abs(nn_att[mask] - true_att[mask])

        cbf_mae, lo, hi = self._bootstrap_ci(cbf_err)
        att_mae, alo, ahi = self._bootstrap_ci(att_err)

        tissue_metrics[tissue_name] = {
            'nn_cbf_mae': cbf_mae, 'nn_cbf_mae_ci': [lo, hi],
            'nn_cbf_bias': float(np.mean(nn_cbf[mask] - true_cbf[mask])),
            'nn_att_mae': att_mae, 'nn_att_mae_ci': [alo, ahi],
            'nn_att_bias': float(np.mean(nn_att[mask] - true_att[mask])),
            'n_voxels': int(n),
            'mean_true_cbf': float(np.mean(true_cbf[mask])),
            'mean_true_att': float(np.mean(true_att[mask])),
        }

    return tissue_metrics
```

### Change 5: Update `run_spatial_validation()` to use multi-sigma and tissue metrics

In `run_spatial_validation()`, update the call (line 977):
```python
# Current:
results = self._run_spatial_at_snr(snr_val, n_phantoms, phantom_size)

# Proposed:
smooth_sigmas = [0.5, 1.0, 2.0, 3.0]
results = self._run_spatial_at_snr(snr_val, n_phantoms, phantom_size,
                                   smooth_sigmas=smooth_sigmas)
```

After computing standard metrics, add smoothed-LS and tissue metrics:
```python
# Smoothed-LS metrics for each sigma
sls_metrics = {}
for sigma in smooth_sigmas:
    m = self._compute_smoothed_ls_metrics(results, sigma)
    if m is not None:
        sls_metrics[sigma] = m

# Tissue-stratified metrics
tissue_metrics = self._compute_tissue_metrics(results)

# Store alongside existing metrics
multi_snr_results[snr_val]['smoothed_ls'] = sls_metrics
multi_snr_results[snr_val]['tissue_stratified'] = tissue_metrics
```

### Change 6: Fix LS parameters (P1 blocker)

In `_run_spatial_at_snr()` lines 763-770, the LS parameters come from `self.params` which inherits from config. This needs to be overridden with corrected values:

```python
# Current (line 763-770):
ls_params = {
    'T1_artery': self.params.T1_artery,
    'T_tau': self.params.T_tau,
    'alpha_PCASL': self.params.alpha_PCASL,
    'alpha_VSASL': self.params.alpha_VSASL,
    'T2_factor': self.params.T2_factor,
    'alpha_BS1': self.params.alpha_BS1
}

# Proposed: Use corrected values for LS, not model training params
ls_params = {
    'T1_artery': 1650.0,     # ASL consensus (Alsop 2015), NOT config value
    'T_tau': self.params.T_tau,
    'alpha_PCASL': self.params.alpha_PCASL,
    'alpha_VSASL': self.params.alpha_VSASL,
    'T2_factor': self.params.T2_factor,
    'alpha_BS1': self.params.alpha_BS1  # 1.0 for synthetic, 0.93 for in-vivo
}
```

**NOTE**: This change should be coordinated with the foundation-lead. It should also match the T1_artery used in signal generation (line 707). The signal generation currently uses `self.params.T1_artery` too, which may also need correction. The cleanest fix is to correct ASLParameters defaults so both signal generation AND LS fitting use 1650.

---

## 4. Runtime Impact

### Current runtime estimate (per SNR level, 50 phantoms)
- NN inference: ~1s per phantom = ~50s
- LS fitting (1:10 subsample): ~100 voxels/phantom * ~10ms/voxel = ~50s
- Smoothed-LS (sigma=2.0): ~50s additional
- **Total per SNR: ~2.5 min**

### Proposed runtime (per SNR level, 50 phantoms)
- NN inference: ~50s (unchanged)
- LS fitting: ~50s (unchanged)
- Smoothed-LS (4 sigmas): ~200s (4x current)
- Tissue classification: negligible
- **Total per SNR: ~5 min**

### Full multi-SNR run (8 levels)
- SNR=10 with 50 phantoms: ~5 min
- 7 other SNRs with 20 phantoms each: ~2 min each = ~14 min
- **Total: ~19 min** (vs ~2.5 min current single-SNR)

This is very manageable for a one-time re-validation.

---

## 5. Alternative: Use run_corrected_validation.py

The standalone script `/Users/adikondepudi/Desktop/asl-multiverse/run_corrected_validation.py` (already created) implements ALL of these changes from scratch without modifying validate.py. It:
- Hardcodes corrected LS params (T1_artery=1650)
- Supports multi-sigma smoothed-LS
- Includes tissue stratification
- Adds Wilcoxon significance tests
- Outputs comprehensive JSON results

**Recommendation**: Use `run_corrected_validation.py` for the immediate P3 re-validation. Apply the changes to `validate.py` later as a permanent improvement, coordinated with P1 fixes.
