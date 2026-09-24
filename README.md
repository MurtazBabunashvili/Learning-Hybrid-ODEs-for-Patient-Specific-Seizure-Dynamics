# Stable Hybrid ODEs for Patient-Specific Seizure Forecasting

Learns continuous-time seizure dynamics from CHB-MIT scalp EEG (24 patients,
canonical 22-channel montage, 256 Hz → 64 Hz → 16 Hz model rate) and forecasts
20 s of EEG from a 10 s prefix.

## Model

Latent neural-mass state with patient-specific parameters plus a learned
residual, linear observation into sensor space:

$$ \dot{x} = f_{\mathrm{Epileptor}}(x; \theta_i) + g_\phi(x), \qquad y = \mu + Hx $$

$x \in \mathbb{R}^5$ (reduced Epileptor), $\theta_i=(x_0, I_1, I_2)$ per patient,
$g_\phi$ a small-init MLP, $H$ column-normalized 22×5. The forecast state is
estimated from the prefix only (ridge init + few shooting GD steps inside a
±2.5 trust region); the 20 s target is never shown to the estimator.

## Data / split

`python scripts/build_manifest.py && python scripts/build_windows.py`
(one-time; training reads `artifacts/splits/splits.pkl`). Onset-relative
window sampling (40/20/20/10/10 inter/far-pre/near-pre/ictal/post), disjoint
patient holdout (train 18 / val 3 / test 3). 3 disjoint-montage recordings
excluded; 28 files zero-fill 4 temporal channels.

## Run

```powershell
python -u main.py [config/config.yaml]
```

All parameters live in `config/config.yaml`. Loss = 20 s forecast MSE +
$\lambda_R \int \|g\|^2$. `main.py` is the entire experiment; `src/` holds
only the data pipeline it imports.

## Status

Forecast floor ~1.10 train / 0.79 val (20 s horizon); causal persistence
baseline 1.68. Residual on/off ablation on one checkpoint: 0.92 vs 1.14 —
the correction carries ~20% of forecast skill. Legacy code preserved in
`legacy_backup.zip`.
