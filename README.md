# Time-Resolved Analysis of Circadian Rhythm Dynamics

This repository contains the Python workflow and outputs for an MSc Bioinformatics project examining circadian rhythm dynamics in ex vivo PER2::LUC recordings from the suprachiasmatic nucleus (SCN) and dorsal root ganglia (DRG) before and after forskolin treatment. It also evaluates continuous wavelet transform (CWT), short-time Fourier transform (STFT), and Hilbert-based methods using recording-anchored synthetic signals.

## Requirements

The workflow was developed with Python 3.12.4 and requires NumPy, pandas, SciPy, Matplotlib, and pyBOAT.

```bash
python -m pip install numpy pandas scipy matplotlib pyboat
```

## Repository structure

- `data/raw/`: original LumiCycle recordings
- `scripts/`: analysis scripts
- `results/`: tables and figures produced by each analysis step

All scripts construct paths relative to the project directory and can therefore be run from any working directory.

## Workflow

Run the scripts in the following order:

```bash
python scripts/MSC-STEP1A.py
python scripts/MSC-STEP1B.py
python scripts/MSC-STEP2A.py
python scripts/MSC-STEP2B.py
python scripts/MSC-STEP3A-RUN.py
python scripts/MSC-STEP3B.py
python scripts/MSC-STEP3C.py
python scripts/MSC-STEP3D.py
python scripts/MSC-STEP4.py
python scripts/MSC-STEP5.py
```

Figures for STEP3A–3C can be regenerated without rerunning the benchmarks:

```bash
python scripts/MSC-STEP3ABC-FIGURES.py
```

## Analysis outline

STEP1 prepares the recordings and empirical features. STEP2 fits a ten-parameter deterministic model and generates 530 local synthetic sources from 53 recording anchors. STEP3 benchmarks the three time-resolved methods, tests robustness, performs oracle rescue analyses, and evaluates dynamic period profiles. STEP4 applies the selected methods to the experimental recordings. STEP5 performs animal-level DRG analyses.

Eligibility is target-specific. Eligible cases form the primary synthetic benchmark, difficult cases provide supplementary evidence, and ineligible cases are excluded only from the affected target analysis. Bootstrap confidence intervals resample whole recording anchors after combining the ten variants belonging to each anchor. Biological inference in STEP4 and STEP5 is performed at the animal level.

STFT phase shift is not evaluated. The corrections in STEP3C use known synthetic nuisance information and are oracle analyses rather than procedures intended for direct application to unknown experimental recordings.
