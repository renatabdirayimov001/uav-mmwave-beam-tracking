# Leakage-aware, closed-loop evaluation of camera-aided beam tracking for drone mmWave links

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24322380.svg)](https://doi.org/10.5281/zenodo.24322380)

Code and results for the paper *"Beyond Random Splits and Oracle Beam History: Leakage-Aware, Closed-Loop Evaluation of Camera-Aided Beam Tracking for Drone Millimeter-Wave Links"* (under review).

**Author:** Renat Abdirayimov, Department of Infocommunication Engineering, Tashkent University of Information Technologies named after Muhammad al-Khwarizmi, Tashkent, Uzbekistan ([ORCID 0009-0005-3636-8864](https://orcid.org/0009-0005-3636-8864)).

The experiments use the real 60 GHz drone measurements of **DeepSense 6G Scenario 23**. The dataset is **not** redistributed here; download it from the DeepSense 6G website (https://www.deepsense6g.net/scenario-23/) or use a Kaggle mirror of the development set.

## What is evaluated

* **Split protocols:** random 5-fold vs flight-disjoint 5-fold (GroupKFold over flights).
* **Horizons:** current beam (h = 0) and 1, 3, 5 samples ahead (median sample interval 142 ms).
* **Models:** persistence, GPS (KNN, LightGBM), GPS + history, ResNet-18 embedding + LightGBM (alone, + GPS, + GPS + history), GRU over beam + telemetry, fine-tuned ResNet-18 (plain, augmented, + history), and the proposed persistence-anchored residual predictor **PAR** with causal background-subtraction cues.
* **Ablations:** absolute vs residual target, history only, without telemetry, without image cue, two-sided vs causal background.
* **Closed-loop tracking (h = 0):** only the top-k predicted beams (k = 1, 2, 3) are measured and the strongest measured beam becomes the history. **PAR-CF** = log-linear fusion of PAR with an augmentation-trained ResNet-18.
* **Image degradations:** blur, low light, occlusion, 30% dropped frames (models trained on clean images).
* **Statistics:** per-flight Wilcoxon signed-rank tests with Holm correction, cluster-bootstrap 95% CIs.

## Repository layout

| Path | Content |
|---|---|
| `notebooks/00_figure1_assets.ipynb` | Frames and difference maps for Fig. 1 |
| `notebooks/01_benchmark_v4_open_loop.ipynb` | First open-loop benchmark (two-sided background) |
| `notebooks/02_benchmark_v5_causal_ablations.ipynb` | Main benchmark: causal cues, all baselines, ablations, robustness, cost |
| `notebooks/03_closed_loop_v6_fusion.ipynb` | Closed-loop tracking under degradations and PAR-CF fusion |
| `results/v4`, `results/v5`, `results/v6` | Per-fold, per-flight and summary results (CSV), logs (`report.txt`) |
| `scripts/make_figures.py`, `scripts/make_tables.py` | Figures and LaTeX tables of the paper from the CSV files |

## Running

The notebooks were run on Kaggle (GPU T4) with two inputs: the Scenario 23 development set and the ImageNet ResNet-18 weights file `resnet18-f37072fd.pth` (torchvision). Import a notebook, add both inputs, select a GPU accelerator and use *Save & Run All*. Run times: v5 about 2 h, v6 about 1 h.

To regenerate the figures and tables from the stored results:

```bash
pip install -r requirements.txt
python scripts/make_figures.py   # writes figures/
python scripts/make_tables.py    # writes tables/
```

## Main results (flight-disjoint)

| | h = 0 | h = 1 | h = 3 | h = 5 |
|---|---|---|---|---|
| Persistence | 0.823 | 0.738 | 0.615 | 0.538 |
| GRU (beam + telemetry) | 0.813 | 0.739 | 0.653 | 0.592 |
| ResNet-18 | 0.832 | 0.671 | 0.472 | 0.350 |
| **PAR** | **0.852** | **0.804** | **0.731** | **0.668** |

Closed-loop tracking accuracy with two probed beams per sample (8.2% of a full sweep):

| | Clean | Blur | Frame drop | Low light | Occlusion |
|---|---|---|---|---|---|
| ResNet-18 (augmented) | 0.971 | 0.921 | 0.950 | 0.815 | 0.639 |
| PAR | 0.919 | 0.806 | 0.918 | 0.864 | 0.861 |
| **PAR-CF** | **0.976** | **0.965** | **0.964** | **0.965** | **0.953** |

## License

Code: MIT (see `LICENSE`). The DeepSense 6G data are subject to the license of the dataset authors.
