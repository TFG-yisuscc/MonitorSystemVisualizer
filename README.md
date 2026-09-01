# monitorviz

> [Versión en español](README.es.md)

Metrics visualizer for data produced by
[MonitorSystemCplusplus](https://github.com/TFG-yisuscc/MonitorSystemCplusplus),
part of my Bachelor's thesis (TFG).

## Installation

```bash
git clone <repo-url> visualizador-tfg
cd visualizador-tfg
git submodule add git@github.com:TFG-yisuscc/TFG-DATA.git data
git submodule update --init --recursive
uv sync
```

## PyCharm setup

1. File > Settings > Project > Python Interpreter
2. Add Interpreter > Add Local Interpreter > Select existing
3. Select the path: `<project>/.venv/bin/python`
   (on Windows: `<project>\.venv\Scripts\python.exe`)

In PyCharm 2024.3.2+ you can also choose the "uv" interpreter type directly.

## Structure

```
.
├── src/monitorviz/        Package source
│   ├── models/             Pydantic models (hw, meta, prompt, run)
│   ├── io/                 Run discovery, loaders and parsers
│   ├── transforms/         Derived metrics, aggregations and FoM
│   └── viz/                Plotting primitives, composite figures and style
├── tests/                  Pytest suite
│   └── fixtures/           Sample runs used by the tests
├── notebooks/              Executable analyses
│   ├── 06..12_*.ipynb       Main experiment notebooks (E0-E5, perplexity)
│   ├── secundarios/         Exploratory/support notebooks (00-05)
│   └── figures/              Chart images exported from the notebooks
└── data/                   Git submodule (TFG-DATA) with the raw runs
    ├── E0-FAN/ E0-NOFAN/    Baseline experiment (fan on/off)
    ├── E1/                  Quantization experiment
    ├── E2/                  Context-size experiment
    ├── E3/                  Batch-size experiment
    ├── E5/                  Hailo accelerator experiment
    └── Perplejidad/         Perplexity experiment
```

- **`src/monitorviz/`** — the installable package. `models/` defines the pydantic schemas for a run's metadata, prompts and hardware samples; `io/` discovers and parses run directories on disk; `transforms/` turns raw samples into derived metrics, aggregations and figures of merit (FoM); `viz/` holds the reusable plotting helpers the notebooks call into.
- **`tests/`** — unit tests for the package, with `fixtures/` holding small synthetic runs (different formats: llama.cpp, OLLAMA, with/without hardware metrics, with/without `meta.yaml`) used to exercise the loaders without needing the full dataset.
- **`notebooks/`** — all analysis notebooks. The numbered notebooks at the root (06-12) are the main thesis experiments; `secundarios/` holds earlier exploratory notebooks and the methodology figures; `figures/` collects exported PNGs referenced by the write-up.
- **`data/`** — a git submodule pointing at [TFG-DATA](https://github.com/TFG-yisuscc/TFG-DATA), containing the raw runs produced by MonitorSystemCplusplus, one subfolder per experiment.

## Usage

### Launch JupyterLab

```bash
uv run jupyter lab
```

### Notebooks

Notebooks live in `notebooks/`. The main experiment notebooks (06–12) are at the root:

| Notebook | Contents |
|---|---|
| `06_E0_baseline.ipynb` | Experiment E0 — baseline (fan on/off) |
| `07_E1_cuantizacion.ipynb` | Experiment E1 — quantization |
| `08_E2_contexto.ipynb` | Experiment E2 — context size |
| `09_E3_batch.ipynb` | Experiment E3 — batch size |
| `10_E4_engine.ipynb` | Experiment E4 — OLLAMA vs llama.cpp |
| `11_E5_hailo.ipynb` | Experiment E5 — Hailo accelerator |
| `12_perplejidad.ipynb` | Perplexity analysis |

> The annex subsections in these notebooks contain chart drafts that either render incorrectly or lack relevance — they can safely be ignored.

Support and exploratory notebooks used to understand LLM behaviour and quirks are in `notebooks/secundarios/`:

| Notebook | Contents |
|---|---|
| `00_metodologia.ipynb` | Methodology diagrams and figures for the thesis |
| `01_exploracion_run.ipynb` | Detailed exploration of a single run |
| `02_hardware_timeline.ipynb` | Hardware time series (temperature, power…) |
| `03_inferencia_metricas.ipynb` | Inference metrics comparison across models |
| `04_comparativa_global.ipynb` | Global comparison across experiments |
| `05_experimentos_parametricos.ipynb` | Parametric analysis (context, batch, quantization) |

All notebooks automatically detect the project root regardless of where they are launched from (`notebooks/` or `notebooks/secundarios/`).

> **Case sensitivity note:** the git submodule checks out the data directory as `tfg-data` (lowercase), while the notebooks expect `TFG-DATA`. If needed, update the name in the configuration cell of each notebook.

> Secondary notebooks were run on a reduced subset of the data. Running them on the full dataset may take a long time and is not recommended. They may also contain errors or incorrect visualisations.
