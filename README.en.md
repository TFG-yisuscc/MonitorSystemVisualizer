# monitorviz

> [Versión en español](README.md)

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
src/monitorviz/   package source
  models/         pydantic models
  io/             run readers
  transforms/     derived metrics and aggregations
  viz/            faceted plotting helpers
tests/            tests + fixtures
notebooks/        executable analyses
data/             git submodule with runs (TFG-DATA)
```

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

> Secondary notebooks were run on a reduced subset of the data. Running them on the full dataset may take a long time and is not recommended. They may also contain errors or incorrect visualisations.
