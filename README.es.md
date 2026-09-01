# monitorviz

> [English version](README.md)

Visualizador de métricas para los datos producidos por
[MonitorSystemCplusplus](https://github.com/TFG-yisuscc/MonitorSystemCplusplus),
parte de mi TFG.

## Instalación

```bash
git clone <url-del-repo> visualizador-tfg
cd visualizador-tfg
git submodule add git@github.com:TFG-yisuscc/TFG-DATA.git data
git submodule update --init --recursive
uv sync
```

## Configuración en PyCharm

1. File > Settings > Project > Python Interpreter
2. Add Interpreter > Add Local Interpreter > Select existing
3. Selecciona la ruta: `<proyecto>/.venv/bin/python`
   (en Windows: `<proyecto>\.venv\Scripts\python.exe`)

En PyCharm 2024.3.2+ también puedes elegir el tipo "uv" directamente.

## Estructura

```
.
├── src/monitorviz/        Código del paquete
│   ├── models/             Modelos pydantic (hw, meta, prompt, run)
│   ├── io/                 Descubrimiento, lectura y parseo de runs
│   ├── transforms/         Métricas derivadas, agregaciones y FoM
│   └── viz/                Primitivas de plotting, figuras compuestas y estilo
├── tests/                  Suite de tests (pytest)
│   └── fixtures/           Runs de ejemplo usados en los tests
├── notebooks/              Análisis ejecutables
│   ├── 06..12_*.ipynb       Notebooks de los experimentos principales (E0-E5, perplejidad)
│   ├── secundarios/         Notebooks de exploración/soporte (00-05)
│   └── figures/              Imágenes exportadas desde los notebooks
└── data/                   Submódulo git (TFG-DATA) con los runs en bruto
    ├── E0-FAN/ E0-NOFAN/    Experimento baseline (fan activo/pasivo)
    ├── E1/                  Experimento de cuantización
    ├── E2/                  Experimento de tamaño de contexto
    ├── E3/                  Experimento de batch size
    ├── E5/                  Experimento del acelerador Hailo
    └── Perplejidad/         Experimento de perplejidad
```

- **`src/monitorviz/`** — el paquete instalable. `models/` define los esquemas pydantic para los metadatos de un run, los prompts y las muestras de hardware; `io/` descubre y parsea los directorios de runs en disco; `transforms/` convierte las muestras en bruto en métricas derivadas, agregaciones y figuras de mérito (FoM); `viz/` contiene los helpers de plotting reutilizables que llaman los notebooks.
- **`tests/`** — tests unitarios del paquete, con `fixtures/` conteniendo runs sintéticos pequeños (distintos formatos: llama.cpp, OLLAMA, con/sin métricas de hardware, con/sin `meta.yaml`) usados para probar los loaders sin necesitar el dataset completo.
- **`notebooks/`** — todos los notebooks de análisis. Los notebooks numerados en la raíz (06-12) son los experimentos principales del TFG; `secundarios/` contiene los notebooks exploratorios previos y las figuras de metodología; `figures/` recoge los PNG exportados que se referencian en la memoria.
- **`data/`** — un submódulo git que apunta a [TFG-DATA](https://github.com/TFG-yisuscc/TFG-DATA), con los runs en bruto producidos por MonitorSystemCplusplus, una subcarpeta por experimento.

## Uso

### Lanzar JupyterLab

```bash
uv run jupyter lab
```

### Notebooks

Los notebooks están en `notebooks/`. Los experimentos principales (06–12) están en la raíz:

| Notebook | Contenido |
|---|---|
| `06_E0_baseline.ipynb` | Experimento E0 — baseline (fan activo/pasivo) |
| `07_E1_cuantizacion.ipynb` | Experimento E1 — cuantización |
| `08_E2_contexto.ipynb` | Experimento E2 — tamaño de contexto |
| `09_E3_batch.ipynb` | Experimento E3 — batch size |
| `10_E4_engine.ipynb` | Experimento E4 — OLLAMA vs llama.cpp |
| `11_E5_hailo.ipynb` | Experimento E5 — acelerador Hailo |
| `12_perplejidad.ipynb` | Análisis de perplejidad |
> La subseccióbn de anexos en estos notebooks incluye intentos de gráficas que no se visualizan correctamente o que carecen de relevancia, se recomienda ignorarlas,

Los notebooks de soporte y exploración que se han utilizado para comprender el funcionamiento y peculiaridades de los LLM están en `notebooks/secundarios/`:

| Notebook | Contenido |
|---|---|
| `00_metodologia.ipynb` | Diagramas y figuras metodológicas del TFG |
| `01_exploracion_run.ipynb` | Exploración detallada de un run individual |
| `02_hardware_timeline.ipynb` | Series temporales de hardware (temperatura, potencia…) |
| `03_inferencia_metricas.ipynb` | Comparativa de métricas de inferencia entre modelos |
| `04_comparativa_global.ipynb` | Comparativa global entre experimentos |
| `05_experimentos_parametricos.ipynb` | Análisis paramétrico (contexto, batch, cuantización) |

Todos los notebooks detectan automáticamente la raíz del proyecto independientemente de desde dónde se lancen (`notebooks/` o `notebooks/secundarios/`).

> **Nota sobre mayúsculas:** el submódulo git clona el directorio de datos como `tfg-data` (minúsculas), mientras que los notebooks esperan `TFG-DATA`. Si es necesario, ajusta el nombre en la celda de configuración de cada notebook.
> Los notebooks secundarios se emplearon en un subconjunto reducido, de usar el dataset completo, pueden tardar mucho por lo que se desaconseja su uso. Además pueden contener errores o visualizaciones incorrectas.