# monitorviz

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
src/monitorviz/   código del paquete
  models/         modelos pydantic
  io/             lectura de runs
  transforms/     cálculos derivados y agregaciones
  viz/            helpers de plotting
tests/            tests + fixtures
notebooks/        análisis ejecutables
data/             submódulo con runs (TFG-DATA)
```

## Uso

(Pendiente: se rellena tras implementar loaders y notebooks.)
