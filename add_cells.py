#!/usr/bin/env python3
"""Script para añadir celdas a los notebooks."""

import json
from pathlib import Path

def create_markdown_cell(source_lines):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source_lines
    }

def create_code_cell(source_lines):
    """Create a code cell."""
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source_lines
    }

def add_cells_to_notebook(nb_path, cells):
    """Add cells to a notebook."""
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    nb['cells'].extend(cells)

    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"✓ {nb_path}: {len(cells)} celdas añadidas")

# === NOTEBOOK 02 ===
nb02_path = Path('/home/yisus/PycharmProjects/Visualizer/notebooks/02_hardware_timeline.ipynb')

nb02_cells = [
    create_markdown_cell([
        "## Comparativa de métricas hardware por modelo\n",
        "\n",
        "Barplot agrupado de cuatro métricas hardware clave, agrupadas por modelo\n",
        "y coloreadas por configuración (ventilador activo / sin ventilador).\n",
        "Inspirado en Velasco-Montero et al. (2022), Cap. 4, Fig. 4.5.\n",
        "\n",
        "Cuando estén disponibles runs con fan=False (Experimento E0), aparecerá\n",
        "la barra comparativa. Con los datos actuales (fan=True) se muestra una\n",
        "barra por modelo."
    ]),
    create_code_cell([
        "# Agregar métricas hw por run\n",
        "hw_full = coll.hw_metrics_df()\n",
        "pm_full = coll.prompt_metrics_df()\n",
        "\n",
        "hw_summary_rows = []\n",
        "for r in coll.runs:\n",
        "    hw_r = hw_full[hw_full[\"run_id\"] == r.run_id]\n",
        "    pm_r = pm_full[\n",
        "        (pm_full[\"run_id\"] == r.run_id) &\n",
        "        (~pm_full[\"is_empty_generation\"]) &\n",
        "        (pm_full[\"latency_ms\"] > 0)\n",
        "    ]\n",
        "    if hw_r.empty:\n",
        "        continue\n",
        "    active = hw_r[hw_r[\"cpu_usage_pct\"] > 50]\n",
        "    label = getattr(r, \"model_label\", r.model_short)\n",
        "    fan_label = \"Con ventilador\" if r.meta.fan else \"Sin ventilador\"\n",
        "    hw_summary_rows.append({\n",
        "        \"model\":    label,\n",
        "        \"config\":   fan_label,\n",
        "        \"cpu_pct\":  active[\"cpu_usage_pct\"].mean() if not active.empty\n",
        "                    else float(\"nan\"),\n",
        "        \"tokens_s\": pm_r[\"tokens_per_second\"].mean() if not pm_r.empty\n",
        "                    else float(\"nan\"),\n",
        "        \"power_w\":  hw_r[\"internal_power_w\"].mean(),\n",
        "        \"mem_mb\":   hw_r[\"mem_used_mb\"].mean(),\n",
        "    })\n",
        "\n",
        "hw_summary = pd.DataFrame(hw_summary_rows)\n",
        "\n",
        "METRICS = [\n",
        "    (\"cpu_pct\",   \"CPU media (%)\",      \"(a)\", (50, 105)),\n",
        "    (\"tokens_s\",  \"Throughput (tok/s)\", \"(b)\", None),\n",
        "    (\"power_w\",   \"Potencia (W)\",       \"(c)\", None),\n",
        "    (\"mem_mb\",    \"Memoria (MB)\",       \"(d)\", None),\n",
        "]\n",
        "\n",
        "models  = sorted(hw_summary[\"model\"].unique())\n",
        "configs = sorted(hw_summary[\"config\"].unique())\n",
        "colors  = [\"steelblue\", \"tomato\", \"seagreen\", \"goldenrod\"]\n",
        "x_pos   = np.arange(len(models))\n",
        "n_cfg   = len(configs)\n",
        "width   = 0.75 / max(n_cfg, 1)\n",
        "\n",
        "fig, axes = plt.subplots(len(METRICS), 1, figsize=(12, 4 * len(METRICS)))\n",
        "if len(METRICS) == 1:\n",
        "    axes = [axes]\n",
        "\n",
        "for ax, (col, ylabel, label, ylim) in zip(axes, METRICS):\n",
        "    data = hw_summary.dropna(subset=[col])\n",
        "    for i, (cfg, color) in enumerate(zip(configs, colors)):\n",
        "        vals = []\n",
        "        for m in models:\n",
        "            sub = data[(data[\"model\"] == m) & (data[\"config\"] == cfg)]\n",
        "            vals.append(sub[col].mean() if not sub.empty else float(\"nan\"))\n",
        "        offset = (i - n_cfg / 2 + 0.5) * width\n",
        "        ax.bar(x_pos + offset, vals, width * 0.9,\n",
        "               label=cfg, color=color, alpha=0.85)\n",
        "    ax.set_xticks(x_pos)\n",
        "    ax.set_xticklabels(models)\n",
        "    ax.set_ylabel(ylabel)\n",
        "    if ylim:\n",
        "        ax.set_ylim(*ylim)\n",
        "    ax.text(-0.06, 1.03, label, transform=ax.transAxes,\n",
        "            fontweight=\"bold\", fontsize=12)\n",
        "    if col == \"cpu_pct\":\n",
        "        ax.legend(loc=\"upper right\")\n",
        "\n",
        "fig.suptitle(\n",
        "    \"Métricas hardware por modelo y configuración de refrigeración\",\n",
        "    y=1.01, fontsize=13,\n",
        ")\n",
        "fig.tight_layout()\n",
        "plt.show()\n"
    ])
]

# === NOTEBOOK 04 ===
nb04_path = Path('/home/yisus/PycharmProjects/Visualizer/notebooks/04_comparativa_global.ipynb')

nb04_cells = [
    create_markdown_cell([
        "## Resumen global: gráficas de radar\n",
        "\n",
        "Visualización polar de las métricas clave normalizadas en el rango [0, 1],\n",
        "donde el exterior del radar siempre representa el mejor valor posible.\n",
        "Inspirado en Velasco-Montero et al. (2022), Cap. 4, Fig. 4.8.\n",
        "\n",
        "Las métricas inversas (menor es mejor: TTFT, PPL, energía, temperatura)\n",
        "se invierten antes de normalizar, de modo que un vértice exterior significa\n",
        "\"mejor rendimiento\" para todas las dimensiones.\n",
        "\n",
        "Se generan dos vistas:\n",
        "- **(a) Vista de rendimiento**: throughput, TTFT, PPL, energía/token.\n",
        "- **(b) Vista de hardware**: MBU, temperatura, potencia, memoria."
    ]),
    create_code_cell([
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "def _radar_chart(\n",
        "    df: pd.DataFrame,\n",
        "    metrics: list[tuple[str, str, bool]],\n",
        "    title: str,\n",
        "    ax: plt.Axes,\n",
        "    colors: list[str],\n",
        ") -> None:\n",
        "    \"\"\"Draw a radar chart on ax.\n",
        "\n",
        "    metrics: list of (column, label, higher_is_better)\n",
        "    \"\"\"\n",
        "    # Keep only rows with all metrics available\n",
        "    cols = [m[0] for m in metrics]\n",
        "    data = df.dropna(subset=cols).copy()\n",
        "    if data.empty:\n",
        "        ax.text(0.5, 0.5, \"Sin datos suficientes\",\n",
        "                ha=\"center\", va=\"center\", transform=ax.transAxes)\n",
        "        return\n",
        "\n",
        "    labels = [m[1] for m in metrics]\n",
        "    N = len(metrics)\n",
        "    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()\n",
        "    angles += angles[:1]  # close the polygon\n",
        "\n",
        "    # Normalize each metric to [0, 1] (1 = best)\n",
        "    normalized = {}\n",
        "    for col, _, higher_is_better in metrics:\n",
        "        col_min = data[col].min()\n",
        "        col_max = data[col].max()\n",
        "        rng = col_max - col_min if col_max != col_min else 1.0\n",
        "        if higher_is_better:\n",
        "            normalized[col] = (data[col] - col_min) / rng\n",
        "        else:\n",
        "            normalized[col] = (col_max - data[col]) / rng\n",
        "\n",
        "    # Draw grid and labels\n",
        "    ax.set_theta_offset(np.pi / 2)\n",
        "    ax.set_theta_direction(-1)\n",
        "    ax.set_xticks(angles[:-1])\n",
        "    ax.set_xticklabels(labels, size=9)\n",
        "    ax.set_ylim(0, 1)\n",
        "    ax.set_yticks([0.25, 0.5, 0.75, 1.0])\n",
        "    ax.set_yticklabels([\"0.25\", \"0.5\", \"0.75\", \"1.0\"], size=7)\n",
        "    ax.grid(color=\"gray\", alpha=0.3)\n",
        "    ax.set_title(title, pad=15, fontsize=11, fontweight=\"bold\")\n",
        "\n",
        "    # Plot each model\n",
        "    for (_, row), color in zip(data.iterrows(), colors):\n",
        "        vals = [normalized[col][row.name] for col, _, _ in metrics]\n",
        "        vals += vals[:1]\n",
        "        ax.plot(angles, vals, lw=2, color=color, label=row[\"model_label\"])\n",
        "        ax.fill(angles, vals, alpha=0.12, color=color)\n",
        "\n",
        "    ax.legend(loc=\"upper right\",\n",
        "              bbox_to_anchor=(1.35, 1.15), fontsize=9)\n",
        "\n",
        "\n",
        "# Preparar el summary con model_label\n",
        "summary_r = summary.copy()\n",
        "if \"model_label\" not in summary_r.columns:\n",
        "    summary_r[\"model_label\"] = summary_r[\"model_short\"].map(\n",
        "        lambda x: x[:10]\n",
        "    )\n",
        "\n",
        "colors_radar = [\"steelblue\", \"tomato\", \"seagreen\", \"goldenrod\", \"mediumpurple\"]\n",
        "\n",
        "# Vista (a): rendimiento\n",
        "perf_metrics = [\n",
        "    (\"tokens_per_s_mean\",     \"Throughput\\n(tok/s)\",    True),\n",
        "    (\"ttft_ms_mean\",          \"TTFT\",                   False),  # menor = mejor\n",
        "    (\"perplexity_geomean\",    \"Perplejidad\",            False),  # menor = mejor\n",
        "    (\"energy_per_token_j\",    \"Energía/token\",          False),  # menor = mejor\n",
        "]\n",
        "perf_metrics_available = [\n",
        "    (c, l, h) for c, l, h in perf_metrics\n",
        "    if c in summary_r.columns\n",
        "]\n",
        "\n",
        "# Vista (b): hardware\n",
        "hw_metrics_radar = [\n",
        "    (\"mbu_pct\",        \"MBU (%)\",          True),\n",
        "    (\"temp_max_c\",     \"Temperatura\\nmáx\", False),  # menor = mejor\n",
        "    (\"power_mean_w\",   \"Potencia\\nmedia\",  False),  # menor = mejor\n",
        "    (\"mem_pct_mean\",   \"Memoria (%)\",      False),  # menor = mejor (si existe)\n",
        "]\n",
        "hw_metrics_available = [\n",
        "    (c, l, h) for c, l, h in hw_metrics_radar\n",
        "    if c in summary_r.columns\n",
        "]\n",
        "\n",
        "fig = plt.figure(figsize=(16, 7))\n",
        "\n",
        "if perf_metrics_available:\n",
        "    ax_perf = fig.add_subplot(121, polar=True)\n",
        "    _radar_chart(summary_r, perf_metrics_available,\n",
        "                 \"(a) Vista de rendimiento\", ax_perf, colors_radar)\n",
        "\n",
        "if hw_metrics_available:\n",
        "    ax_hw = fig.add_subplot(122, polar=True)\n",
        "    _radar_chart(summary_r, hw_metrics_available,\n",
        "                 \"(b) Vista de hardware\", ax_hw, colors_radar)\n",
        "\n",
        "if not perf_metrics_available and not hw_metrics_available:\n",
        "    print(\"Sin métricas suficientes para el radar.\")\n",
        "else:\n",
        "    fig.suptitle(\n",
        "        \"Resumen global por modelo — gráficas de radar\\n\"\n",
        "        \"(exterior = mejor valor)\",\n",
        "        y=1.03, fontsize=13,\n",
        "    )\n",
        "    plt.tight_layout()\n",
        "    plt.show()\n"
    ])
]

# Aplicar cambios
add_cells_to_notebook(nb02_path, nb02_cells)
add_cells_to_notebook(nb04_path, nb04_cells)

print("\n✓ Todas las celdas añadidas correctamente")

