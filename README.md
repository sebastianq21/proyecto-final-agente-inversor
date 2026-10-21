# Agente Inversor — DQN sobre GOOGL

Proyecto final de *Simulación y Aprendizaje por Refuerzo* (Universidad de La Sabana, 2026-2).

**Pregunta:** ¿puede un agente de RL aprender una estrategia de inversión que funcione mejor que
estrategias simples (comprar y mantener, o decidir al azar) con datos que no vio en el
entrenamiento?

El proyecto parte de [`rl_games_sb`](https://github.com/emiliomunozai/rl_games_sb), el repositorio
base del curso (Stable-Baselines3 + Gymnasium + MLflow + Optuna). Su README original está en
[`docs/README_base.md`](docs/README_base.md).

![GOOGL 2009-2018: precio, drawdown y volatilidad](results/eda/googl_contexto.png)

Entre 2009 y 2018 el precio de GOOGL se multiplicó por 6,4, pero en el camino tuvo seis caídas de
más de 15 %, épocas de volatilidad entre 6 % y 60 % anual y saltos de hasta +16 % en días de
resultados trimestrales, que no se pueden anticipar mirando solo el precio. Cada día el agente
decide qué parte del capital tener invertida (0 %, 50 % o 100 %) y paga 10 bps (0,10 %) cada vez
que opera.

**Resultado:** en el período de prueba (2017-2018), que evaluamos una sola vez, el agente final
(DQN con penalización por operar y ruido en lo que ve durante el entrenamiento) gana **+39,4 %**,
mientras que comprar y mantener gana **+56,4 %**. Su caída máxima es algo menor (−13,9 % frente a
−15,4 %). Le gana al agente aleatorio (+6,8 %) y a mantener siempre el 50 % invertido (+26,0 %),
pero **no a comprar y mantener**. Las primeras versiones del agente se aprendían de memoria el
período de entrenamiento; el ruido ayudó a reducir ese problema. Todo el detalle está en el
informe.

## Documentación

| Documento | Para qué |
|---|---|
| [`docs/informe.md`](docs/informe.md) | **Informe del proyecto**: planteamiento, entorno, recompensa, evaluación, resultados y discusión |
| [`docs/decisiones.md`](docs/decisiones.md) | Bitácora: las decisiones, hipótesis y resultados de cada iteración, en orden |
| [`notebooks/eda.ipynb`](notebooks/eda.ipynb) | Análisis exploratorio de los datos |
| [`docs/README_base.md`](docs/README_base.md) | README original del repositorio del curso |

## Código

| Pieza | Dónde |
|---|---|
| Datos e indicadores (sin usar datos del futuro; normalización solo con train) | `src/rl_games_sb/data.py` |
| Entorno de inversión (Gymnasium) | `src/rl_games_sb/invest_env.py` |
| Agente (DQN de SB3) y su construcción | `src/rl_games_sb/registry.py`, `configs/googl_dqn_*.yaml` |
| Entrenamiento (+ MLflow) y ajuste (Optuna) | `src/rl_games_sb/experiments.py` |
| Baselines | `src/rl_games_sb/baselines.py` |
| Evaluación / backtest | `src/rl_games_sb/evaluate.py` (`rlgames-sb backtest`) |
| Análisis exploratorio (generado por `scripts/make_eda_notebook.py`) | `notebooks/eda.ipynb` |
| Pruebas (datos del futuro, costos, tiempos, recompensas) | `tests/test_invest_env.py` |
| Resultados (CSV y figuras) | `results/<train\|val\|test>/`, `results/curves/`, `results/eda/` |

## Dataset

`STOCKS_GOOGL`, que viene en [`gym-anytrading`](https://github.com/AminHP/gym-anytrading): precios
diarios de GOOGL (apertura, máximo, mínimo, cierre y volumen) del 2009-05-22 al 2018-08-29, en total
2.335 días. Hay una copia en `data/STOCKS_GOOGL.csv`,
sha256 `050162e746a59f0c6e0a144978afb4adddb65d8b72dd331c774210cbdec38a5b`.

| Período | Fechas | Uso |
|---|---|---|
| train | 2009-05 → 2015-12 | El agente aprende (ventanas aleatorias de 1 año) |
| val | 2016 | Elegir recompensa e hiperparámetros |
| test (held-out) | 2017-01 → 2018-08 | Evaluación final, **una sola vez** |

## Instalación

Se necesita [uv](https://docs.astral.sh/uv/), que también instala Python 3.11.

```bash
uv sync
uv run pytest            # 9 pruebas: datos del futuro, normalización, costos, tiempos, recompensas, ruido
```

## Entornos

| Id | Período | Episodio | Recompensa |
|---|---|---|---|
| `googl-sparse-v0` | train | 252 días aleatorios | R0: ganancia en USD al vender, sin costos (como `stocks-v0`) |
| `googl-log-v0` | train | 252 días aleatorios | R1: retorno log diario neto de costos |
| `googl-dd-v0` | train | 252 días aleatorios | R2: R1 − aumento del drawdown |
| `googl-turnover-v0` | train | 252 días aleatorios | R3: R1 − 0,005 × fracción operada |
| `googl-turnover-noise05-v0`, `-noise10-v0` | train | 252 días aleatorios | R3 con ruido gaussiano (σ = 0,5 / 1,0) en los indicadores |
| `googl-val-v0` / `googl-test-v0` | val / test | período completo | retorno log (el mismo puntaje para todos) |

```bash
uv run rlgames-sb inspect --env googl-log-v0 --steps 3
```

- **Estado (9 valores):** `ret_1`, `ret_5`, `ret_20`, `gap_sma20`, `trend_20_50`, `vol_20`,
  `rsi_14`, `volume_z` y la fracción invertida.
- **Acciones:** 0 → 0 %, 1 → 50 %, 2 → 100 % invertido.
- **Costo:** 10 bps sobre el monto que se opera.
- **Momento de la decisión:** al cierre del día t; el resultado es lo que se mueve el precio entre
  el cierre de t y el de t+1.

## Reproducir los resultados

Las semillas 0-4 fijan la red, la exploración y las ventanas de entrenamiento, y el agente
aleatorio usa las semillas 0-99. Con las mismas semillas se obtienen exactamente los mismos
resultados.

```bash
# 1. Iteración de recompensas: 4 recompensas x 5 semillas, comparadas en val (~25 min en CPU)
bash scripts/train_rewards.sh
uv run rlgames-sb curves configs/googl_dqn_sparse.yaml configs/googl_dqn_log.yaml \
    configs/googl_dqn_dd.yaml configs/googl_dqn_turnover.yaml   # curvas de aprendizaje

# 2. Un solo entrenamiento (primero se borra el modelo guardado, porque train lo retoma si existe)
uv run rlgames-sb delete dqn --env googl-log-v0 --seed 0
uv run rlgames-sb train --config configs/googl_dqn_log.yaml --seed 0

# 3. Baselines solos (sin configs) o junto a agentes, en val
uv run rlgames-sb backtest --split val
uv run rlgames-sb backtest configs/googl_dqn_log.yaml --split val

# 4. Ajuste de hiperparámetros con Optuna (puntúa en val)
uv run rlgames-sb tune --config configs/googl_dqn_log.yaml

# 5. Iteración 3 y evaluación final en prueba (ya se hizo una vez; agente final: R3 + ruido 0,5)
bash scripts/train_rewards.sh turnover_noise05 turnover_noise10
uv run rlgames-sb backtest configs/googl_dqn_turnover_noise05.yaml --split test --final

# 6. Análisis exploratorio (reejecuta el notebook y regenera sus figuras)
uv run python scripts/make_eda_notebook.py

# Experimentos registrados
uv run mlflow ui        # http://127.0.0.1:5000, experimento "googl"
```

Cada `backtest` guarda en `results/<split>/`: `metrics.csv` (una fila por semilla o corrida),
`summary.csv` (promedio y desviación), `days_*.csv` (valor de cada día) y las figuras `equity.png` y
`positions_*.png`.
