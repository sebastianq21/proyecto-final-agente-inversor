# Bitácora de decisiones

Registro de las decisiones metodológicas, en orden. Cada entrada: qué se decidió, por qué y qué
alternativa se descartó.

## 2026-10-19 — Punto de partida

**Base de código.** Copia propia (clon, sin modificar el original) de
[`rl_games_sb`](https://github.com/emiliomunozai/rl_games_sb), el repositorio del curso. Se conservan su CLI, MLflow y Optuna; se agregan `data.py`, `invest_env.py`,
`baselines.py`, `evaluate.py` y el comando `backtest`.

**Dataset.** `STOCKS_GOOGL` incluido en `gym-anytrading` (el "dataset común" del curso): GOOGL diario,
2009-05-22 → 2018-08-29, 2.335 días. Copia en `data/STOCKS_GOOGL.csv`
(sha256 `050162e7…cec38a5b`).

**Por qué no usar `stocks-v0` tal cual.** Revisión de su código fuente:

| Aspecto | `stocks-v0` | Problema |
|---|---|---|
| Acciones | 2: vender/comprar (todo o nada) | Sin exposición intermedia |
| Observación | Ventana de 30 precios crudos y su diferencia | Escala ×6 en el período; no estacionaria |
| Recompensa | Ganancia en USD solo al cerrar una posición larga | Escasa, en escala absoluta y **sin costos** |
| Costos | 1 % venta + 0,5 % compra, solo en `total_profit` | Desconectados de lo que el agente optimiza |
| Partición | `frame_bound` = todo el dataset | Sin train/test |
| Valor | Se actualiza solo al operar | Sin curva de capital diaria ni drawdown |

Se construye `InvestEnv` con el mismo dataset, y la recompensa de `stocks-v0` se reproduce como
variante `sparse` (R0) para medir su efecto bajo las mismas condiciones.

**Partición temporal.** train 2009-05 → 2015-12 (1.616 días con indicadores completos), val 2016
(252), test 2017-01 → 2018-08 (418). Val elige la recompensa y los hiperparámetros; test se
evalúa **una sola vez** con el agente congelado (`backtest --split test --final`).

**Momento de la decisión.** Al cierre de t, con datos hasta t; se rebalancea al cierre de t
pagando 10 bps sobre el monto operado; el resultado es el movimiento de cierre t → cierre t+1.
Supuesto: se puede operar al precio de cierre observado (orden "market-on-close"). Alternativa
más conservadora descartada por simplicidad: ejecutar a la apertura de t+1.

**Estado (9 variables).** `ret_1`, `ret_5`, `ret_20`, `gap_sma20`, `trend_20_50`, `vol_20`,
`rsi_14`, `volume_z` (z-score con media/std de train) + fracción invertida actual. Justificación
en `notebooks/eda.ipynb`.

**Acciones.** {0 %, 50 %, 100 %}, rebalanceo diario al objetivo (el 50 % deriva con el precio y se
reajusta, pagando costo sobre lo operado). Sin cortos ni apalancamiento.

**Episodios.** Train: ventanas aleatorias de 252 días (diversidad de regímenes, más episodios).
Val/test: el período completo, determinista.

**Cambios mínimos al repositorio base.**
- `save_path(..., seed)`: una carpeta de guardado por semilla (antes se sobrescribían).
- `eval_env` en los configs: `train` y `tune` (Optuna) evalúan en `googl-val-v0`. Sin esto, Optuna
  puntuaría sobre el mismo período de entrenamiento.
- `train` retoma un modelo guardado si existe: `scripts/train_rewards.sh` borra antes de entrenar.

**Evaluación común.** Todas las estrategias corren en `googl-<split>-v0` (recompensa = retorno log,
cuya suma = log del crecimiento del capital), con los mismos costos y capital inicial.

## 2026-10-19 — Iteración 1 de la recompensa (R0, R1, R2)

DQN, 200k pasos, 5 semillas. Resultados en val 2016 (media de semillas; `results/val_metrics_iter1.csv`):

| Estrategia | Retorno | Máx. DD | Costos | Operaciones | Exposición |
|---|---|---|---|---|---|
| R0 `sparse` | +2,9 % | −11,6 % | $6.775 | 92 | 73 % |
| R1 `log` | −8,4 % | −14,1 % | $8.588 | 125 | 40 % |
| R2 `dd` | −5,3 % | −10,6 % | $7.170 | 94 | 28 % |
| Comprar y mantener | +4,2 % | −13,5 % | $100 | 1 | 100 % |
| 50 % fijo | +2,6 % | −6,9 % | $106 | 1 | 50 % |
| Aleatorio (100) | −8,6 % | −14,5 % | $10.646 | 168 | 50 % |

**Problema 1, exceso de operaciones.** 90-125 cambios de posición al año; 6-9 % del capital en costos.

**Hipótesis descartada: indecisión.** Si los Q de las acciones estuvieran casi empatados, el agente
cambiaría por ruido. Medido en val: la brecha mediana entre la mejor y la segunda acción es 0,0065
(R1), 6,5 veces el costo de un cambio completo. La red no está indecisa.

**Problema 2, sobreajuste (confirmado).** En train (2009-2015) R1 gana ×43 (Sharpe 3,5, MDD −7,7 %)
y R2 ×59 (Sharpe 3,9, MDD −4,9 %), frente a ×3,4 de comprar y mantener. Con 200k pasos sobre 1.616
días, cada día se ve ~125 veces: la red memoriza el período. R2 cumple su objetivo en train (caídas
mínimas) solo memorizando: caso de reward hacking. R0 sobreajusta menos (señal escasa) y aprende
algo cercano a "estar invertido casi siempre".

## 2026-10-19 — Iteración 2: decisión

1. **R3 `turnover`** = R1 − κ·|Δposición|, κ = 0,005 (5 veces el costo real), solo en la recompensa
   de entrenamiento; el costo del entorno sigue en 10 bps. Ataca el problema 1: un cambio de
   posición debe prometer más de lo que cuesta.
2. **Curvas de aprendizaje** (`curve:` en los configs): cada 10k pasos se mide el agente greedy en
   todo train y en val. Ataca el problema 2: muestra cuándo empieza la memorización y permite elegir
   el número de pasos con val (nunca con test).
3. Se reentrenan R0-R2 con las mismas semillas: deben dar resultados idénticos a la iteración 1
   (prueba de reproducibilidad).

## 2026-10-20 — Iteración 2: resultados

**Reproducibilidad.** R0, R1 y R2 reentrenados con las mismas semillas dieron métricas idénticas a
la iteración 1 (diferencia máxima 0,0).

**R3 en val (200k pasos, media de 5 semillas):** −6,8 %, máx. DD −12,4 %, costos USD 5.895 (−31 %
frente a R1), 92 operaciones. Mejor que R1, pero sigue operando en exceso y por debajo de comprar y
mantener.

**Curvas de aprendizaje** (`results/curves/curves.png`):
- R1, R2 y R3 superan a comprar y mantener **en train** desde 20-30k pasos y llegan a ×43-55: la
  memorización empieza muy temprano.
- En val ningún punto es claramente mejor; la dispersión entre semillas (0,04-0,10 en log del
  crecimiento) es del orden de las diferencias entre puntos. Elegir el "mejor paso" sería elegir
  ruido: **el early stopping no resuelve el problema**.
- R3 queda por encima de R1 en val en casi todos los puntos.
- R0 casi no memoriza (×1,5 en train) y termina en +2,6 % en val, con alta dispersión.

**Regla de selección del agente final (fijada antes de mirar test):** la configuración con mayor
log del crecimiento medio en val (5 semillas); en empate, la que opere menos.

## 2026-10-20 — Documentación

- Se agrega la Figura 1 del dataset (`results/eda/googl_contexto.png`) y el estado del arte con
  referencias verificadas. Hallazgo: los 8 mayores movimientos diarios son días de resultados
  trimestrales (verificado en prensa para los 3 mayores).
- Estructura: README corto como puerta de entrada; `docs/informe.md` es el entregable (secciones
  a-e); esta bitácora registra el proceso.

## 2026-10-20 — Iteración 3: R3 con ruido en las observaciones

**Decisión.** La iteración 2 mostró que detener antes no evita la memorización. Se agrega ruido
gaussiano a los 8 indicadores **solo durante el entrenamiento** (la posición, los precios y los
entornos de evaluación quedan intactos): el mismo día nunca se ve igual dos veces. Dos intensidades,
σ = 0,5 y σ = 1,0 (unidades de z-score), sobre R3, con los mismos hiperparámetros, 200k pasos y 5
semillas.

**Resultados.**

| Variante | Train (×capital, 5 semillas) | Val: retorno | Máx. DD | Costos | Operaciones | log-crec. val (media ± std) |
|---|---|---|---|---|---|---|
| R3 sin ruido | ×22 a ×65 | −6,8 % | −12,4 % | USD 5.895 | 92 | −0,073 ± 0,070 |
| R3 + ruido 0,5 | ×7,3 a ×9,5 | +5,3 % | −9,8 % | USD 1.246 | 20 | **0,051** ± 0,057 |
| R3 + ruido 1,0 | ×1,7 a ×5,1 | +5,0 % | −10,5 % | USD 385 | 7 | 0,046 ± 0,073 |
| Comprar y mantener | ×3,4 | +4,2 % | −13,5 % | USD 100 | 1 | 0,042 |

- El ruido reduce fuertemente la memorización: con σ = 1,0 el desempeño en train queda en torno al
  de comprar y mantener.
- Primera vez que los agentes igualan o superan a comprar y mantener en val, con menor caída
  máxima y entre 5 y 18 veces menos operaciones que en la iteración 2.
- La ventaja sobre comprar y mantener (~1 punto) es mucho menor que la dispersión entre semillas
  (±6-7 puntos): **comparable, no mejor** de forma distinguible.
- Elegir el mejor de 6 configuraciones en val infla su resultado en val (sesgo de selección); el
  test lo corrige.

**Agente final según la regla fijada:** R3 + ruido σ = 0,5 (mayor log-crecimiento medio en val).
La diferencia con σ = 1,0 está dentro del ruido; se mantiene la regla escrita antes de ver los
resultados para no ajustar a posteriori.

## 2026-10-20 — Evaluación final en test (ejecutada una sola vez, 11:52)

Agente congelado: R3 + ruido σ = 0,5, 5 semillas. `backtest --split test --final`.

| Estrategia | Retorno | Máx. DD | Costos | Operaciones | Exposición | Sharpe |
|---|---|---|---|---|---|---|
| Agente final (media de 5) | +39,4 % | −13,9 % | USD 2.511 | 36 | 73 % | 1,23 |
| Comprar y mantener | +56,4 % | −15,4 % | USD 100 | 1 | 100 % | 1,45 |
| 50 % fijo | +26,0 % | −7,8 % | USD 156 | 1 | 50 % | 1,44 |
| Aleatorio (100) | +6,8 % | −13,4 % | USD 19.078 | 278 | 50 % | 0,36 |

- No supera a comprar y mantener; ninguna semilla lo hace (la mejor, +55,7 %).
- Semilla 0 atípica: 37 % de exposición, +3,4 %.
- Salidas sincronizadas el 30-oct-2017 (4 semillas, día siguiente a resultados del 3T-2017) y el
  1-jun-2018 (5 semillas); no anticiparon caídas.
- La ventaja en val (+1,1 puntos) no se sostuvo: era ruido entre semillas + sesgo de selección.

## Pendientes

- [x] Comparar R0/R1/R2 en val e iterar.
- [x] Iteración 2: R3 y curvas de aprendizaje.
- [x] Iteración 3: ruido en las observaciones.
- [x] Congelar agente y correr test una sola vez.
- [x] Documentar anomalías del agente.
- [x] Reflexión (grupal, por indicación del profesor) en `informe.md`, sección e.6.
- ~~Optuna corto sobre la recompensa elegida~~ — descartado: se mantuvieron los mismos
  hiperparámetros en todas las variantes para que la comparación de recompensas fuera justa, y con
  252 días de validación y la dispersión observada entre semillas, el ajuste fino habría elegido
  ruido.
