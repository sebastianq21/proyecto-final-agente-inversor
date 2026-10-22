# Ampliación: qué mejora al agente y qué no

Este documento extiende el trabajo del grupo (`docs/informe.md`). El agente final del grupo es DQN con recompensa R3 (penalización por rotación) y ruido 0,5 en las observaciones, entrenado 200k pasos. Aquí probamos una palanca a la vez, siempre con 5 semillas, para ver cuánto aporta cada una, cuál no aporta y cuál empeora.

## Cómo se midió

- **Una sola variable por experimento.** Cada variante cambia una cosa respecto de la anterior y deja todo lo demás igual: mismo entorno, mismo costo de 10 pb, mismas características y mismos hiperparámetros.
- **Validación (2016):** es donde se eligen las variantes. La métrica es el log del crecimiento del capital, promediado sobre las 5 semillas. Es la misma regla del informe.
- **Prueba (2017 a agosto de 2018):** el grupo ya la había mirado. Por eso aquí todas las variantes se reportan en prueba, pero ninguna se eligió con ella. Hay que leer estas cifras como un estudio de ablación, no como un resultado limpio de selección.
- **Ensamble:** una sola política que cada día toma la mediana de las acciones de las 5 semillas (0 %, 50 % o 100 % invertido). No requiere reentrenar.
- **Agente del grupo:** sus modelos no estaban versionados (`saves/` está en `.gitignore`), así que se reentrenaron con su config y sus semillas. La reproducción no salió idéntica: con las mismas semillas, las curvas se separan desde los 20k pasos, probablemente por diferencias de CPU e hilos entre máquinas. Por eso la tabla muestra la cifra publicada por el grupo y también la reproducida.

## Resultados

Retorno acumulado (media de 5 semillas; el ensamble es una sola corrida). Entre paréntesis va el log-crecimiento medio ± desviación entre semillas.

| Variante | Qué cambia | Val 2016 | Prueba 2017–18 | Sharpe prueba | Operaciones prueba |
|---|---|---|---|---|---|
| Grupo, publicado | — | +5,3 % (0,051 ± 0,057) | +39,4 % | 1,23 | 35,8 |
| Grupo, reproducido | — | +6,0 % (0,057 ± 0,053) | +30,7 % (0,25 ± 0,21) | 0,90 | 30,6 |
| DQN 2M | 10× más pasos | +10,0 % (0,093 ± 0,072) | +43,1 % (0,36 ± 0,08) | 1,25 | 17,8 |
| **DDQN 2M** | + Double DQN | **+12,1 % (0,114 ± 0,033)** | +48,0 % (0,39 ± 0,03) | 1,40 | 17,8 |
| DDQN 2M Sharpe dif. | recompensa Sharpe diferencial en vez de R3 | +1,6 % (0,013 ± 0,088) | +15,3 % (0,13 ± 0,17) | 0,63 | 101 |
| DDQN 2M multiactivo | entrena con GOOGL + 11 activos | +7,5 % (0,072 ± 0,032) | +52,7 % (0,42 ± 0,06) | 1,43 | 11 |
| Grupo, ensamble | ensamble de las 5 semillas | +16,3 % | +45,1 % | 1,48 | 17 |
| DQN 2M, ensamble | ensamble | +14,6 % | +51,8 % | 1,41 | 15 |
| DDQN 2M, ensamble | ensamble | +15,1 % | +51,7 % | 1,46 | 10 |
| DDQN 2M multiactivo, ensamble | ensamble | +6,2 % | +53,1 % | 1,38 | 7 |
| Comprar y mantener | — | +4,2 % | +56,4 % | 1,45 | 1 |
| 50 % fijo | — | +2,6 % | +26,0 % | 1,44 | 1 |
| Aleatorio (100 corridas) | — | −8,6 % | +6,8 % | 0,36 | 278 |

Por la regla de selección por semilla, gana **DDQN 2M**. Entre los ensambles, el mejor en validación es el del grupo (+16,3 %), seguido muy de cerca por DDQN 2M (+15,1 %).

Archivos:
- `results/ampliacion/final_val/` y `final_test/`: métricas por corrida, resumen, curvas de capital y posiciones.
- `results/ampliacion/curvas_2m.png`: curvas de aprendizaje.
- `results/ampliacion/datos/`: los activos adicionales.

## Qué aprendimos

**1. Entrenar más ayuda, y mucho.** Con 200k pasos el agente seguía lejos de converger. Con 2M, la validación sube de 0,057 a 0,093 (DQN), y la dispersión en prueba entre semillas baja de ±0,21 a ±0,08. Lo que en la curva del grupo parecía un techo a los 200k era el ruido de evaluar en un solo año.

**2. Double DQN aporta estabilidad más que retorno.** Frente a DQN 2M, sube un poco la media (0,093 → 0,114) y reduce a la mitad la dispersión entre semillas (0,072 → 0,033 en validación, 0,08 → 0,03 en prueba). Es coherente con su propósito: DQN sobreestima los valores Q al usar el máximo de la misma red que evalúa, y Double DQN separa quién elige la acción de quién la evalúa.

**3. El ensamble es la palanca más barata, pero solo ayuda cuando las semillas ya son buenas.** No requiere reentrenar, y mejora a las tres variantes con R3 entrenadas en GOOGL tanto en validación como en prueba. También reduce operaciones y costos, porque una semilla aislada que cambia de opinión no alcanza a mover la mediana. Con el ensamble, el agente del grupo pasa de +30,7 % a +45,1 % en prueba. Ese es el único caso de la tabla que supera a comprar y mantener en Sharpe (1,48 contra 1,45). En cambio, no rescata a la variante de Sharpe diferencial: empeora en validación (+1,6 % → −4,1 %) y en prueba. Tampoco ayuda a la multiactivo, cuyas semillas ya coinciden casi siempre: baja en validación (+7,5 % → +6,2 %) y queda igual en prueba. Promediar corrige la variabilidad entre semillas, no un sesgo que comparten todas.

**4. La recompensa de Sharpe diferencial empeora todo.** Es la variante que más memoriza: en entrenamiento llega a log 3,4, contra 2,8 de R3 (ver `curvas_2m.png`). Y es la que peor generaliza. Hace unas 100 operaciones en prueba, contra 18 de R3, y paga cuatro veces más en costos. La razón es que esta recompensa mide retorno por unidad de riesgo, pero no tiene ningún freno explícito a la rotación. La penalización por rotación de R3 resulta ser lo que contiene ese sobreajuste. Lección: cambiar la recompensa sin conservar el freno a operar es peor que no cambiarla.

**5. Más datos frenan la memorización, pero empujan al agente hacia comprar y mantener.** Con 11 activos adicionales, el agente deja de memorizar GOOGL: en entrenamiento queda en log 1,4, contra 2,8 del entrenado solo con GOOGL y 1,23 de comprar y mantener. Sus semillas son las más parecidas entre sí en validación (±0,032). Pero lo que aprende de 2009–2015, un mercado alcista para casi todos los activos, es a quedarse invertido: su exposición en prueba es de 0,94 (0,99 en el ensamble) y apenas opera. En prueba le va bien (+52,7 %) porque 2017–2018 también fue alcista. En validación, en cambio, queda por debajo de DDQN 2M. Lección: más datos reducen el sobreajuste, pero si todos vienen del mismo régimen de mercado le enseñan al agente una política casi pasiva.

**6. Ninguna variante supera a comprar y mantener en retorno durante la prueba.** Con posiciones entre 0 % y 100 %, la única forma de ganarle es salir antes de las caídas. En 2017–2018 GOOGL subió un 56 % con una caída máxima de 15 %, así que hubo poco que esquivar. En validación (2016, un año lateral) todas las variantes basadas en R3 sí le ganan.

## Limitaciones

- Validación es un solo año (2016). Una validación walk-forward, con varios años, haría más confiable la selección. Queda como trabajo futuro.
- Prueba ya había sido vista por el grupo antes de esta ampliación. Estas cifras sirven para comparar variantes entre sí, no como una estimación sin sesgo del desempeño futuro.
- La reproducción del agente del grupo no es exacta entre máquinas: con las mismas semillas se obtienen trayectorias distintas.
- Los datos de Yahoo están ajustados por dividendos; los de GOOGL vienen de gym-anytrading. Las características son retornos y razones de precios, así que no dependen de esa escala.

## Cómo reproducir

```bash
uv run python scripts/download_assets.py          # activos adicionales, solo hasta 2015
for s in 0 1 2 3 4; do
  uv run rlgames-sb train --config configs/googl_ddqn_turnover_noise05_2m.yaml --seed $s
done                                                # igual con las demás configs *_2m.yaml
uv run rlgames-sb backtest configs/googl_dqn_turnover_noise05.yaml configs/googl_dqn_turnover_noise05_2m.yaml \
  configs/googl_ddqn_turnover_noise05_2m.yaml configs/googl_ddqn_dsr_noise05_2m.yaml \
  configs/multi_ddqn_turnover_noise05_2m.yaml --split val --ensemble --out results/ampliacion/final_val
```

Cada corrida de 2M pasos tarda unos 40 minutos en CPU; 5 semillas en paralelo, cerca de una hora.
