# Guía de decisiones del proyecto «Agente Inversor»

Este documento explica **por qué** se tomó cada decisión del proyecto, qué alternativa se descartó y con qué referencia de la bibliografía del curso se puede defender. Está pensado para estudiar antes de la presentación: cada sección termina con una frase corta para responder al panel.

Hay dos versiones del experimento:

- **v1** (`configs/default.json` → `results/main`): la entrega original del grupo.
- **v2** (`configs/v2.json` → `results/v2`): agrega una tercera recompensa contra la sobreoperación y entrena con ventanas aleatorias. Todo lo de v1 sigue siendo reproducible.

---

## 0. El problema en una imagen

```text
 fin del día t                                   fin del día t+1
 ─────────────┬───────────────────────────────────────┬──────────────
  el agente   │  1. observa s_t (solo datos ≤ t)       │
  decide aquí │  2. elige a_t ∈ {0 %, 50 %, 100 %}     │
              │  3. se rebalancea al cierre t, paga 0,1 %
              │  4. el precio se mueve  p_t → p_{t+1}  │
              │  5. recibe r_{t+1} y observa s_{t+1} ──┘
```

Es la secuencia $S_t \to A_t \to R_{t+1}, S_{t+1}$ del capítulo 3 de Sutton & Barto: un **proceso de decisión de Markov** (MDP). El agente no controla el precio, que es exógeno; solo controla **cuánto capital expone**. Por eso la posición actual forma parte del estado: cambiarla cuesta dinero, y es lo que vuelve el problema realmente secuencial. Sin costos, cada día sería una decisión independiente.

> **Al panel:** «Formulamos la inversión como un MDP donde la acción es la fracción invertida, el precio es exógeno y la posición actual entra al estado porque cambiarla tiene un costo».

---

## 1. Datos

| Decisión | Por qué | Alternativa descartada |
|---|---|---|
| BTC/USDT diario de Binance, 2020–2024 (1.827 velas) | Datos públicos, sin huecos de fin de semana y con regímenes muy distintos: subida 2020–21, caída 2022, recuperación 2023–24. Eso hace visible la no estacionariedad. | Intradía: muchas más filas, pero más ruido y más impacto relativo de los costos, lo que complica el problema sin aportar a los objetivos del curso. |
| CSV guardado en el repo con hash SHA-256 | Cualquiera reproduce sin internet y se puede comprobar que es exactamente el mismo archivo. | Descargar en cada ejecución: los datos de la API pueden cambiar. |

**Pendiente:** el enunciado recomienda el dataset común del profesor. Si se publicó, conviene correr `configs/v2.json` también sobre él cambiando solo `"data"`.

---

## 2. División temporal: entrenamiento / validación / prueba (60/20/20)

| Partición | Fechas | Uso |
|---|---|---|
| Entrenamiento | 2020-01-21 → 2023-01-08 | El agente aprende. El escalador se ajusta aquí. |
| Validación | 2023-01-09 → 2024-01-04 | Se elige algoritmo y recompensa. |
| Prueba *held-out* | 2024-01-05 → 2024-12-31 | Se mira **una sola vez**, al final. |

**¿Por qué tres bloques si el enunciado pide dos?** Porque el enunciado prohíbe usar el held-out para elegir hiperparámetros o la mejor estrategia. Si solo hubiera train y test, la única forma de comparar recompensas sería mirando el test, y eso es exactamente la fuga que penaliza la rúbrica (criterio 4, nivel «deficiente»). La validación es el lugar «legal» para comparar.

**¿Por qué cronológico y sin barajar?** En series de tiempo, barajar mete el futuro en el entrenamiento: el agente aprendería de 2024 para decidir en 2022.

> **Al panel:** «Usamos validación para todas las decisiones y abrimos la prueba una sola vez, con la selección ya congelada en `selection_frozen.json`».

---

## 3. Estado: qué observa el agente (7 variables)

| Variable | Qué mide | Por qué sirve para decidir |
|---|---|---|
| `return_1` | Retorno de ayer | El movimiento más reciente; captura choques. |
| `momentum_5` | Retorno de 5 días | Tendencia corta. |
| `momentum_20` | Retorno de 20 días | Tendencia de ~1 mes; en cripto las tendencias persisten algo. |
| `gap_sma20` | Distancia del precio a su media de 20 días | Indica si el precio está «estirado» respecto a su nivel reciente. |
| `volatility_20` | Desviación de retornos, 20 días | El EDA muestra que la volatilidad cambia mucho; el riesgo de estar invertido no es constante. |
| `current_weight` | Fracción invertida ahora | Sin esto el agente no sabe si una acción implica operar (y pagar) o no. |
| `drawdown` | Caída del portafolio desde su máximo | Información de riesgo propia del inversionista, que el precio no da. |

**Decisiones de diseño:**

- **Indicadores relativos, no precios.** El precio de BTC pasó de ~7.000 a ~100.000. Una red que ve precios absolutos en validación ve valores que nunca vio en entrenamiento. Los retornos y ratios están en la misma escala en cualquier año.
- **Estandarizar con media y desviación solo de entrenamiento.** Si se usara todo el historial, la media «sabría» cómo es 2024. Esto lo verifica la prueba `test_test_data_do_not_affect_training_scaler`.
- **Recortar a [-10, 10].** Las colas de cripto son pesadas: un día extremo podría producir un valor gigante que desestabilice la red.
- **Ventanas causales.** Todas las medias móviles usan solo datos ≤ t. Lo verifica `test_future_prices_do_not_change_past_features`.
- **El volumen se analiza pero no entra al estado.** Mantener el estado pequeño reduce el riesgo de sobreajuste con solo ~1.000 días de entrenamiento.
- **`momentum_20` y `gap_sma20` están correlacionados** (lo muestra el EDA). Se dejaron ambos porque la red puede manejar redundancia; en Q-learning tabular se descartó `gap_sma20` para no multiplicar celdas.

**Limitación honesta:** el estado no es realmente markoviano, porque el precio depende de factores externos que el agente no ve. Técnicamente es un POMDP que se trata como MDP.

> **Al panel:** «El estado tiene solo información disponible al cierre de t, en unidades relativas, escalada con estadísticas de entrenamiento, y lo verificamos con pruebas automáticas».

---

## 4. Acciones: 0 %, 50 %, 100 %

**Por qué discreto:** DQN produce un valor Q por acción, así que necesita un conjunto finito de acciones (Mnih et al., 2013). Además hace la política interpretable y permite comparar con Q-learning tabular.

**Por qué tres niveles y no cinco:** cada nivel extra multiplica las combinaciones estado-acción que el agente debe aprender con los mismos ~1.000 días. Tres niveles ya distinguen «fuera», «a medias» y «todo dentro».

**Qué NO puede hacer el agente** (el enunciado lo pregunta explícitamente):

- vender en corto (ganar con caídas);
- apalancarse (>100 %);
- elegir 25 % o 75 %;
- usar órdenes límite o elegir la hora de ejecución.

**Detalle sutil:** la acción es un **peso objetivo**, no «comprar» o «vender». Si el agente elige 50 % y el precio sube, el peso real queda en 52 %, y repetir 50 % implica vender un poco. Medimos que esos rebalanceos cuestan solo ~50 USDT al año, así que no son el problema de costos.

---

## 5. Cuándo se ejecuta la orden (lookahead)

- El agente decide con el cierre de t y **se ejecuta al mismo cierre t**. El resultado se calcula con el movimiento t → t+1.
- Esto **no es fuga de información**, porque p_{t+1} nunca entra a la decisión.
- Sí es un **supuesto optimista**: en la realidad no se puede operar exactamente al precio que se acaba de observar. Lo más realista sería ejecutar en la apertura de t+1 con spread. Queda documentado como limitación.

> **Al panel:** «Distinguimos lookahead, que no hay, de ejecución idealizada, que sí asumimos y declaramos».

---

## 6. Costos de transacción

- **0,10 % del valor negociado** (10 bps, el ejemplo del enunciado). Es también la comisión que usa el capítulo 10 de Lapan.
- El simulador resuelve $X = w\,[V - c\,|X-B|]$. En palabras: si se quiere 100 % invertido, la comisión sale del mismo capital y no de un préstamo. Así nunca queda efectivo negativo (prueba `test_full_investment_cannot_borrow_commission`).
- El costo ya está dentro de $V_{t+1}$, así que **toda recompensa basada en el patrimonio lo incluye automáticamente**.

---

## 7. Recompensa: la iteración (criterio 3 de la rúbrica)

El enunciado pide el ciclo: recompensa → entrenar → observar → identificar el problema → modificar → reentrenar → comparar. Así quedó:

### R1 — `log`: $r = 100\cdot\log(V_{t+1}/V_t)$

- **Por qué log y no retorno simple:** la suma de log-retornos es el log del crecimiento total del capital. Maximizar la suma de recompensas (lo que hace RL, con γ≈1) equivale a maximizar la riqueza final compuesta. Con retornos simples eso no se cumple: +50 % y luego −50 % suman 0, pero el capital quedó en 75 %.
- **Por qué ×100:** solo cambia la escala para que la red no trabaje con números diminutos (~0,01). No cambia qué política es mejor.
- **Problema observado:** no castiga el riesgo más allá de la pérdida misma.

### R2 — `risk`: $r = 100\,[\log(V_{t+1}/V_t) - \lambda\,\min(g,0)^2]$, con λ = 10

- **Hipótesis:** castigar más las pérdidas grandes que las pequeñas, de forma cuadrática, para reducir drawdowns.
- **Lo que pasó:** casi nada. Con una caída de −1 %, la penalización es 10·0,01² = 0,001, apenas un 10 % extra sobre la pérdida. **λ = 10 resultó demasiado débil** y R2 se comportó casi igual que R1.
- **Lo más importante:** al revisar la validación de v1, el problema real no eran las pérdidas, sino que **el DQN operaba demasiado**: 159 operaciones en promedio en el año de validación, con una semilla en 277, es decir, un cambio cada 1–2 días.

### R3 — `turnover` (nueva en v2): $r = 100\,[\log(V_{t+1}/V_t) - \kappa\cdot\text{fracción negociada}]$, con κ = 0,005

- **Ataca directamente el problema observado:** cada cambio de posición «duele» más que su costo real, así el agente solo cambia si espera un movimiento que lo justifique.
- **Por qué κ = 0,005:** se calibró **solo con entrenamiento** (celda nueva del `eda.ipynb`). Un cambio completo de ida y vuelta (0 → 100 → 0) cuesta 0,2 % real + 1,0 % de penalización = 1,2 %. El movimiento diario mediano en entrenamiento es 1,74 %, y el 61 % de los días supera ese umbral. Es decir, κ desalienta operar por ruido sin prohibir operar.
- **Riesgo de reward hacking que se vigiló:** si κ es grande, el agente aprende a quedarse en efectivo para no ser castigado. Por eso se reportan el tiempo en efectivo y la exposición media, y no solo la recompensa.

### Comparación en validación (v2, media de 3 semillas, DQN)

| Recompensa | Retorno | Drawdown máx. | Costos (USDT) | Operaciones | Puntaje |
|---|---:|---:|---:|---:|---:|
| R1 log | 52,2 % | −21,6 % | 7.372 | 124,7 | 0,414 |
| R2 risk | 65,2 % | −23,1 % | 7.610 | 102,7 | 0,536 |
| **R3 turnover** | **88,7 %** | **−17,0 %** | **3.989** | **65,7** | **0,802** |

R3 redujo las operaciones a la mitad, bajó los costos 46 %, y aun así obtuvo **más** retorno y **menos** drawdown. Por eso fue la seleccionada, usando validación.

> **Al panel:** «La segunda recompensa atacaba un problema que no era el principal. Al mirar el comportamiento en validación vimos sobreoperación y diseñamos la tercera contra eso. Esa es la iteración real».

---

## 8. Algoritmos

### Por qué dos algoritmos

El indicador de desempeño pide comparar «desde métodos clásicos hasta Deep RL». Se usan:

- **Q-learning tabular** (Sutton & Barto, §6.5): es la referencia clásica. Discretiza el estado en 3⁵ = 243 celdas. Sirve para mostrar el límite de las tablas: no generaliza entre celdas y con pocos datos quedan celdas sin visitar.
- **DQN** (Mnih et al., 2013; Stable-Baselines3): aproxima Q(s,a) con una red neuronal y usa el estado continuo completo.

### Por qué DQN y no PPO

- Las acciones son **discretas y pocas**, el caso natural de DQN (lo recomienda el enunciado).
- DQN es *off-policy*: reutiliza experiencias del *replay buffer*, lo que es valioso cuando solo hay ~1.000 días de datos. PPO (Schulman et al., 2017) es *on-policy* y descarta la experiencia tras cada actualización, así que necesita más interacción.
- PPO sería la opción natural con acciones **continuas** (cualquier peso entre 0 y 1). Queda como extensión.

### Qué DQN es exactamente

La documentación de SB3 aclara que su DQN es la versión **básica**, sin Double DQN, Dueling DQN ni Prioritized Replay. Sí incluye **red objetivo** y **replay buffer**:

- El *replay buffer* rompe la correlación entre días consecutivos. Mnih et al. (2013) lo justifican porque aprender de muestras consecutivas es ineficiente por su fuerte correlación.
- La *red objetivo* mantiene fijo el «blanco» de Bellman por un tiempo, para que la red no persiga un objetivo que se mueve con cada actualización.

**Extensión posible:** Double DQN (van Hasselt et al., 2015) corrige la sobreestimación de Q. Con recompensas ruidosas como las del mercado, esa sobreestimación es un riesgo real.

### Hiperparámetros del DQN

| Parámetro | Valor | Por qué |
|---|---:|---|
| Red | 2 capas × 64, ReLU | 7 entradas y 3 salidas: una red pequeña basta y sobreajusta menos. |
| `learning_rate` | 3e-4 | Valor estándar de Adam; pasos pequeños porque la recompensa es ruidosa. |
| `batch_size` | 64 | Promedia el ruido del gradiente sin costo alto. |
| `gamma` (γ) | 0,99 | Horizonte efectivo ~1/(1−γ) = 100 días: importan las consecuencias de semanas, no solo de mañana. |
| `buffer_size` | 20.000 | Guarda las últimas 20.000 transiciones: en v1, todo lo visto; en v2, ~55 ventanas distintas. Da diversidad de experiencias. |
| `learning_starts` | 500 | No entrenar con un buffer casi vacío. |
| `train_freq` / `gradient_steps` | 4 / 1 | Una actualización cada 4 pasos: menos correlación y menos sobreajuste. |
| `target_update_interval` | 500 | Cada cuánto se copia la red al objetivo. |
| ε | 1,0 → 0,05 en el 40 % inicial | Explorar mucho al principio y luego explotar lo aprendido. |
| `timesteps` | 12.000 (v1) / 30.000 (v2) | En v2 los episodios son ventanas de 360 días; 30.000 pasos ≈ 83 ventanas distintas. |
| Criterio de parada | Número fijo de pasos | Sin parada temprana para no introducir otra decisión basada en datos. |

---

## 9. Episodios: ventanas aleatorias (cambio de v2)

- **v1:** cada episodio recorre siempre todo el entrenamiento desde el 21 de enero de 2020. El agente ve ~11 veces **exactamente la misma historia**, así que puede memorizarla en vez de aprender reglas generales.
- **v2:** cada episodio empieza en un día aleatorio de entrenamiento y dura **360 días**, el mismo largo que validación y prueba. Es la idea de `random_ofs_on_reset` del capítulo 10 de Lapan. Validación y prueba siguen recorriéndose completas y de forma determinista (`test_evaluation_env_is_deterministic_full_interval`).

**Resultado honesto:** con R1 y R2, este cambio **empeoró** la validación (DQN log pasó de 88,1 % a 52,2 %). Con R3 funcionó bien. v2 cambió a la vez las ventanas y el número de pasos, así que no se puede atribuir el efecto a uno solo. Aislarlo requeriría un experimento adicional que cambie una cosa a la vez.

---

## 10. Semillas (7, 21, 42)

- Las semillas cambian la inicialización de la red y la exploración, **no el mercado**. Tres semillas miden qué tan estable es el **aprendizaje**, no qué tan buena es la estrategia en otros mercados.
- **Nunca se elige la mejor semilla.** Se elige la combinación algoritmo + recompensa por el **promedio** de las tres, y en prueba se reportan las tres.

---

## 11. Regla de selección

$$\text{puntaje} = \text{retorno}_{val} - 0{,}5\cdot|\text{drawdown}_{val}|$$

- Combina rendimiento y riesgo en un número. El 0,5 expresa cuánto drawdown se «perdona» por cada punto de retorno, y se fijó **antes** de ver resultados.
- Otro inversionista podría usar otro coeficiente. Es una preferencia declarada, no una verdad.

---

## 12. Baselines

| Baseline | Para qué sirve |
|---|---|
| Comprar y mantener | ¿El agente aporta algo frente a no hacer nada inteligente? Es el rival difícil. |
| Aleatorio (3 semillas) | Piso: ¿el agente es mejor que el azar? También muestra cuánto destruyen los costos al operar sin criterio. |
| Efectivo | Control: retorno 0, drawdown 0; detecta si operar destruyó capital. |

Todos usan las mismas fechas, el mismo capital y la misma comisión.

---

## 13. Resultados en prueba (2024)

| Estrategia | Retorno medio | Drawdown medio | Costos (USDT) | Operaciones | Retorno / \|DD\| |
|---|---:|---:|---:|---:|---:|
| Comprar y mantener | **111,8 %** | −26,2 % | 100 | 1 | **4,27** |
| **v2: DQN / turnover (seleccionado)** | 67,6 % | **−18,8 %** | 3.525 | 48,3 | 3,59 |
| v1: DQN / log (seleccionado) | 89,5 % | −26,1 % | 7.885 | 133,0 | 3,43 |
| Aleatorio | 11,9 % | −35,2 % | 17.417 | 283,3 | 0,34 |
| Efectivo | 0,0 % | 0,0 % | 0 | 0 | — |

Por semilla, el DQN/turnover obtuvo 18,5 %, 112,5 % y 71,7 %.

**Cómo leerlo:**

1. **Ninguna versión le gana a comprar y mantener en retorno.** 2024 fue un año muy alcista para BTC, y en un año así casi cualquier salida del mercado cuesta retorno.
2. **v2 cambió el perfil de riesgo:** 7,3 puntos menos de drawdown que comprar y mantener, 55 % menos costos y 64 % menos operaciones que v1. A cambio, obtuvo menos retorno.
3. **En retorno por unidad de drawdown, v2 mejora a v1** (3,59 frente a 3,43), pero sigue por debajo de comprar y mantener (4,27).
4. **La dispersión entre semillas sigue siendo enorme** (18 % a 112 %). La semilla 21 casi replica comprar y mantener. Esta es la limitación más importante.
5. **Q-learning fracasa fuera de muestra** con todas las recompensas: es positivo en validación y negativo en prueba. Es un ejemplo claro de falta de generalización de la versión tabular.

**Respuesta a la pregunta central del proyecto:** con esta formulación y estos datos, **no**. El agente aprendió políticas no triviales, claramente mejores que el azar y con menos riesgo, pero no superó de forma robusta a la estrategia simple en datos no vistos. El enunciado dice expresamente que un resultado negativo bien evaluado puede obtener la nota máxima.

---

## 14. Comportamiento inesperado (el enunciado pide al menos uno)

- **Semilla 7 de v2 «demasiado prudente»:** pasó el **73 %** de 2024 en efectivo (ver `results/v2/positions.png`) y obtuvo 18,5 % mientras el mercado subía 112 %. Es el riesgo anticipado en la sección 7: la penalización por operar puede empujar a quedarse fuera. La recompensa no se «hackeó» contablemente, pero el comportamiento resultante es demasiado conservador.
- **La misma configuración produce agentes opuestos:** la semilla 21 estuvo invertida el 64 % del tiempo y la semilla 7 solo el 27 %. No logramos explicar del todo qué lo causa; responderlo requiere variar una fuente de aleatoriedad a la vez.
- **El cambio a ventanas aleatorias empeoró R1 y R2** (sección 9).

---

## 15. Limitaciones y honestidad metodológica

- **⚠ El periodo de prueba de v2 no es estrictamente «no visto».** El grupo ya había visto los resultados de 2024 en v1. Las decisiones de v2 se justificaron con validación (la sobreoperación aparece ahí) y con entrenamiento (κ), pero quien diseñó v2 conocía el resultado de prueba de v1. **Esto debe declararse.** La solución rigurosa sería un nuevo periodo final, por ejemplo 2025, o una evaluación *walk-forward*.
- Un solo activo y una sola división temporal; el mercado es no estacionario.
- Ejecución al mismo cierre observado; sin spread, deslizamiento ni impuestos; el efectivo no gana intereses.
- κ y λ no se optimizaron con un barrido; se fijaron con un argumento.
- v2 corrió en Linux con Python 3.13. En otro computador, **reentrenar** puede dar números algo distintos (DQN es sensible a detalles numéricos). **Reevaluar** los modelos guardados debe dar los mismos resultados.

---

## 16. Preguntas probables del panel

**¿Por qué no usaron precios directamente en el estado?** Porque cambian de escala entre años; la red vería en prueba valores fuera de lo aprendido. Los indicadores relativos son comparables en cualquier periodo.

**¿Cómo saben que no hay fuga de información?** Por tres mecanismos: (1) división cronológica sin barajar, (2) escalador ajustado solo en entrenamiento, y (3) dos pruebas automáticas que modifican el futuro y verifican que el pasado no cambia.

**¿Por qué el agente no le gana a comprar y mantener?** En un año muy alcista, la mejor estrategia es estar siempre dentro; cualquier salida cuesta retorno y cualquier operación cuesta comisión. El agente redujo el riesgo, pero no anticipó las subidas.

**¿Qué es reward hacking aquí?** Que el agente maximice la recompensa sin hacer lo que queremos. Con R3, el riesgo es quedarse en efectivo para no ser penalizado; la semilla 7 se acerca a eso.

**¿Por qué tres semillas son suficientes?** No lo son para afirmar nada estadísticamente. Sirven para mostrar que el aprendizaje es inestable, que es en sí un hallazgo.

**¿Qué harían diferente?** Evaluación *walk-forward*, ejecución en la apertura siguiente, Double DQN, más semillas, un barrido corto de κ en validación, y cambiar una cosa a la vez.

---

## 17. Mapa contra la rúbrica

| Criterio | Dónde está la evidencia |
|---|---|
| 1. Formulación | Secciones 0, 3 y 4 de esta guía; README §1 |
| 2. Entorno | `src/environment.py`, 15 pruebas en `tests/`, secciones 5 y 6 |
| 3. Recompensa | Sección 7: tres versiones, iteración motivada por el comportamiento observado |
| 4. Held-out y baselines | `selection_frozen.json`, sección 13, `results/*/test_metrics.csv` |
| 5. Discusión y reproducibilidad | Secciones 14–15, `REFLEXION_INDIVIDUAL.md`, README §9 |

## Referencias

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2.ª ed.). Cap. 3 (MDP), §6.5 (Q-learning), §16.5 (DQN en Atari), §17.4 (diseño de la señal de recompensa).
- Mnih, V., et al. (2013). *Playing Atari with Deep Reinforcement Learning*. arXiv:1312.5602.
- van Hasselt, H., Guez, A., & Silver, D. (2015). *Deep Reinforcement Learning with Double Q-learning*. arXiv:1509.06461.
- Wang, Z., et al. (2016). *Dueling Network Architectures for Deep RL*. arXiv:1511.06581.
- Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv:1707.06347.
- Lapan, M. (2024). *Deep Reinforcement Learning Hands-On* (3.ª ed.), cap. 10 (trading con DQN).
- Powell, W. B. (2019). *Reinforcement Learning and Stochastic Optimization*. Wiley.
- Stable-Baselines3, documentación de DQN. Gymnasium, API `Env`.
