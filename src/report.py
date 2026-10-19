"""Informe en español con valores calculados, sin resultados inventados."""
from pathlib import Path
import pandas as pd


def table(frame, columns):
    header = "| " + " | ".join(columns) + " |\n|" + "|".join(["---"]*len(columns)) + "|\n"
    return header + "\n".join("| " + " | ".join(str(row[c]) for c in columns) + " |" for _, row in frame.iterrows())


def write_report(out, manifest, cfg, validation, results, selected):
    from .data import prepare
    data, _, periods, _, _ = prepare(cfg["data"], cfg)
    training = data.iloc[:periods["train"][1]+1]
    low = training.loc[training.return_1.idxmin()]
    high_vol = training.loc[training.volatility_20.idxmax()]
    low_vol = training.loc[training.volatility_20.idxmin()]
    asset_drawdown = (training.close/training.close.cummax()-1).min()
    volume_sentence = (f"El volumen diario mediano fue {training.volume.median():,.2f} BTC. "
                       if 'volume' in training else "El CSV no contiene volumen. ")
    winner = results[(results.method == selected["method"]) & (results.reward == selected["reward"])]
    buy = results[results.method == "buy_hold"].iloc[0]
    cash = results[results.method == "cash"].iloc[0]
    rows = []
    labels = {"qlearning": "Q-learning", "dqn": "DQN", "buy_hold": "Comprar y mantener",
              "random": "Aleatorio", "cash": "Efectivo"}
    for (kind, reward), group in results.groupby(["method", "reward"]):
        rows.append({"Estrategia": labels[kind]+(" / "+reward if reward != "baseline" else ""),
                     "Retorno medio": f"{group['return'].mean():.2%}",
                     "DE retorno": f"{group['return'].std():.2%}" if len(group)>1 else "No aplica",
                     "Drawdown medio": f"{group.max_drawdown.mean():.2%}",
                     "Costos medios": f"{group.costs.mean():,.2f}",
                     "Operaciones medias": f"{group.trades.mean():.1f}",
                     "Réplicas": len(group)})
    rows = pd.DataFrame(rows)
    vr = []
    for (kind, reward), group in validation.groupby(["method", "reward"]):
        vr.append({"Método": labels[kind], "Recompensa": reward,
                   "Retorno": f"{group['return'].mean():.2%}",
                   "Drawdown": f"{group.max_drawdown.mean():.2%}",
                   "Costos": f"{group.costs.mean():,.2f}",
                   "Tiempo en efectivo": f"{group.cash_fraction.mean():.2%}",
                   "Puntaje": f"{group.score.mean():.6f}"})
    vr = pd.DataFrame(vr)
    splits = "\n".join(f"- {k}: {v['start'][:10]} a {v['end'][:10]}, {v['transitions']} transiciones."
                       for k, v in manifest["splits"].items())
    gain = winner['return'].mean()-buy['return']
    verdict = ("En esta ventana, el promedio no respalda una mejora del agente seleccionado sobre comprar y mantener."
               if gain <= 0 else "En esta ventana, el promedio de retorno fue mayor que comprar y mantener; esto no demuestra una ventaja en otros periodos.")
    q_log_test = results[(results.method == 'qlearning') & (results.reward == 'log')]
    q_log_val = validation[(validation.method == 'qlearning') & (validation.reward == 'log')]
    beats = int((winner['return'] > buy['return']).sum())
    obs = []
    for _, row in winner.iterrows():
        name = f"{row.method}_{row.reward}_{int(row.seed)}"
        curve = pd.read_csv(out/f"test_{name}.csv").iloc[1:]
        worst = curve.loc[curve.net_return.idxmin()]
        obs.append(f"Semilla {int(row.seed)}: {row.cash_fraction:.1%} de decisiones con exposición cero, "
                   f"exposición objetivo media {row.mean_target_weight:.1%}. Peor transición: "
                   f"{str(worst.decision_timestamp)[:10]} a {str(worst.timestamp)[:10]}, "
                   f"acción {int(worst.action)}, retorno neto {worst.net_return:.2%}.")
    log_v = validation[(validation.method == selected['method']) & (validation.reward == 'log')]
    risk_v = validation[(validation.method == selected['method']) & (validation.reward == 'risk')]
    turn_v = validation[(validation.method == selected['method']) & (validation.reward == 'turnover')]
    turnover_text = "" if turn_v.empty else (
        f"\n\nLa versión 3 (v2 del proyecto) es r3 = 100 {{log[V(t+1)/V(t)] - kappa × |X-B|/V}}, con kappa = "
        f"{cfg.get('turnover_kappa', 0.005)}. Penaliza la fracción del patrimonio negociada, porque en validación de la "
        f"primera ejecución el problema observado fue la sobreoperación y no las pérdidas diarias. Para "
        f"{labels[selected['method']]}, turnover presentó retorno {turn_v['return'].mean():.2%}, drawdown "
        f"{turn_v.max_drawdown.mean():.2%}, costos {turn_v.costs.mean():,.2f} y {turn_v.trades.mean():.1f} operaciones medias.")
    n_cand = validation.groupby(["method", "reward"]).ngroups
    window = cfg.get("episode_length")
    episode_text = (f"episodios de {window} pasos que empiezan en una fecha aleatoria del entrenamiento"
                    if window else "episodios completos")
    budget_q = f"{cfg['q_episodes']} × {window}" if window else f"{cfg['q_episodes']} × {manifest['splits']['train']['transitions']}"
    text = f"""# Aprendizaje por refuerzo para decidir la exposición a BTC/USDT

## Alcance y pregunta de investigación

¿Puede un agente aprender una política que supere a comprar y mantener fuera del entrenamiento, después de pagar costos y considerar las pérdidas? Se implementaron Q-learning tabular y DQN con acciones de 0 %, 50 % y 100 %. El capital inicial fue de 100.000 unidades de cotización, aquí USDT. No se realiza conversión a pesos colombianos.

El candidato elegido en validación fue {labels[selected['method']]} con recompensa {selected['reward']}. En prueba obtuvo un retorno medio de {winner['return'].mean():.2%}; comprar y mantener obtuvo {buy['return']:.2%}. La diferencia fue {gain*100:.2f} puntos porcentuales. El drawdown medio del agente fue {winner.max_drawdown.mean():.2%}, frente a {buy.max_drawdown:.2%} del baseline. Estos valores describen una ventana histórica; no prueban superioridad en otros periodos.

## Datos y separación temporal

Se utilizaron 1.827 velas diarias de BTC/USDT del mercado spot de Binance, del 1 de enero de 2020 al 31 de diciembre de 2024. La página de Gym Trading Env facilitada para el trabajo documenta la descarga de criptomonedas con CCXT. Este proyecto usa el endpoint público REST de Binance para conservar una instantánea CSV sin añadir ese paquete. La referencia orienta la fuente; no es un archivo de precios entregado por el profesor.

Cada timestamp identifica el comienzo de la vela UTC. El cierre y los indicadores de esa fila se consideran conocidos al terminar el día. Las fechas del informe son etiquetas de velas, no horas de disponibilidad intradía. El volumen corresponde a BTC. El CSV y sus metadatos están incluidos; repetir el experimento no requiere red.

Tras descartar las primeras 20 observaciones de calentamiento, la división cronológica es 60 % / 20 % / 20 %:

{splits}

Las transiciones entre particiones se excluyen: cada periodo inicia con efectivo y termina en su último precio. Una media móvil de validación o prueba puede usar precios anteriores ya conocidos. Ninguna transición de entrenamiento utiliza un precio de validación. La media y la desviación del escalamiento se ajustan exclusivamente en entrenamiento; después quedan fijas. No se barajan las filas.

SHA-256 del CSV: `{manifest['sha256']}`.

## Formulación del proceso de decisión

El problema se formula como una secuencia de decisiones en la que el agente observa información disponible hasta el instante $t$, elige una exposición objetivo y recibe una recompensa después del movimiento $t \\rightarrow t+1$. La observación DQN tiene siete componentes:

| Componente | Definición | Ventana / rango | Justificación |
|---|---|---|---|
| `return_1` | $p_t/p_{{t-1}}-1$ | 1 día | Resume el movimiento más reciente. |
| `momentum_5` | $p_t/p_{{t-5}}-1$ | 5 días | Captura tendencia de corto plazo. |
| `momentum_20` | $p_t/p_{{t-20}}-1$ | 20 días | Captura una tendencia de horizonte mayor. |
| `gap_sma20` | $p_t/SMA_{{20,t}}-1$ | 20 días | Mide separación frente al nivel reciente. |
| `volatility_20` | desviación estándar de `return_1` | 20 días | Aproxima el régimen de variabilidad. |
| `current_weight` | valor riesgoso / patrimonio | $[0,1]$ | Informa cuánta exposición ya posee el portafolio. |
| `drawdown` | patrimonio / máximo previo − 1 | $[-1,0]$ | Informa la pérdida respecto al máximo del patrimonio. |

Los cinco indicadores de mercado se estandarizan con media y desviación calculadas **exclusivamente en entrenamiento** y después se limitan al intervalo `[-10, 10]`. Las ventanas son causales: no se centran ni se rellenan con observaciones futuras. La observación es una representación parcial del mercado; no se demuestra la propiedad de Markov, pues regímenes y variables externas pueden afectar al precio siguiente.

El espacio de acciones es discreto e interpretable:

| Acción | Peso objetivo | Interpretación |
|---:|---:|---|
| 0 | 0 % | Todo el patrimonio en efectivo. |
| 1 | 50 % | Exposición intermedia a BTC. |
| 2 | 100 % | Exposición total a BTC. |

La granularidad 0/50/100 % mantiene el problema interpretable, permite usar DQN naturalmente y hace posible comparar con Q-learning tabular. Se admiten participaciones fraccionarias, pero no ventas en corto ni apalancamiento. Tampoco se modelan órdenes límite, intereses sobre efectivo o impacto de mercado.

## Entorno y contabilidad

Al finalizar la vela t, el agente observa el estado, solicita una exposición y el simulador ejecuta al precio de cierre t. Luego valora la posición al cierre t+1 y entrega la recompensa. Se trata de una ejecución idealizada al mismo cierre observado. No usa el cierre t+1 para elegir la acción, pero supone una liquidez y un precio de ejecución que no están garantizados en una operación real. Evaluar con ejecución en la apertura siguiente y datos consistentes de OHLC sería una extensión necesaria antes de interpretar los resultados como operables.

Sea V el patrimonio antes de operar, B el valor del activo antes del rebalanceo, X el valor del activo tras operar y c = {cfg['cost_rate']:.4f}. Se resuelve X = w[V - c|X-B|]. El costo monetario es C = c|X-B| y el efectivo restante es V-C-X. En compras, X = w(V+cB)/(1+wc); en ventas, X = w(V-cB)/(1-wc). Esto evita invertir todo el patrimonio y pagar la comisión con un saldo negativo.

Si R = p(t+1)/p(t)-1, entonces V(t+1) = V-C+XR. El peso deriva con el precio: mantener la misma acción 50 % puede exigir operar de nuevo. El costo depende del volumen monetario negociado, no de si cambió la etiqueta de acción. Comprar y mantener paga su compra inicial y conserva el activo. Todos comienzan en efectivo, pagan la misma tasa y se valoran a mercado al final, sin venta terminal obligatoria. El drawdown incluye el capital inicial.

La serie de mercado es exógena: las decisiones del agente no afectan el precio. El episodio termina al agotar las transiciones del periodo. El simulador implementa `reset` y `step` de Gymnasium y supera el comprobador de Stable-Baselines3.

### Auditoría de causalidad y lookahead

El conjunto de información usado para decidir en $t$ contiene únicamente precios e indicadores calculados hasta $t$. La acción se obtiene **antes** de consultar el movimiento $p_{{t+1}}/p_t$, que solo se utiliza para actualizar el patrimonio y calcular la recompensa. Dos pruebas automáticas refuerzan esta propiedad: `test_future_prices_do_not_change_past_features` modifica precios futuros y comprueba que las características históricas permanecen iguales; `test_test_data_do_not_affect_training_scaler` altera el tramo final y comprueba que la media y desviación del escalador de entrenamiento no cambian.

La ejecución al cierre $t$ después de conocer ese mismo cierre debe distinguirse de una fuga de información futura: no introduce $p_{{t+1}}$ en la decisión, pero sí es una **suposición optimista de ejecución**. En un escenario operativo más realista, la acción decidida al cierre debería ejecutarse en la apertura siguiente e incluir spread y deslizamiento. Esta limitación se mantiene explícita para no sobreinterpretar los resultados.

## Diseño e iteración de la recompensa

La versión inicial es r1 = 100 log[V(t+1)/V(t)]. Incluye las comisiones dentro del patrimonio. La suma de log-retornos equivale al logaritmo del crecimiento acumulado, pero no penaliza por separado la exposición a pérdidas diarias.

La versión revisada es r2 = 100 {{log[V(t+1)/V(t)] - lambda × min(g, 0)^2}}, con g = V(t+1)/V(t)-1 y lambda = {cfg['risk_lambda']:.1f}. El segundo término penaliza pérdidas netas grandes. El factor 100 es fijo en ambas versiones y solo cambia la escala. La penalización adicional cambia el objetivo económico y puede favorecer quedarse en efectivo.

Se entrenó primero la versión log y después la versión risk de cada algoritmo. La comparación utiliza validación; lambda y el presupuesto de entrenamiento se fijaron antes de abrir la prueba. Para {labels[selected['method']]}, log presentó retorno medio {log_v['return'].mean():.2%}, drawdown {log_v.max_drawdown.mean():.2%} y tiempo en efectivo {log_v.cash_fraction.mean():.2%}. Risk presentó retorno {risk_v['return'].mean():.2%}, drawdown {risk_v.max_drawdown.mean():.2%} y tiempo en efectivo {risk_v.cash_fraction.mean():.2%}. Esos datos permiten revisar si el cambio redujo pérdidas o si restringió demasiado la participación.{turnover_text}

La hipótesis de la segunda versión fue controlar las pérdidas diarias; no se afirma que su coeficiente sea óptimo. Las diferencias entre versiones son evidencia descriptiva de esta ejecución, no una prueba causal aislada de la penalización.

{table(vr, list(vr.columns))}

![Comparación en validación](reward_comparison.png)

Las barras de error representan una desviación estándar muestral del puntaje entre semillas. No son intervalos de confianza sobre mercados futuros.

Un puntaje alto de recompensa no basta para preferir una estrategia: quedarse fuera del mercado puede evitar penalizaciones y perder una subida. Por eso se reportan métricas monetarias, exposición y operaciones. No se identificó una explotación contable del entorno en las pruebas; eso no descarta otros comportamientos no deseados.

## Algoritmos y entrenamiento

Q-learning utiliza una tabla de 243 estados y 3 acciones. Discretiza momentum de 5 y 20 días y volatilidad estandarizados con cortes -0,5 y 0,5; el peso con 0,25 y 0,75; y el drawdown con -0,15 y -0,05. Descarta dos indicadores del estado completo para contener el número de celdas. Usa alpha = 0,1, gamma = 0,99 y {cfg['q_episodes']} {episode_text}. Epsilon baja linealmente de 1 a 0,05 en el primer 80 % de episodios. En evaluación se elige argmax; los empates favorecen la acción cero, lo cual puede sesgar estados sin visitas hacia efectivo.

DQN aproxima $Q(s,a)$ con una red de siete entradas, dos capas ocultas de 64 unidades con ReLU y tres salidas. La arquitectura es deliberadamente pequeña porque el estado es de baja dimensión y el objetivo es evitar complejidad innecesaria. Usa CPU y una hebra de PyTorch.

| Hiperparámetro | Valor | Justificación |
|---|---:|---|
| `learning_rate` | {cfg['dqn']['learning_rate']} | Paso de actualización conservador para reducir oscilaciones. |
| `batch_size` | {cfg['dqn']['batch_size']} | Equilibra estabilidad del gradiente y costo computacional. |
| `gamma` | {cfg['dqn']['gamma']} | Da peso alto a consecuencias futuras de la exposición. |
| `buffer_size` | {cfg['dqn']['buffer_size']} | Aumenta diversidad de experiencias y reduce correlación temporal inmediata. |
| `learning_starts` | {cfg['dqn']['learning_starts']} | Evita entrenar con un buffer prácticamente vacío. |
| `train_freq` | {cfg['dqn']['train_freq']} | Espacia actualizaciones y limita correlación entre muestras consecutivas. |
| `gradient_steps` | {cfg['dqn']['gradient_steps']} | Una actualización por evento de entrenamiento para no sobreoptimizar cada minibatch. |
| `target_update_interval` | {cfg['dqn']['target_update_interval']} | Mantiene un objetivo de Bellman más estable mediante la red target. |
| `exploration_fraction` | {cfg['dqn']['exploration_fraction']} | Reserva una fracción sustancial del presupuesto para exploración. |
| $\\epsilon$ | 1,0 → 0,05 | Transición de exploración amplia a explotación mayoritaria. |
| `net_arch` | [64, 64] | Capacidad suficiente para siete variables y tres acciones sin sobredimensionar. |

El entrenamiento termina al agotar {cfg['timesteps']} pasos, sin parada temprana basada en prueba, y se guarda el último modelo.

Semillas de entrenamiento y del baseline aleatorio: {cfg['seeds']}. La política de evaluación DQN es determinista. Las semillas replican la inicialización y la exploración, no crean nuevos mercados independientes. El número de interacciones de Q-learning es {budget_q}; el de DQN es {cfg['timesteps']}. La comparación no iguala el presupuesto de cómputo. Repetir episodios reutiliza el mismo recorrido histórico y puede producir memorización.

![Curvas de entrenamiento](learning_curves.png)

Las curvas muestran una media móvil dentro de cada entrenamiento. Sus alturas no comparan directamente objetivos distintos: risk resta una penalización que log no contiene. Mejorar la recompensa de entrenamiento tampoco demuestra que el agente generalice, como muestra la diferencia entre validación y prueba.

## Protocolo de evaluación

Se entrenan {n_cand} candidatos: cada algoritmo con cada recompensa. La regla de selección, fijada antes de prueba, es la media entre semillas de retorno de validación menos 0,5 veces el valor absoluto del drawdown máximo de validación. Este coeficiente expresa una preferencia explícita por limitar pérdidas; otra preferencia podría elegir otro candidato.

El archivo selection_frozen.json conserva la selección antes de ejecutar los recorridos finales. Se elige una combinación de algoritmo y recompensa, sin escoger la semilla ganadora. No se reentrena con validación. Se evalúan sus tres réplicas en prueba y también se muestran los otros candidatos como auditoría descriptiva. Sus resultados finales no cambian la selección ni los parámetros.

Los baselines son comprar y mantener, aleatorio uniforme entre las tres acciones y efectivo. Comparten fechas, capital inicial y comisiones. Comprar y mantener y efectivo tienen un solo recorrido porque no dependen de semillas. Se reporta la media y la desviación estándar muestral entre semillas; esta dispersión no es un intervalo de confianza sobre mercados futuros. No se realizan pruebas de significancia con esas tres réplicas como si fueran trayectorias independientes.

## Resultados fuera de muestra

{table(rows, list(rows.columns))}

Los costos están expresados en USDT nominales acumulados. Un drawdown de -20 % corresponde a una caída del 20 % desde un máximo. La desviación entre semillas puede ser cero si todas convergen a la misma política. La recompensa total no se compara directamente entre log y risk, pues miden objetivos distintos.

![Capital y drawdown](equity_drawdown.png)

El agente seleccionado pagó {winner.costs.mean():,.2f} USDT de costos medios, frente a {buy.costs:,.2f} de comprar y mantener. La comparación debe considerar conjuntamente retorno y drawdown: una mejora de un indicador con deterioro del otro no implica dominar al baseline.

En {beats} de las {len(winner)} semillas el agente seleccionado superó el retorno de comprar y mantener. Sus retornos variaron de {winner['return'].min():.2%} a {winner['return'].max():.2%}. Esa amplitud muestra que la conclusión cambia si se publica solo una semilla. La diferencia media fue {gain*100:.2f} puntos porcentuales. {verdict}

El cambio de validación a prueba también produjo un fallo de generalización observable: Q-learning con recompensa log pasó de un retorno medio de {q_log_val['return'].mean():.2%} a {q_log_test['return'].mean():.2%}. Esta comparación no mantiene fijo el mercado, por lo que no permite identificar la causa. Sí muestra que un resultado favorable en validación no asegura que la política conserve su desempeño después.

![Posiciones del agente](positions.png)

### Comportamiento observado y anomalías

{' '.join(obs)}

Como diagnóstico reproducible, cada CSV conserva la decisión, el retorno siguiente, la comisión y la exposición. El peor paso anterior permite revisar una pérdida concreta. Una política con exposición casi constante también merece revisión: puede haber aprendido una solución sencilla o haber dejado de distinguir estados. La causa no puede atribuirse a una sola variable sin nuevos experimentos realizados sobre entrenamiento y validación.

## Análisis exploratorio

El cuaderno notebooks/eda.ipynb analiza solo entrenamiento en su configuración inicial. Incluye precio, distribución de retornos, volatilidad móvil, drawdown del activo, volumen e indicadores. Se reserva el EDA completo para la descripción posterior al cierre del experimento. La siguiente figura reúne todo el historial y no se utilizó para seleccionar modelos.

![Historial completo, descripción posterior](eda.png)

Los retornos y la volatilidad se construyen con ventanas móviles para que el estado no dependa del nivel absoluto de BTC. La exposición y el drawdown añaden información del portafolio que el precio no describe. El volumen se analiza, pero se excluye del estado para mantener una representación pequeña. El archivo eda_train_summary.csv contiene estadísticas calculadas únicamente con entrenamiento.

En entrenamiento, el cierre mínimo fue {training.close.min():,.2f} USDT y el máximo {training.close.max():,.2f} USDT. La mayor caída diaria fue {low.return_1:.2%}, en la vela {str(low.timestamp)[:10]}. El drawdown máximo del precio fue {asset_drawdown:.2%}. Por eso una recompensa basada solo en crecimiento no describe toda la experiencia de pérdidas del inversionista.

La volatilidad móvil de 20 días alcanzó {high_vol.volatility_20:.2%} diario el {str(high_vol.timestamp)[:10]} y bajó hasta {low_vol.volatility_20:.2%} el {str(low_vol.timestamp)[:10]}. Estos cambios justifican observar volatilidad en vez de asumir un riesgo constante. {volume_sentence}La correlación de momentum a 20 días con la distancia a su media fue {training.momentum_20.corr(training.gap_sma20):.3f}, lo que indica información parcialmente redundante.

## Limitaciones y respuesta a la pregunta

En esta ejecución, la diferencia media de retorno del candidato seleccionado respecto a comprar y mantener fue {gain*100:.2f} puntos porcentuales. El resultado debe leerse junto con el riesgo y los costos de la tabla. No autoriza a afirmar que RL sea superior en mercados financieros en general. El baseline efectivo obtuvo {cash['return']:.2%}, lo que ayuda a detectar si operar destruyó capital.

El estudio usa un activo y una sola división temporal. El periodo final puede favorecer un régimen particular; una evaluación walk-forward en nuevos periodos ayudaría a estimar estabilidad. No incluye spread, deslizamiento, impuestos o restricciones de tamaño de orden. El efectivo no gana intereses y USDT se trata como unidad estable. La ejecución al mismo cierre observado es una aproximación optimista. Los datos de una bolsa no representan todos los mercados.

El periodo final de este experimento ya fue observado y, por integridad metodológica, no debe reutilizarse para ajustar arquitectura, recompensa o hiperparámetros. Cualquier cambio posterior debe tratar esta ejecución como exploratoria y reservar un nuevo periodo intacto para una evaluación final.

## Reflexión individual

### 30. ¿Cuál fue el resultado que más me sorprendió?

Al revisar los resultados, lo que más me sorprendió fue la **variabilidad entre semillas del DQN seleccionado**. Con el mismo mercado, arquitectura, recompensa e hiperparámetros, las tres réplicas produjeron retornos entre {winner['return'].min():.2%} y {winner['return'].max():.2%}. Una sola semilla habría permitido construir una conclusión muy diferente. Esto demuestra que una ejecución favorable no basta para hablar de robustez y que reportar la media y la dispersión cambia sustancialmente la interpretación.

### 31. ¿Qué decisión metodológica tuvo mayor impacto?

La separación temporal de entrenamiento, validación y prueba tuvo el mayor impacto en la interpretación. El candidato quedó congelado por validación antes de abrir el periodo final y se conservó aun cuando su retorno medio quedó por debajo de Buy & Hold. También resultó decisivo incluir comisiones: el agente seleccionado pagó {winner.costs.mean():,.2f} USDT en promedio, frente a {buy.costs:,.2f} USDT de comprar y mantener. Sin costos, una política con alto turnover podría parecer artificialmente competitiva.

### 32. ¿Qué cambiaría si repitiera el experimento?

Usaría varios cortes walk-forward, ejecutaría la decisión del cierre $t$ en la apertura $t+1$ e incorporaría spread y deslizamiento. También igualaría mejor los presupuestos de interacción de Q-learning y DQN y aumentaría el número de semillas. Todas esas decisiones se fijarían antes de reservar una nueva prueba final.

### 33. ¿Qué comportamiento no logró explicarse completamente?

No quedó completamente explicada la diferencia de exposición entre semillas DQN. Los registros permiten localizar las decisiones, pero no aislar si la causa principal es la inicialización, la exploración o la cobertura del replay buffer. Para responderlo haría falta un nuevo experimento que varíe una fuente de aleatoriedad a la vez.

### 34. ¿Cuál fue la principal diferencia entre RL teórico y su aplicación financiera?

En un entorno didáctico, repetir episodios suele producir experiencia útil bajo una dinámica relativamente estable. En este problema, repetir un episodio significa recorrer muchas veces **la misma historia de mercado**, que además es no estacionaria. El agente puede mejorar su recompensa en esa trayectoria sin aprender una política estable para otro régimen. Además, en finanzas no basta observar recompensa: costos, drawdown, exposición y estabilidad fuera de muestra son parte de la calidad de la política.

### Síntesis

El proyecto no demuestra que DQN sea superior a comprar y mantener; en esta ventana, el promedio no lo fue. El aprendizaje principal es que un experimento de RL financiero debe valorar la disciplina metodológica por encima del resultado favorable: evitar fuga temporal, separar selección y prueba, reportar semillas, incluir costos y conservar conclusiones negativas cuando los datos las muestran.

## Reproducción y archivos de evidencia

Desde la carpeta del proyecto: python -m pip install -r requirements.txt; python -m unittest discover -s tests -v; python -m src.train --config configs/default.json --output results/replica_01. Los comandos detallados están en README.md. No se necesita descargar datos para reproducir esta instantánea.

Cada ejecución guarda modelos, escalador, configuración, versiones, hash del dataset y métricas por semilla. Los archivos test_*.csv permiten recalcular las métricas y reconstruir los gráficos. Las curvas de aprendizaje DQN están en models/*_monitor.csv; las de Q-learning en learning_*.csv. Los errores de red o dependencias no se sustituyen por datos inventados.

## Referencias

- Sutton, R. S., y Barto, A. G. (2018). Reinforcement Learning: An Introduction, segunda edición. https://incompleteideas.net/book/the-book-2nd.html
- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. https://doi.org/10.1038/nature14236
- Stable-Baselines3. Documentación DQN. https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
- Gymnasium. API Env. https://gymnasium.farama.org/api/env/
- Gym Trading Env. Download market data. https://gym-trading-env.readthedocs.io/en/latest/download.html
- Binance. Documentación de velas del API spot. https://developers.binance.com/docs/binance-spot-api-docs/rest-api/market-data-endpoints#klinecandlestick-data
"""
    (out/"INFORME.md").write_text(text, encoding="utf-8")
