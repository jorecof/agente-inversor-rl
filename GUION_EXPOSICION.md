# Guion de exposición — 5 minutos

## 0:00–0:40 | Problema y objetivo

El proyecto responde una pregunta concreta: **¿puede un agente de Reinforcement Learning aprender una estrategia de exposición a BTC que se comporte mejor que estrategias simples cuando se evalúa en datos que no vio durante el entrenamiento?**

El agente parte de **100.000 USDT** y en cada día decide entre tres acciones: mantener **0 %, 50 % o 100 %** del patrimonio expuesto a BTC. La evaluación no se limita a la rentabilidad: también considera drawdown y costos de transacción.

## 0:40–1:20 | Datos, estado y control de fuga temporal

Se utilizaron velas diarias BTC/USDT de Binance entre 2020 y 2024. Los datos se separaron cronológicamente en entrenamiento, validación y prueba. El periodo final de 2024 se reservó como *held-out*.

El estado del DQN tiene siete variables: retorno de 1 día, momentum de 5 y 20 días, distancia frente a la media móvil de 20 días, volatilidad de 20 días, peso actual del portafolio y drawdown. Los indicadores usan únicamente información disponible hasta el instante de decisión. El escalamiento se ajusta solo con entrenamiento y queda congelado para validación y prueba.

La implementación incluye pruebas automáticas para comprobar que modificar precios futuros no cambia características pasadas y que los datos de prueba no alteran el escalador de entrenamiento.

## 1:20–2:10 | Entorno, costos y recompensa

La comisión es de **0,10 %** del valor negociado. El entorno resuelve el rebalanceo considerando la comisión para evitar financiar costos con deuda.

Se compararon dos recompensas. La primera es el crecimiento logarítmico del patrimonio. La segunda añade una penalización cuadrática cuando el retorno diario es negativo. La intención era reducir pérdidas grandes.

Sin embargo, en validación el DQN con recompensa log obtuvo **88,07 % de retorno y −17,17 % de drawdown**, mientras que la recompensa de riesgo obtuvo **75,99 % y −19,02 %**. La modificación no produjo la mejora esperada. Por eso se seleccionó DQN con recompensa logarítmica.

## 2:10–3:05 | Algoritmo y protocolo de selección

Se compararon Q-learning tabular y DQN. El DQN usa una red de **7 entradas, dos capas ocultas de 64 neuronas con ReLU y 3 salidas**. Usa replay buffer, política epsilon-greedy y red objetivo.

Se ejecutaron tres semillas: 7, 21 y 42. La combinación de algoritmo y recompensa se seleccionó únicamente con validación mediante un puntaje que combina retorno y drawdown. Después se guardó `selection_frozen.json`. Solo entonces se ejecutó la prueba final.

Este punto es importante: **no se eligió la mejor semilla y no se cambió el modelo después de ver 2024**.

## 3:05–4:05 | Resultados

En prueba, Buy & Hold obtuvo **111,76 % de retorno** y **−26,15 % de drawdown**.

El DQN seleccionado obtuvo en promedio **89,49 % de retorno**, **−26,11 % de drawdown** y aproximadamente **7.885 USDT en costos de transacción**. El agente aleatorio obtuvo solo **11,91 %** y un drawdown de **−35,20 %**.

Por tanto, el DQN sí aprendió una política claramente mejor que la aleatoria, pero **no superó a Buy & Hold en retorno medio**. Además, pagó aproximadamente 79 veces más costos que Buy & Hold.

La sensibilidad a la semilla fue alta: las tres réplicas del DQN seleccionado produjeron retornos entre **40,98 % y 150,75 %**. Una sola semilla habría podido dar una conclusión engañosa.

## 4:05–4:40 | Limitaciones y comportamiento inesperado

El principal riesgo metodológico es la no estacionariedad del mercado. Entrenar muchas veces sobre la misma historia no equivale a observar nuevos mercados.

También existe un posible *reward hacking*: una recompensa con penalización de pérdidas puede incentivar permanecer demasiado tiempo en efectivo. Por eso se reportan exposición, operaciones, costos y retorno monetario, no solo recompensa.

El supuesto más optimista es ejecutar al mismo cierre que el agente acaba de observar. No usa información de $t+1$, pero no incorpora spread ni deslizamiento. Una versión futura debería decidir al cierre y ejecutar en la apertura siguiente.

## 4:40–5:00 | Conclusión

La conclusión del proyecto no es que DQN sea superior al mercado. En esta ventana no lo fue en promedio. El aprendizaje principal es metodológico: en RL financiero hay que separar temporalmente los datos, congelar la selección antes del test, comparar con baselines simples, incluir costos y conservar resultados negativos.

El siguiente paso sería una evaluación walk-forward con varios periodos no vistos, mayor número de semillas y una ejecución más realista.

---

## Preguntas probables del panel

**¿Cómo evitan lookahead?**  
Los indicadores son causales, el escalador se ajusta solo en entrenamiento, no se cruzan particiones y la prueba no participa en selección. Hay pruebas unitarias específicas de causalidad.

**¿Por qué DQN?**  
Porque el espacio de acciones es discreto y pequeño. DQN aproxima el valor de las tres acciones a partir de un estado continuo de siete variables.

**¿Por qué mantener Q-learning?**  
Sirve como referencia clásica y permite comprobar si la red neuronal aporta valor frente a una aproximación tabular más simple.

**¿Por qué no elegir la semilla 21, que obtuvo el mejor retorno?**  
Porque seleccionar la mejor semilla después de observar prueba introduce sesgo. Las semillas son réplicas del entrenamiento y deben reportarse en conjunto.

**¿Qué evidencia muestra la iteración de recompensa?**  
Se entrenaron y validaron dos funciones: crecimiento logarítmico y crecimiento con penalización de pérdidas. La segunda no mejoró el objetivo esperado y el informe conserva ese resultado.

**¿Qué mejorarían primero?**  
Ejecución en apertura siguiente, spread/deslizamiento y evaluación walk-forward con nuevas ventanas held-out.
