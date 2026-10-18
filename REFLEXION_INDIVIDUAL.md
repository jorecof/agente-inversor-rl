# Reflexión individual — Proyecto final «Agente Inversor»

Esta reflexión responde de manera explícita a las preguntas planteadas en la actividad y se apoya únicamente en los resultados obtenidos en el experimento reproducible incluido en el repositorio.

## 30. ¿Cuál fue el resultado que más me sorprendió?

El resultado que más me sorprendió fue la **variabilidad del DQN entre semillas aun usando exactamente el mismo mercado, arquitectura, recompensa e hiperparámetros**. El candidato seleccionado, DQN con recompensa logarítmica, obtuvo en prueba retornos aproximados de **40,98 %, 150,75 % y 76,76 %** para las semillas 7, 21 y 42. La media fue **89,49 %**, pero una sola semilla habría permitido contar una historia completamente diferente: la mejor réplica superó a comprar y mantener, mientras que el promedio de las tres no lo hizo.

Este resultado cambió mi forma de interpretar un agente de Reinforcement Learning. Una curva favorable o una ejecución sobresaliente no son suficientes para afirmar que existe una estrategia robusta. La dispersión entre semillas muestra que la política aprendida depende de la trayectoria de exploración, de la inicialización de la red y de qué experiencias terminan siendo más influyentes en el replay buffer.

## 31. ¿Qué decisión metodológica tuvo mayor impacto en el agente?

Considero que la decisión metodológica de mayor impacto fue **separar entrenamiento, validación y prueba, y congelar el candidato antes de observar el periodo final**. El agente DQN con recompensa logarítmica fue elegido porque obtuvo el mejor puntaje medio en validación, no porque fuera el mejor retrospectivamente en 2024.

Esta decisión evitó seleccionar el modelo conociendo de antemano el resultado final. También permitió observar un fenómeno importante: Q-learning con recompensa log pasó de un retorno promedio positivo en validación a un resultado ligeramente negativo en prueba. Sin la separación temporal, esa pérdida de generalización habría podido quedar oculta.

Una segunda decisión relevante fue incorporar costos de transacción. El DQN seleccionado pagó en promedio **7.885,40 USDT**, frente a aproximadamente **99,90 USDT** de comprar y mantener. Por ello, una política que cambia frecuentemente de exposición puede parecer activa e inteligente, pero destruir una fracción importante de su ventaja por *turnover*.

## 32. ¿Qué cambiaría si tuviera que repetir el experimento?

Realizaría cuatro cambios principales.

Primero, utilizaría una evaluación **walk-forward con varias ventanas temporales**, en lugar de depender de una sola partición final. Esto permitiría observar al agente en regímenes alcistas, bajistas y laterales sin reutilizar el mismo periodo para tomar decisiones metodológicas.

Segundo, modificaría el supuesto de ejecución. En el experimento actual el agente observa el cierre de la vela $t$ y el simulador ejecuta idealizadamente a ese mismo cierre. Aunque no se utiliza el precio $t+1$ para escoger la acción, es un supuesto optimista. En una nueva versión decidiría al cierre de $t$ y ejecutaría en la apertura de $t+1$, incorporando además spread y deslizamiento.

Tercero, igualaría mejor el presupuesto experimental entre Q-learning y DQN. Actualmente Q-learning recorre repetidamente la misma historia durante 200 episodios y DQN usa 12.000 pasos. Una comparación más controlada debería justificar el presupuesto de interacciones y el número de actualizaciones de cada método.

Cuarto, ampliaría el análisis de sensibilidad con más semillas y, si los recursos lo permiten, variaría de forma controlada un pequeño conjunto de hiperparámetros usando únicamente entrenamiento y validación. El periodo de prueba quedaría completamente aislado hasta el cierre del estudio.

## 33. ¿Qué comportamiento del agente no logró explicar completamente?

No quedó completamente explicada la **diferencia de exposición entre las semillas del DQN seleccionado**. En prueba, la semilla 7 permaneció aproximadamente 38,0 % del tiempo con exposición cero y tuvo un peso objetivo medio cercano a 43,5 %. La semilla 21 permaneció en efectivo solo cerca de 19,4 % y alcanzó una exposición media de aproximadamente 78,9 %, mientras que la semilla 42 estuvo fuera del mercado cerca de 51,8 % y mantuvo una exposición media cercana a 35,7 %.

Los archivos de transición permiten identificar cuándo aparecen las decisiones y las pérdidas, pero no permiten atribuir esa diferencia a una única causa. Puede provenir de la inicialización, de la secuencia de exploración, de la cobertura del replay buffer o de aproximaciones diferentes de la función $Q$. Para aislar la causa sería necesario diseñar un experimento adicional donde se controle una fuente de variación a la vez y se reserve un nuevo conjunto de prueba que no haya sido observado.

## 34. ¿Cuál fue la principal diferencia entre lo aprendido en Reinforcement Learning y su aplicación a este problema?

En entornos didácticos de Reinforcement Learning es común asumir que el entorno es relativamente estable y que repetir episodios produce nuevas experiencias útiles. En una serie histórica financiera esto cambia: **repetir el episodio significa recorrer una y otra vez la misma historia de precios**. El agente puede mejorar su recompensa dentro de ese recorrido sin haber aprendido una regla que generalice a un régimen futuro.

Además, en finanzas la recompensa no puede analizarse de forma aislada. Una política puede obtener una recompensa alta y al mismo tiempo presentar un drawdown elevado, costos excesivos o un comportamiento poco realista. Esto obliga a evaluar retorno, riesgo, costos, exposición, robustez entre semillas y desempeño en datos no vistos.

Mi principal aprendizaje es que el valor del proyecto no está en demostrar que DQN “gana” al mercado. En esta prueba, de hecho, el promedio del agente quedó por debajo de comprar y mantener. El valor está en **formular correctamente el proceso de decisión, impedir fuga temporal, comparar contra baselines honestos y conservar el resultado aunque contradiga la hipótesis inicial**. Esa disciplina metodológica es más importante que seleccionar retrospectivamente la curva más atractiva.

## Conclusión personal

El proyecto muestra que un agente puede aprender comportamientos distintos de una política aleatoria, pero también que la generalización financiera es difícil. La combinación de no estacionariedad, costos de transacción y sensibilidad a la semilla limita la evidencia a favor de una superioridad del agente. Si continuara el trabajo, priorizaría robustez fuera de muestra y realismo de ejecución antes que aumentar la complejidad de la red neuronal.
