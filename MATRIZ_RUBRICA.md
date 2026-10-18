# Matriz de cumplimiento de la rúbrica — objetivo 5,0/5,0

Esta matriz permite al evaluador localizar rápidamente la evidencia de cada criterio. No sustituye el informe; organiza las pruebas ya incluidas en el repositorio.

## 1. Formulación del problema: estado y acciones — objetivo 1,0

**Exigencia de nivel excelente:** estado claramente definido con indicadores justificados y espacio de acciones discreto e interpretable, coherente con el objetivo de inversión.

**Evidencia:**

- `README.md`, secciones 2.1 y 2.2: tabla de las siete variables del estado, fórmulas, ventanas y justificación.
- `src/data.py`: construcción causal de `return_1`, `momentum_5`, `momentum_20`, `gap_sma20` y `volatility_20`.
- `src/environment.py`: incorporación de `current_weight` y `drawdown`, y `Discrete(3)`.
- `results/main/INFORME.md`: formulación económica y limitaciones del estado.

**Comprobación clave:** ninguna variable requiere información posterior a $t$ y el escalador se ajusta únicamente con entrenamiento.

## 2. Implementación del entorno de inversión — objetivo 1,0

**Exigencia de nivel excelente:** entorno funcional, costo por operar y sin usar información futura.

**Evidencia:**

- `src/environment.py`: contabilidad del portafolio, rebalanceo y comisión de 0,10 % sobre el valor negociado.
- `tests/test_environment.py`: pruebas de efectivo, inversión total, Buy & Hold, venta, rebalanceo, recompensa y drawdown.
- `tests/test_environment.py::test_future_prices_do_not_change_past_features`: auditoría de causalidad de los indicadores.
- `tests/test_environment.py::test_test_data_do_not_affect_training_scaler`: prueba de aislamiento del escalador.
- `results/VERIFICACION.md`: registro de 12 pruebas y `check_env` de Stable-Baselines3.

**Supuesto declarado:** se decide al final de la vela $t$ y se ejecuta idealizadamente al mismo cierre. No se usa $p_{t+1}$ para decidir, pero el precio de ejecución es optimista; el informe lo reconoce y propone ejecutar en apertura $t+1$ como mejora.

## 3. Diseño de la recompensa — objetivo 1,0

**Exigencia de nivel excelente:** recompensa justificada e iterada; comparar al menos dos alternativas y discutir comportamiento inducido.

**Evidencia:**

- Recompensa V1 `log`: crecimiento logarítmico del patrimonio neto.
- Recompensa V2 `risk`: V1 menos penalización cuadrática de pérdidas diarias.
- `configs/default.json`: `risk_lambda = 10.0`.
- `results/main/validation_metrics.csv`: comparación cuantitativa de ambas recompensas para Q-learning y DQN.
- `results/main/reward_comparison.png`: comparación gráfica.
- `README.md` sección 5 e `INFORME.md`: hipótesis de la modificación, resultado obtenido y riesgo de incentivar permanencia en efectivo.

**Conclusión:** la segunda recompensa no produjo la mejora esperada; se conserva el resultado como parte de la iteración metodológica.

## 4. Evaluación en datos no vistos y baselines — objetivo 1,0

**Exigencia de nivel excelente:** prueba held-out sin reajuste, métricas claras y comparación con comprar y mantener y agente aleatorio.

**Evidencia:**

- Separación temporal 60/20/20 en `configs/default.json` y `manifest.json`.
- `selection_frozen.json`: combinación algoritmo/recompensa congelada antes de prueba.
- `src/train.py`: entrenamiento/validación primero; prueba se recorre después de persistir la selección.
- Baselines: Buy & Hold, aleatorio y efectivo (`src/baselines.py`).
- `results/main/test_metrics.csv` y `test_summary.csv`: retorno, drawdown, costos, operaciones y exposición.
- `results/main/equity_drawdown.png`: comparación visual bajo la misma ventana final.

**Resultado seleccionado:** DQN/log = 89,49 % de retorno medio y −26,11 % de drawdown medio. Buy & Hold = 111,76 % y −26,15 %. Aleatorio = 11,91 % y −35,20 %.

## 5. Discusión, reflexión y reproducibilidad — objetivo 1,0

**Exigencia de nivel excelente:** discusión honesta de limitaciones, reward hacking y no estacionariedad; reflexión personal; semillas fijas e instrucciones de ejecución.

**Evidencia:**

- `results/main/INFORME.md`: limitaciones, no estacionariedad, sensibilidad entre semillas, costos, ejecución idealizada y ausencia de garantías de rentabilidad futura.
- `README.md`: discusión de *reward hacking* por permanencia en efectivo y limitaciones estructurales.
- `REFLEXION_INDIVIDUAL.md`: respuestas explícitas a las preguntas 30–34.
- `configs/default.json`: semillas `[7, 21, 42]`.
- `requirements.txt`: versiones fijadas.
- `results/main/manifest.json`: versiones reales, plataforma, hash del dataset y particiones.
- `README.md`: comandos de instalación, pruebas, entrenamiento y reevaluación.
- `src/reevaluate.py`: reevaluación de modelos guardados sin entrenar ni modificar la selección.

## Resumen

El proyecto incorpora evidencia explícita para los cinco criterios de la rúbrica. El resultado final no se presenta como prueba de superioridad financiera: la metodología conserva el desempeño real del held-out, incluso cuando Buy & Hold obtiene un retorno medio superior. Esa transparencia es parte central del diseño del entregable.
