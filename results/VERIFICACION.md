# Verificación de la entrega

- Las 12 pruebas de tests/test_environment.py pasaron con unittest.
- El entorno pasó check_env de Stable-Baselines3.
- Se entrenaron 12 modelos con las semillas 7, 21 y 42.
- La selección se guardó antes del primer recorrido de prueba: DQN con recompensa log.
- La carga de los modelos guardados y su reevaluación reprodujeron las métricas monetarias, con tolerancia relativa 1e-9 y absoluta 1e-7.
- Se ejecutaron todas las celdas de código del cuaderno EDA sin errores, con un backend gráfico sin interfaz.
- Se revisaron visualmente las figuras de capital, drawdown, exposición, comparación de recompensas, aprendizaje y EDA.

La auditoría no entrena modelos nuevos ni utiliza la prueba para modificar el agente. Sus métricas están en auditoria_01/test_metrics.csv.
