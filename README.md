# Proyecto final — Agente inversor con Aprendizaje por Refuerzo

**Activo:** BTC/USDT (Binance Spot)  
**Algoritmos:** Q-learning tabular y Deep Q-Network (DQN)  
**Acciones:** 0 %, 50 % o 100 % del patrimonio expuesto a BTC  
**Evaluación:** separación temporal entrenamiento → validación → prueba *held-out*  
**Baselines:** comprar y mantener, agente aleatorio y efectivo  

Este repositorio implementa un entorno de inversión reproducible y compara agentes de Reinforcement Learning con estrategias simples bajo las mismas fechas, capital inicial y costos de transacción. La prioridad metodológica es evitar fuga temporal, congelar la selección antes de la prueba final y reportar resultados favorables y desfavorables sin escoger retrospectivamente la mejor semilla.

> **Resultado principal.** El candidato seleccionado únicamente con validación fue **DQN + recompensa logarítmica**. En el periodo final 2024 obtuvo un retorno medio de **89,49 %**, frente a **111,76 %** de comprar y mantener. El drawdown medio fue **−26,11 %** frente a **−26,15 %** del baseline. Por tanto, en esta ventana el agente **no supera en retorno medio** a comprar y mantener, aunque sí supera ampliamente al agente aleatorio. El resultado negativo frente al baseline principal se conserva porque la prueba final no se utiliza para reajustar el experimento.

El informe completo y las cifras reproducibles están en [`results/main/INFORME.md`](results/main/INFORME.md). También se incluye una versión autónoma en [`results/main/INFORME.html`](results/main/INFORME.html).

---

## 1. Correspondencia directa con la rúbrica

| Criterio | Evidencia principal |
|---|---|
| **Formulación del problema** | Estado de 7 variables, acciones discretas 0/50/100 %, justificación económica y causalidad temporal en este README, `src/data.py`, `src/environment.py` e `INFORME.md`. |
| **Entorno de inversión** | Contabilidad con comisión de 0,10 %, Gymnasium, pruebas unitarias, chequeo de causalidad y explicación explícita del supuesto de ejecución al cierre. |
| **Diseño de recompensa** | Dos versiones: `log` y `risk`; ambas se entrenan, comparan en validación y se documenta el comportamiento inducido. |
| **Held-out y baselines** | Train/validation/test cronológicos; selección congelada en `selection_frozen.json`; Buy & Hold, aleatorio y efectivo bajo las mismas condiciones. |
| **Discusión, reflexión y reproducibilidad** | Limitaciones, *reward hacking*, no estacionariedad, variación entre semillas, reflexión individual y comandos reproducibles. |

La matriz detallada de cumplimiento está en [`MATRIZ_RUBRICA.md`](MATRIZ_RUBRICA.md).

---

## 2. Formulación del problema como MDP

En cada instante de decisión el agente recibe un estado $s_t$, selecciona una acción $a_t$, el portafolio se rebalancea, el mercado evoluciona y se obtiene una recompensa $r_{t+1}$.

### 2.1 Estado observado por el agente

La observación DQN tiene **7 componentes**. Los cinco primeros provienen únicamente de precios disponibles hasta el instante $t$ y se estandarizan usando exclusivamente estadísticas del conjunto de entrenamiento.

| Variable | Definición | Ventana / rango | Justificación |
|---|---|---|---|
| `return_1` | $p_t/p_{t-1}-1$ | 1 día | Captura el movimiento más reciente. |
| `momentum_5` | $p_t/p_{t-5}-1$ | 5 días | Resume tendencia de corto plazo. |
| `momentum_20` | $p_t/p_{t-20}-1$ | 20 días | Resume tendencia de horizonte mayor. |
| `gap_sma20` | $p_t/SMA_{20,t}-1$ | 20 días | Mide separación frente al nivel reciente. |
| `volatility_20` | desviación estándar de `return_1` | 20 días | Aproxima el régimen de variabilidad/riesgo. |
| `current_weight` | valor riesgoso / patrimonio | $[0,1]$ | Informa la exposición que ya tiene el portafolio. |
| `drawdown` | patrimonio / máximo histórico − 1 | $[-1,0]$ | Informa la pérdida respecto al máximo alcanzado. |

Los indicadores de mercado se estandarizan con media y desviación calculadas **solo en train**, se congelan para validación/prueba y se limitan a `[-10, 10]`. No se usan ventanas centradas ni relleno hacia atrás.

### 2.2 Espacio de acciones

El espacio es `Discrete(3)`:

| Acción | Peso objetivo | Interpretación económica |
|---:|---:|---|
| `0` | 0 % | Mantener todo el patrimonio en efectivo. |
| `1` | 50 % | Mantener exposición intermedia a BTC. |
| `2` | 100 % | Mantener exposición total a BTC. |

La granularidad 0/50/100 % es deliberada: mantiene el problema interpretable, permite usar DQN de forma natural y posibilita una comparación coherente contra Q-learning tabular. No se permiten ventas en corto ni apalancamiento.

---

## 3. Datos y separación temporal

La instantánea contiene **1.827 velas diarias BTC/USDT** de Binance entre 2020 y 2024. Después del calentamiento de 20 observaciones, la separación usada en el experimento es:

| Partición | Fechas | Transiciones | Uso permitido |
|---|---|---:|---|
| Entrenamiento | 2020-01-21 → 2023-01-08 | 1.083 | Aprender parámetros y ajustar escalamiento. |
| Validación | 2023-01-09 → 2024-01-04 | 360 | Comparar algoritmo/recompensa y congelar selección. |
| Prueba *held-out* | 2024-01-05 → 2024-12-31 | 361 | Evaluación final; no selecciona ni reajusta. |

No se barajan filas y ninguna transición cruza de una partición a otra. El hash SHA-256 del CSV queda registrado en el manifiesto para comprobar que la reevaluación usa exactamente la misma instantánea.

### Auditoría de *lookahead*

La información del estado en $t$ usa únicamente precios hasta $t$. La acción se determina antes de utilizar $p_{t+1}$ y la recompensa depende del movimiento posterior $t \rightarrow t+1$. Además, `tests/test_environment.py` verifica que modificar precios futuros no cambia características pasadas y que modificar el tramo de prueba no altera el escalador de entrenamiento.

El simulador **ejecuta idealizadamente al cierre $t$ después de observar ese cierre**. Esto no introduce $p_{t+1}$ en la decisión, pero sí supone que se puede negociar exactamente al último precio observado, sin spread ni deslizamiento. Se documenta como una limitación. Para una simulación operativa más conservadora, la extensión recomendada es decidir al cierre $t$ y ejecutar en la apertura $t+1$.

---

## 4. Entorno de inversión y costos

Capital inicial: **100.000 USDT**. Comisión: **0,10 %** del valor monetario realmente negociado.

Si $V$ es el patrimonio antes de operar, $B$ el valor riesgoso previo, $w$ el peso objetivo y $c$ la comisión, el simulador resuelve el valor objetivo $X$ considerando el costo de la propia operación:

$$
X = w\,[V-c|X-B|]
$$

Así, una acción de 100 % no financia la comisión con deuda. Repetir 50 % puede generar una nueva operación porque el peso real deriva cuando cambia el precio. Buy & Hold paga la compra inicial y luego conserva el activo.

El entorno usa la interfaz Gymnasium `reset/step` y fue validado con `check_env` de Stable-Baselines3. Las pruebas unitarias cubren efectivo, inversión total, compra y mantenimiento, venta, rebalanceo, recompensas, drawdown y causalidad temporal.

---

## 5. Diseño e iteración de la recompensa

La actividad exige una iteración real de diseño, por lo que se entrenaron **dos funciones distintas** antes de abrir el periodo de prueba.

### Versión 1 — crecimiento logarítmico

$$
r_t^{(1)} = 100\log\left(\frac{V_{t+1}}{V_t}\right)
$$

Ventaja: al sumar recompensas se representa directamente el crecimiento compuesto del patrimonio y las comisiones ya están incorporadas en $V_{t+1}$.

Problema observado: no penaliza de forma adicional una pérdida diaria grande; una política puede aceptar caídas pronunciadas si espera compensarlas con crecimiento posterior.

### Versión 2 — crecimiento con penalización de pérdida

$$
r_t^{(2)} = 100\left[\log\left(\frac{V_{t+1}}{V_t}\right)-\lambda\min(g_t,0)^2\right],\qquad \lambda=10
$$

con $g_t=V_{t+1}/V_t-1$.

La hipótesis era reducir pérdidas grandes. En validación, para DQN, la versión `log` alcanzó **88,07 %** de retorno y **−17,17 %** de drawdown, mientras `risk` alcanzó **75,99 %** y **−19,02 %**. En esta ejecución la modificación no consiguió el objetivo esperado; por ello se conserva `log` como candidato seleccionado. Esto es una iteración documentada, no una afirmación de que $\lambda=10$ sea óptimo.

Un riesgo de *reward hacking* es que una penalización de pérdidas incentive permanecer fuera del mercado para evitar castigos. Por eso la evaluación final no usa solamente la recompensa: reporta retorno monetario, drawdown, costos, operaciones y tiempo en efectivo.

---

## 6. Algoritmos

### Q-learning tabular

Se utiliza como referencia clásica. Discretiza momentum de 5 y 20 días, volatilidad, posición y drawdown en 243 estados, con tres acciones. Parámetros principales: $\alpha=0.1$, $\gamma=0.99$, 200 episodios y $\epsilon$ decreciente de 1 a 0,05.

### Deep Q-Network (DQN)

Arquitectura:

```text
Estado (7)
   ↓
Dense 64 + ReLU
   ↓
Dense 64 + ReLU
   ↓
Q(a=0), Q(a=1), Q(a=2)
```

Hiperparámetros:

| Parámetro | Valor | Razón de diseño |
|---|---:|---|
| `learning_rate` | 0.0003 | Paso pequeño para reducir oscilaciones en la aproximación de Q. |
| `buffer_size` | 20.000 | Mantiene diversidad de transiciones y rompe correlación temporal inmediata. |
| `learning_starts` | 500 | Evita actualizar la red con un buffer prácticamente vacío. |
| `batch_size` | 64 | Compromiso entre estabilidad del gradiente y costo computacional. |
| `gamma` | 0.99 | Da peso alto a consecuencias futuras de las decisiones de inversión. |
| `train_freq` | 4 | Reduce actualizaciones excesivamente correlacionadas paso a paso. |
| `gradient_steps` | 1 | Una actualización por evento de entrenamiento; evita sobreoptimizar cada lote. |
| `target_update_interval` | 500 | Estabiliza el objetivo de Bellman mediante una red objetivo más lenta. |
| `exploration_fraction` | 0.40 | Mantiene exploración sustancial durante la primera parte del entrenamiento. |
| $\epsilon$ | 1,0 → 0,05 | Pasa de exploración amplia a política mayoritariamente codiciosa. |
| Red | `[64, 64]` | Suficiente capacidad para 7 entradas y 3 acciones sin sobredimensionar el problema. |

Se entrenan tres semillas (`7`, `21`, `42`) para no depender de una única inicialización. Estas semillas **no son mercados independientes**; representan variación del proceso de aprendizaje sobre el mismo historial.

---

## 7. Selección y evaluación *held-out*

Se comparan cuatro candidatos en validación: Q-learning/DQN × recompensa log/risk. La regla fijada es:

$$
score = retorno_{validación} - 0.5\,|drawdown_{validación}|
$$

La combinación seleccionada fue **DQN + log**. La elección se persiste en `results/main/selection_frozen.json` **antes** de recorrer prueba. No se selecciona la mejor semilla y no se reentrena con el periodo final.

### Resultado final 2024

| Estrategia | Retorno medio | Drawdown medio | Costos medios (USDT) | Operaciones medias |
|---|---:|---:|---:|---:|
| Comprar y mantener | **111,76 %** | −26,15 % | 99,90 | 1,0 |
| **DQN / log (seleccionado)** | **89,49 %** | **−26,11 %** | **7.885,40** | **133,0** |
| DQN / risk | 89,14 % | −26,43 % | 8.002,27 | 127,3 |
| Q-learning / log | −3,56 % | −23,49 % | 11.696,24 | 224,0 |
| Q-learning / risk | −4,35 % | −14,17 % | 7.015,43 | 121,3 |
| Aleatorio | 11,91 % | −35,20 % | 17.416,71 | 283,3 |
| Efectivo | 0,00 % | 0,00 % | 0,00 | 0,0 |

Interpretación:

- El DQN seleccionado quedó **22,27 puntos porcentuales** por debajo de Buy & Hold en retorno medio.
- Su drawdown medio fue prácticamente igual al de Buy & Hold: diferencia aproximada de **0,05 puntos porcentuales**.
- Pagó aproximadamente **79 veces** más costos que Buy & Hold, evidencia de que el *turnover* importa económicamente.
- Superó al agente aleatorio por aproximadamente **77,59 puntos porcentuales** de retorno medio.
- Una de las tres semillas DQN sí superó Buy & Hold, pero los retornos variaron entre **40,98 % y 150,75 %**. Por eso no se reporta selectivamente la mejor semilla.

La conclusión correcta no es que RL “ganó”, sino que el agente aprendió políticas no triviales pero mostró **alta sensibilidad a la semilla y costos de operación elevados**, y no demostró superioridad robusta sobre un baseline sencillo en el periodo no visto.

---

## 8. Limitaciones, no estacionariedad y comportamiento inesperado

El estudio tiene limitaciones importantes que deben formar parte de la conclusión:

- un solo activo y una única ventana temporal no representan todos los regímenes de mercado;
- el mercado es no estacionario: relaciones útiles durante entrenamiento pueden cambiar en validación/prueba;
- repetir episodios reutiliza la misma historia, por lo que el agente puede memorizar regularidades del tramo de entrenamiento;
- la simulación no incluye spread, deslizamiento, impuestos, impacto de mercado ni intereses del efectivo;
- ejecutar al mismo cierre observado es un supuesto optimista;
- tres semillas miden sensibilidad del entrenamiento, no incertidumbre estadística sobre mercados futuros;
- la gran diferencia de exposición entre semillas DQN sigue sin quedar completamente explicada;
- una política que se mantiene demasiado en efectivo podría estar explotando la recompensa de riesgo en lugar de aprender una mejor estrategia de inversión.

La extensión metodológica prioritaria sería una evaluación **walk-forward**, con ejecución en el siguiente precio negociable y un nuevo periodo final que permanezca intacto después de fijar decisiones e hiperparámetros.

---

## 9. Reflexión individual

La reflexión completa está en [`REFLEXION_INDIVIDUAL.md`](REFLEXION_INDIVIDUAL.md). En síntesis, el principal aprendizaje es que en Reinforcement Learning financiero la calidad del experimento no puede evaluarse por una sola curva de recompensa: separación temporal, costos, baselines y estabilidad entre semillas cambian la interpretación del resultado.

---

## 10. Reproducibilidad

### Requisitos

Ejecución original registrada: Python 3.9.6, CPU, macOS arm64. Las versiones de las bibliotecas se fijan en `requirements.txt` y se guardan nuevamente en `results/main/manifest.json`.

### Instalación

Descarga o clona este repositorio y ubícate en su carpeta raíz. Luego crea el entorno virtual:

```bash
python -m venv .venv
```

Linux/macOS:

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Pruebas

```bash
python -m unittest discover -s tests -v
```

### Ejecutar una réplica nueva

```bash
python -m src.train --config configs/default.json --output results/replica_01
```

El comando entrena 12 modelos (2 algoritmos × 2 recompensas × 3 semillas), selecciona solamente con validación, congela la elección y después ejecuta la prueba y los baselines.

### Reevaluar modelos guardados sin reentrenar

```bash
python -m src.reevaluate --run results/main --output results/auditoria_01
```

Este comando comprueba el hash del dataset, conserva la selección congelada y reproduce las métricas monetarias de los modelos guardados.

### Regenerar el informe HTML

```bash
python -m src.export_html --source results/main/INFORME.md --output results/main/INFORME.html
```

---

## 11. Estructura del repositorio

```text
.
├── configs/
│   └── default.json
├── data/
│   ├── btcusdt_daily.csv
│   └── btcusdt_daily.metadata.json
├── notebooks/
│   ├── eda.ipynb
│   └── proyecto_inversion.ipynb
├── results/
│   ├── main/
│   │   ├── INFORME.md
│   │   ├── INFORME.html
│   │   ├── models/
│   │   ├── test_metrics.csv
│   │   ├── validation_metrics.csv
│   │   ├── selection_frozen.json
│   │   └── *.png
│   └── VERIFICACION.md
├── src/
│   ├── agent.py
│   ├── baselines.py
│   ├── data.py
│   ├── environment.py
│   ├── evaluate.py
│   ├── report.py
│   └── train.py
├── tests/
│   └── test_environment.py
├── GUION_EXPOSICION.md
├── MATRIZ_RUBRICA.md
├── REFLEXION_INDIVIDUAL.md
├── requirements.txt
└── README.md
```

---

## 12. Evidencia ya generada

- `results/main/equity_drawdown.png`: capital y drawdown frente a baselines.
- `results/main/positions.png`: evolución de la exposición del agente seleccionado.
- `results/main/reward_comparison.png`: comparación de recompensas en validación.
- `results/main/learning_curves.png`: curvas de aprendizaje.
- `results/main/eda.png`: análisis descriptivo del historial.
- `results/main/test_metrics.csv`: métricas por estrategia/semilla.
- `results/main/test_summary.csv`: agregados por estrategia.
- `results/main/selection_frozen.json`: evidencia de la elección realizada antes de prueba.
- `results/VERIFICACION.md`: comprobaciones realizadas sobre entorno, pruebas y reevaluación.

---

## 13. Referencias principales

- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2.ª ed.).
- Mnih, V., et al. (2015). *Human-level control through deep reinforcement learning*. Nature, 518.
- Stable-Baselines3. Documentación de DQN.
- Gymnasium. API de `Env`.
- Binance Spot API. Datos de velas (*klines*).

> **Alcance académico:** los resultados corresponden exclusivamente a la ventana histórica, costos y supuestos descritos. No constituyen asesoría financiera ni evidencia de rentabilidad futura.
