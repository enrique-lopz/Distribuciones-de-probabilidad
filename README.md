# 🏛️ Familias Paramétricas de Probabilidad en la Ciencia Actuarial

> **Recurso Docente Interactivo de Exploración, Teoría y Simulación**  
> Diseñado para estudiantes de la **Licenciatura en Actuaría (3er Semestre)**  
> **Facultad de Matemáticas (FMAT) — Universidad Autónoma de Yucatán (UADY)**

---

## 🎯 Propósito del Proyecto

En el tercer semestre de la Licenciatura en Actuaría, los estudiantes abordan materias fundamentales de **Probabilidad**, **Modelado Estocástico** e **Inferencia Estadística**. El dominio de las familias paramétricas de probabilidad es la piedra angular para:
1. **Modelación de Frecuencia y Severidad de Siniestros** (seguros de vida, gastos médicos, daños y automóviles).
2. **Construcción de Tablas de Mortalidad** y cálculos de supervivencia actuarial.
3. **Gestión de Riesgos Financieros y Solvencia** (cálculo de cuantiles y *Value at Risk* - VaR bajo marcos como Solvencia II / CNSF).
4. **Tarificación de Primas y Deducibles** en esquemas de reaseguro y retención de riesgo.

Esta aplicación web construida con **Streamlit** y **Plotly** ofrece una plataforma interactiva, visual y matemáticamente rigurosa que conecta las definiciones abstractas con su aplicación real en la práctica actuarial.

---

## 📦 Catálogo de las 14 Familias Paramétricas Incluidas

### 🎲 Familias Discretas (6)
1. **Bernoulli ($p$):** Indicador de siniestro individual ($I_i \sim \text{Bernoulli}(q)$), probabilidades anuales de fallecimiento $q_x$ en tablas de vida.
2. **Binomial ($n, p$):** Conteo de siniestros en carteras homogéneas e independientes de $n$ pólizas; subdispersión ($\text{Var} < E$).
3. **Binomial Negativa ($r, p$):** Modelo de frecuencia de reclamaciones en carteras heterogéneas (mezcla Poisson-Gamma) y modelado de **sobredispersión** ($\text{Var} > E$).
4. **Hipergeométrica ($N, K, n$):** Muestreo sin reemplazo en auditorías de pólizas, detección de fraude y factor de corrección por población finita.
5. **Poisson ($\lambda$):** Modelo canónico de frecuencia de siniestros por unidad de tiempo; base de los procesos de Poisson homogéneos.
6. **Uniforme Discreta ($a, b$):** Loterías, juegos de azar y esquemas de muestreo aleatorio equiprobable.

### 📈 Familias Continuas (8)
7. **Uniforme Continua ($a, b$):** Generación de variables aleatorias por el método de transformación inversa ($U \sim \text{Unif}(0,1)$); hipótesis de distribución uniforme de muertes (UDD) a edades fraccionarias.
8. **Exponencial ($\lambda$):** Tiempos entre siniestros, propiedad de falta de memoria y análisis de deducibles ordinarios / excesos de pérdida.
9. **Normal / Gaussiana ($\mu, \sigma^2$):** Teorema del Límite Central (TLC) para el monto agregado de siniestros $S = \sum X_i$; rendimientos logarítmicos financieros.
10. **Gamma ($\alpha, \beta$):** Severidad de pérdidas económicas con asimetría positiva; tiempo acumulado hasta el $k$-ésimo siniestro (Erlang); distribución a priori conjugada de Poisson.
11. **Beta ($\alpha, \beta$):** Severidad porcentual de pérdida ($X \in (0,1)$), tasa de pérdida en caso de incumplimiento (*Loss Given Default* - LGD); distribución a priori conjugada de la Binomial.
12. **Chi-cuadrada ($\chi^2(\nu)$):** Pruebas de bondad de ajuste ($\chi^2$ de Pearson) para calibrar modelos de siniestralidad y tablas de mortalidad.
13. **t de Student ($\nu$):** Rendimientos financieros con **colas pesadas** (*fat tails*); estimación de intervalos de confianza para pérdidas con muestras pequeñas; inexistencia de FGM.
14. **F de Fisher-Snedecor ($d_1, d_2$):** Contraste de igualdad de varianzas entre carteras y análisis de varianza (ANOVA) para factores de tarificación de primas.

---

## 🚀 Guía de Instalación y Ejecución Rápida

### 1. Prerrequisitos
- Python 3.10 o superior (probado y optimizado en Python 3.14).

### 2. Clonar o navegar al directorio del proyecto
```bash
cd "\Pruebas_streamlit"
```

### 3. Activar el entorno virtual
En Windows PowerShell:
```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Instalar dependencias (si no se han instalado)
```bash
pip install -r requirements.txt
```

### 5. Iniciar la aplicación Streamlit
```bash
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador web predeterminado en `http://localhost:8501`.

---

## 🛠️ Estructura del Código

- **`app.py`:** Interfaz de usuario interactiva montada con Streamlit. Contiene los 3 módulos principales:
  - **Módulo 1: Explorador Individual:** Sliders dinámicos, cálculo de métricas en tiempo real ($E[X]$, $\text{Var}(X)$, $\sigma$, $\gamma_1$, $\gamma_2$), pestañas de teoría KaTeX, gráficos Plotly interactivos, calculadora de probabilidades y cuantiles (VaR), y simulador Monte Carlo.
  - **Módulo 2: Laboratorio de Comparación & Convergencias:** Experimentos interactivos de sobredispersión (Poisson vs Binomial Negativa), colas pesadas (Normal vs t-Student), y aproximación de Poisson.
  - **Módulo 3: Tabla Maestra & Glosario Actuarial:** Resumen filtrable de fórmulas analíticas y conceptos clave de la ciencia actuarial.
- **`distributions_catalog.py`:** Catálogo modular con todas las definiciones matemáticas, fórmulas en LaTeX, evaluadores numéricos SciPy y notas actuariales detalladas.
- **`plotting_utils.py`:** Funciones de renderizado en Plotly para PMF/PDF, CDF escalonada/continua, histogramas de Monte Carlo y gráficos comparativos superpuestos.
- **`test_suite.py`:** Pruebas unitarias automatizadas que verifican la integridad numérica y gráfica de las 14 distribuciones.

---

## 🧪 Ejecución de Pruebas Automatizadas
Para verificar que todas las distribuciones y gráficos se evalúan correctamente:
```bash
python test_suite.py
```
Debe reportar:
```text
Verificando 14 familias paramétricas...
...
✅ ¡Todas las pruebas matemáticas y de gráficos pasaron exitosamente!
```

---

## 👥 Créditos Académicos
Desarrollado como recurso didáctico para la comunidad de la **Licenciatura en Actuaría** de la **Facultad de Matemáticas (FMAT) — UADY**.

