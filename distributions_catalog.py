"""
Catálogo completo de Familias Paramétricas de Probabilidad
Diseñado para estudiantes de Licenciatura en Actuaría (3er Semestre) - FMAT, UADY.
Incluye especificaciones teóricas rigurosas, fórmulas en LaTeX, momentos analíticos,
funciones de distribución en SciPy y aplicaciones actuariales del mundo real.
"""

import numpy as np
from scipy import stats, special

DISTRIBUTIONS = {
    # =========================================================================
    # FAMILIAS DISCRETAS
    # =========================================================================
    "bernoulli": {
        "id": "bernoulli",
        "name": "Bernoulli",
        "category": "Discreta",
        "notation": r"X \sim \text{Bernoulli}(p)",
        "params": [
            {
                "id": "p",
                "label": "Probabilidad de éxito (p)",
                "latex": r"p \in (0, 1)",
                "min": 0.01,
                "max": 0.99,
                "default": 0.30,
                "step": 0.01,
                "type": "float",
                "help": "Probabilidad de que ocurra el evento de interés (éxito)."
            }
        ],
        "support": r"R_X = \{0, 1\}",
        "pmf_pdf_title": "Función de Masa de Probabilidad (PMF)",
        "pmf_pdf_latex": r"P(X = x) = p^x (1 - p)^{1 - x} \quad \text{para } x \in \{0, 1\}",
        "cdf_latex": r"""F_X(x) = \begin{cases} 
0, & x < 0 \\ 
1 - p, & 0 \le x < 1 \\ 
1, & x \ge 1 
\end{cases}""",
        "mean_latex": r"E[X] = p",
        "var_latex": r"\text{Var}(X) = p(1 - p)",
        "mgf_latex": r"M_X(t) = (1 - p) + p e^t, \quad \forall t \in \mathbb{R}",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos de Riesgo:**
- **Variable Indicadora de Siniestro:** En el modelo de riesgo individual, sea $I_i$ la variable indicadora de si la póliza $i$ sufre o no un siniestro durante la vigencia anual ($I_i \sim Bernoulli(q_i)$).
- **Tablas de Mortalidad:** La probabilidad de que una persona de edad exacta $x$ fallezca antes de cumplir $x+1$ es $q_x$. La supervivencia o muerte en ese año se modela como un ensayo Bernoulli.
- **Tarificación y Primas:** Base primordial para calcular la prima neta de riesgo básica: $P = E[I_i \cdot Y_i] = q_i \cdot E[Y_i]$, donde $Y_i$ es la severidad condicional al siniestro.
""",
        "scipy_dist": lambda params: stats.bernoulli(params["p"]),
        "theoretical_moments": lambda params: (
            params["p"],
            params["p"] * (1 - params["p"]),
            (1 - 2 * params["p"]) / np.sqrt(params["p"] * (1 - params["p"])),
            (1 - 6 * params["p"] * (1 - params["p"])) / (params["p"] * (1 - params["p"]))
        ),
        "get_x_range": lambda params: (np.array([0, 1]), np.array([0, 1]))
    },

    "binomial": {
        "id": "binomial",
        "name": "Binomial",
        "category": "Discreta",
        "notation": r"X \sim \text{Bin}(n, p)",
        "params": [
            {
                "id": "n",
                "label": "Número de ensayos (n)",
                "latex": r"n \in \{1, 2, \dots\}",
                "min": 1,
                "max": 100,
                "default": 20,
                "step": 1,
                "type": "int",
                "help": "Número total de ensayos de Bernoulli idénticos e independientes."
            },
            {
                "id": "p",
                "label": "Probabilidad de éxito (p)",
                "latex": r"p \in (0, 1)",
                "min": 0.01,
                "max": 0.99,
                "default": 0.25,
                "step": 0.01,
                "type": "float",
                "help": "Probabilidad de éxito en cada ensayo independiente."
            }
        ],
        "support": r"R_X = \{0, 1, 2, \dots, n\}",
        "pmf_pdf_title": "Función de Masa de Probabilidad (PMF)",
        "pmf_pdf_latex": r"P(X = x) = \binom{n}{x} p^x (1 - p)^{n - x} \quad \text{para } x \in \{0, 1, \dots, n\}",
        "cdf_latex": r"F_X(x) = \sum_{k=0}^{\lfloor x \rfloor} \binom{n}{k} p^k (1 - p)^{n - k} = I_{1-p}(n - \lfloor x \rfloor, \lfloor x \rfloor + 1)",
        "mean_latex": r"E[X] = np",
        "var_latex": r"\text{Var}(X) = np(1 - p)",
        "mgf_latex": r"M_X(t) = \left(1 - p + p e^t\right)^n, \quad \forall t \in \mathbb{R}",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos de Riesgo:**
- **Cartera Homogénea de Pólizas:** Si una compañía de seguros suscribe un portafolio de $n$ pólizas idénticas e independientes con probabilidad anual de reclamación $p$, el número total de siniestros de la cartera sigue $X \sim Bin(n, p)$.
- **Subdispersión respecto a Poisson:** Observa que $Var(X) = np(1-p) < E[X] = np$. La Binomial modela conteos con varianza menor a la media.
- **Convergencias Clave:**
  - Si $n \to \infty$ y $p \to 0$ tal que $\lambda = np$ es constante, $X \xrightarrow{d} Poisson(\lambda)$ (Ley de Sucesos Raros).
  - Si $n$ es grande y $np, n(1-p) \ge 5$, $X \approx \mathcal{N}(np, np(1-p))$ (Teorema de De Moivre-Laplace).
""",
        "scipy_dist": lambda params: stats.binom(params["n"], params["p"]),
        "theoretical_moments": lambda params: (
            params["n"] * params["p"],
            params["n"] * params["p"] * (1 - params["p"]),
            (1 - 2 * params["p"]) / np.sqrt(params["n"] * params["p"] * (1 - params["p"])),
            (1 - 6 * params["p"] * (1 - params["p"])) / (params["n"] * params["p"] * (1 - params["p"]))
        ),
        "get_x_range": lambda params: (
            np.arange(0, params["n"] + 1),
            np.arange(0, params["n"] + 1)
        )
    },

    "geometric": {
        "id": "geometric",
        "name": "Geométrica",
        "category": "Discreta",
        "notation": r"X \sim \text{Geom}(p)",
        "params": [
            {
                "id": "p",
                "label": "Probabilidad de éxito (p)",
                "latex": r"p \in (0, 1)",
                "min": 0.01,
                "max": 0.99,
                "default": 0.20,
                "step": 0.01,
                "type": "float",
                "help": "Probabilidad constante de éxito en cada ensayo independiente de Bernoulli."
            }
        ],
        "support": r"R_X = \{1, 2, 3, \dots\} \quad (\text{ensayos hasta el 1er éxito})",
        "pmf_pdf_title": "Función de Masa de Probabilidad (PMF)",
        "pmf_pdf_latex": r"P(X = x) = (1 - p)^{x - 1} p \quad \text{para } x \in \{1, 2, 3, \dots\}",
        "cdf_latex": r"""F_X(x) = \begin{cases} 
0, & x < 1 \\ 
1 - (1 - p)^{\lfloor x \rfloor}, & x \ge 1 
\end{cases}""",
        "mean_latex": r"E[X] = \frac{1}{p}",
        "var_latex": r"\text{Var}(X) = \frac{1 - p}{p^2}",
        "mgf_latex": r"M_X(t) = \frac{p e^t}{1 - (1 - p)e^t}, \quad \text{para } t < -\ln(1 - p)",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos de Riesgo:**
- **Tiempos de Espera hasta la Primera Reclamación:** Modela el número de periodos (días, meses, años) o pólizas evaluadas hasta observar el primer siniestro en una cartera.
- **Propiedad de Pérdida de Memoria (*Memoryless Property*):** Es la **única distribución discreta** que carece de memoria:
  $$P(X > s + t \mid X > s) = P(X > t), \quad \forall s, t \in \mathbb{N}$$
  En seguros y teoría de fiabilidad, implica que la antigüedad o el tiempo transcurrido sin registrar siniestros no altera la probabilidad de ocurrencia futura (ausencia de envejecimiento del riesgo).
- **Periodo de Retorno de Riesgos Catastróficos:** En reaseguro y tarificación de eventos extremos (huracanes categoría 5, sismos mayores), si un evento tiene probabilidad anual de ocurrencia $p$, el tiempo (en años) hasta su primera ocurrencia sigue $X \sim \text{Geom}(p)$ con valor esperado $E[X] = 1/p$, conocido formalmente como el *periodo de retorno* ($T = 1/p$ años).
- **Caso Particular de la Binomial Negativa y Clase de Panjer:** Es exactamente $\text{NB}(r=1, p)$. Pertenece a la clase $(a, b, 0)$ de Panjer con $a = 1 - p$ y $b = 0$, permitiendo el uso eficiente de recursiones de Panjer para calcular la distribución de pérdidas agregadas en carteras aseguradas.
- **Parametrización Alternativa:** Si se define sobre el número de fracasos antes del primer éxito $Y = X - 1 \in \{0, 1, 2, \dots\}$, entonces $P(Y = y) = (1 - p)^y p$, con $E[Y] = \frac{1 - p}{p}$ y $\text{Var}(Y) = \frac{1 - p}{p^2}$.
""",
        "scipy_dist": lambda params: stats.geom(params["p"]),
        "theoretical_moments": lambda params: (
            1.0 / params["p"],
            (1.0 - params["p"]) / (params["p"] ** 2),
            (2.0 - params["p"]) / np.sqrt(1.0 - params["p"]),
            6.0 + (params["p"] ** 2) / (1.0 - params["p"])
        ),
        "get_x_range": lambda params: (
            np.arange(1, max(15, int(stats.geom.ppf(0.999, params["p"]))) + 2),
            np.arange(1, max(15, int(stats.geom.ppf(0.999, params["p"]))) + 2)
        )
    },

    "nbinom": {
        "id": "nbinom",
        "name": "Binomial Negativa",
        "category": "Discreta",
        "notation": r"X \sim \text{NB}(r, p)",
        "params": [
            {
                "id": "r",
                "label": "Número de éxitos requeridos (r)",
                "latex": r"r > 0 \text{ (generalizado o } r \in \mathbb{N}\text{)}",
                "min": 1,
                "max": 50,
                "default": 5,
                "step": 1,
                "type": "int",
                "help": "Número de éxitos requeridos hasta detener los ensayos."
            },
            {
                "id": "p",
                "label": "Probabilidad de éxito (p)",
                "latex": r"p \in (0, 1)",
                "min": 0.05,
                "max": 0.95,
                "default": 0.40,
                "step": 0.05,
                "type": "float",
                "help": "Probabilidad de éxito en cada ensayo independiente."
            }
        ],
        "support": r"R_X = \{0, 1, 2, \dots\} \quad (\text{número de fracasos})",
        "pmf_pdf_title": "Función de Masa de Probabilidad (PMF)",
        "pmf_pdf_latex": r"P(X = k) = \binom{k + r - 1}{k} p^r (1 - p)^k = \frac{\Gamma(k + r)}{k! \, \Gamma(r)} p^r (1 - p)^k",
        "cdf_latex": r"F_X(x) = I_p(r, \lfloor x \rfloor + 1)",
        "mean_latex": r"E[X] = \frac{r(1 - p)}{p}",
        "var_latex": r"\text{Var}(X) = \frac{r(1 - p)}{p^2} = E[X] \left(1 + \frac{1 - p}{p}\right)",
        "mgf_latex": r"M_X(t) = \left(\frac{p}{1 - (1 - p)e^t}\right)^r, \quad \text{para } t < -\ln(1 - p)",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos de Riesgo:**
- **La Reina de las Distribuciones de Frecuencia:** En seguros de automóviles y gastos médicos mayores, la distribución Poisson suele fallar porque los asegurados tienen distintos perfiles de riesgo (heterogeneidad de la cartera).
- **Mezcla Poisson-Gamma:** Si cada asegurado tiene una tasa de siniestros $\Lambda \sim \text{Gamma}(\alpha, \beta)$ y condicionalmente $N | \Lambda = \lambda \sim \text{Poisson}(\lambda)$, entonces la distribución marginal no condicionada de siniestros $N$ es exactamente una **Binomial Negativa**.
- **Sobredispersión Actuarial:** $\text{Var}(X) > E[X]$. Permite modelar carteras donde unos pocos asegurados sufren muchos siniestros.
""",
        "scipy_dist": lambda params: stats.nbinom(params["r"], params["p"]),
        "theoretical_moments": lambda params: (
            params["r"] * (1 - params["p"]) / params["p"],
            params["r"] * (1 - params["p"]) / (params["p"] ** 2),
            (2 - params["p"]) / np.sqrt(params["r"] * (1 - params["p"])),
            6 / params["r"] + (params["p"] ** 2) / (params["r"] * (1 - params["p"]))
        ),
        "get_x_range": lambda params: (
            np.arange(0, int(stats.nbinom.ppf(0.999, params["r"], params["p"])) + 3),
            np.arange(0, int(stats.nbinom.ppf(0.999, params["r"], params["p"])) + 3)
        )
    },

    "hypergeom": {
        "id": "hypergeom",
        "name": "Hipergeométrica",
        "category": "Discreta",
        "notation": r"X \sim \text{Hypergeom}(N, K, n)",
        "params": [
            {
                "id": "N",
                "label": "Tamaño de la población (N)",
                "latex": r"N \in \{1, 2, \dots\}",
                "min": 10,
                "max": 200,
                "default": 100,
                "step": 5,
                "type": "int",
                "help": "Número total de elementos en la población o cartera cerrada."
            },
            {
                "id": "K",
                "label": "Éxitos en la población (K)",
                "latex": r"K \in \{0, 1, \dots, N\}",
                "min": 1,
                "max": 100,
                "default": 30,
                "step": 1,
                "type": "int",
                "help": "Número total de pólizas siniestradas o con fraude en la población."
            },
            {
                "id": "n",
                "label": "Tamaño de la muestra (n)",
                "latex": r"n \in \{1, 2, \dots, N\}",
                "min": 1,
                "max": 100,
                "default": 20,
                "step": 1,
                "type": "int",
                "help": "Número de pólizas auditadas sin reemplazo."
            }
        ],
        "support": r"R_X = \{\max(0, n - (N - K)), \dots, \min(n, K)\}",
        "pmf_pdf_title": "Función de Masa de Probabilidad (PMF)",
        "pmf_pdf_latex": r"P(X = x) = \frac{\binom{K}{x} \binom{N - K}{n - x}}{\binom{N}{n}}",
        "cdf_latex": r"F_X(x) = \sum_{k=0}^{\lfloor x \rfloor} \frac{\binom{K}{k} \binom{N - K}{n - k}}{\binom{N}{n}}",
        "mean_latex": r"E[X] = n \frac{K}{N}",
        "var_latex": r"\text{Var}(X) = n \frac{K}{N} \left(1 - \frac{K}{N}\right) \left(\frac{N - n}{N - 1}\right)",
        "mgf_latex": r"M_X(t) = \frac{\binom{N-K}{n}}{\binom{N}{n}} \,_2F_1\left(-n, -K; N - K - n + 1; e^t\right)",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Auditoría de Seguros:**
- **Auditoría de Reservas y Fraude:** Cuando la Comisión Nacional de Seguros y Fianzas (CNSF) o el actuario auditor selecciona una muestra aleatoria de $n$ expedientes **sin reemplazo** de una cartera finita de $N$ pólizas para detectar errores de reserva o reclamaciones fraudulentas.
- **Factor de Corrección por Población Finita:** Observa el término $\frac{N - n}{N - 1} < 1$. Muestra cómo el muestreo sin reemplazo reduce la varianza de la estimación respecto a la Binomial. Si $N \to \infty$, $\text{Hypergeom}(N, K, n) \to \text{Bin}(n, p=K/N)$.
""",
        "scipy_dist": lambda params: stats.hypergeom(M=params["N"], n=params["K"], N=params["n"]),
        "theoretical_moments": lambda params: (
            params["n"] * (params["K"] / params["N"]),
            params["n"] * (params["K"] / params["N"]) * (1 - params["K"] / params["N"]) * ((params["N"] - params["n"]) / (params["N"] - 1)),
            ((params["N"] - 2 * params["K"]) * np.sqrt(params["N"] - 1) * (params["N"] - 2 * params["n"])) /
            (np.sqrt(params["n"] * params["K"] * (params["N"] - params["K"]) * (params["N"] - params["n"])) * (params["N"] - 2)) if params["N"] > 2 else 0.0,
            0.0 # kurtosis simplificada
        ),
        "get_x_range": lambda params: (
            np.arange(max(0, params["n"] - (params["N"] - params["K"])), min(params["n"], params["K"]) + 1),
            np.arange(max(0, params["n"] - (params["N"] - params["K"])), min(params["n"], params["K"]) + 1)
        )
    },

    "poisson": {
        "id": "poisson",
        "name": "Poisson",
        "category": "Discreta",
        "notation": r"X \sim \text{Poisson}(\lambda)",
        "params": [
            {
                "id": "lambda",
                "label": "Tasa promedio de ocurrencia (λ)",
                "latex": r"\lambda > 0",
                "min": 0.1,
                "max": 30.0,
                "default": 4.0,
                "step": 0.2,
                "type": "float",
                "help": "Número esperado de eventos por periodo o intervalo de tiempo."
            }
        ],
        "support": r"R_X = \{0, 1, 2, \dots\}",
        "pmf_pdf_title": "Función de Masa de Probabilidad (PMF)",
        "pmf_pdf_latex": r"P(X = x) = \frac{\lambda^x e^{-\lambda}}{x!} \quad \text{para } x \in \{0, 1, 2, \dots\}",
        "cdf_latex": r"F_X(x) = e^{-\lambda} \sum_{k=0}^{\lfloor x \rfloor} \frac{\lambda^k}{k!} = \frac{\Gamma(\lfloor x \rfloor + 1, \lambda)}{\lfloor x \rfloor!}",
        "mean_latex": r"E[X] = \lambda",
        "var_latex": r"\text{Var}(X) = \lambda",
        "mgf_latex": r"M_X(t) = \exp\left(\lambda (e^t - 1)\right), \quad \forall t \in \mathbb{R}",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos de Riesgo:**
- **Modelo Estándar de Frecuencia de Siniestros:** Es la distribución por excelencia para modelar el número de reclamaciones $N$ en una cartera durante un periodo (por ejemplo, número de accidentes de auto al año).
- **Proceso de Poisson Homogéneo:** Si los siniestros ocurren de manera independiente a una tasa constante $\lambda$, el número de siniestros en $[0, t]$ es $\text{Poisson}(\lambda t)$, y los tiempos entre llegadas consecutivas son exponenciales independientes $\text{Exp}(\lambda)$.
- **Equidispersión:** $E[X] = \text{Var}(X) = \lambda$. En la práctica actuarial, cuando los datos reales muestran $\text{Var} > \text{Media}$, se rechaza Poisson a favor de la Binomial Negativa.
""",
        "scipy_dist": lambda params: stats.poisson(mu=params["lambda"]),
        "theoretical_moments": lambda params: (
            params["lambda"],
            params["lambda"],
            1.0 / np.sqrt(params["lambda"]),
            1.0 / params["lambda"]
        ),
        "get_x_range": lambda params: (
            np.arange(0, int(stats.poisson.ppf(0.9995, params["lambda"])) + 3),
            np.arange(0, int(stats.poisson.ppf(0.9995, params["lambda"])) + 3)
        )
    },

    "discrete_uniform": {
        "id": "discrete_uniform",
        "name": "Uniforme Discreta",
        "category": "Discreta",
        "notation": r"X \sim \text{UnifDiscreta}(a, b)",
        "params": [
            {
                "id": "a",
                "label": "Límite inferior (a)",
                "latex": r"a \in \mathbb{Z}",
                "min": 1,
                "max": 50,
                "default": 1,
                "step": 1,
                "type": "int",
                "help": "Valor entero mínimo del soporte."
            },
            {
                "id": "b",
                "label": "Límite superior (b)",
                "latex": r"b \in \mathbb{Z}, \, b \ge a",
                "min": 2,
                "max": 60,
                "default": 10,
                "step": 1,
                "type": "int",
                "help": "Valor entero máximo del soporte."
            }
        ],
        "support": r"R_X = \{a, a + 1, \dots, b\}, \quad k = b - a + 1",
        "pmf_pdf_title": "Función de Masa de Probabilidad (PMF)",
        "pmf_pdf_latex": r"P(X = x) = \frac{1}{b - a + 1} \quad \text{para } x \in \{a, a + 1, \dots, b\}",
        "cdf_latex": r"""F_X(x) = \begin{cases} 
0, & x < a \\ 
\frac{\lfloor x \rfloor - a + 1}{b - a + 1}, & a \le x \le b \\ 
1, & x > b 
\end{cases}""",
        "mean_latex": r"E[X] = \frac{a + b}{2}",
        "var_latex": r"\text{Var}(X) = \frac{(b - a + 1)^2 - 1}{12}",
        "mgf_latex": r"M_X(t) = \frac{e^{at} - e^{(b + 1)t}}{(b - a + 1)(1 - e^t)}, \quad t \neq 0; \quad M_X(0) = 1",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Simulación:**
- **Loterías y Juegos de Azar:** Modelación de premios y productos de apuestas (por ejemplo, ruleta, dados, melate), donde cada resultado equiprobable tiene una consecuencia financiera fija.
- **Muestreo Aleatorio Simple:** Base para seleccionar expedientes o pólizas al azar en simulaciones estocásticas y validación de bases de datos actuariales.
""",
        "scipy_dist": lambda params: stats.randint(low=params["a"], high=params["b"] + 1),
        "theoretical_moments": lambda params: (
            (params["a"] + params["b"]) / 2.0,
            ((params["b"] - params["a"] + 1) ** 2 - 1) / 12.0,
            0.0,
            -1.2 * ((params["b"] - params["a"] + 1) ** 2 + 1) / ((params["b"] - params["a"] + 1) ** 2 - 1) if (params["b"] - params["a"] + 1) > 1 else 0.0
        ),
        "get_x_range": lambda params: (
            np.arange(params["a"], params["b"] + 1),
            np.arange(params["a"], params["b"] + 1)
        )
    },

    # =========================================================================
    # FAMILIAS CONTINUAS
    # =========================================================================
    "continuous_uniform": {
        "id": "continuous_uniform",
        "name": "Uniforme Continua",
        "category": "Continua",
        "notation": r"X \sim \text{Unif}(a, b)",
        "params": [
            {
                "id": "a",
                "label": "Límite inferior (a)",
                "latex": r"a \in \mathbb{R}",
                "min": -50.0,
                "max": 50.0,
                "default": 0.0,
                "step": 1.0,
                "type": "float",
                "help": "Extremo inferior del intervalo continuo."
            },
            {
                "id": "b",
                "label": "Límite superior (b)",
                "latex": r"b \in \mathbb{R}, \, b > a",
                "min": -40.0,
                "max": 100.0,
                "default": 10.0,
                "step": 1.0,
                "type": "float",
                "help": "Extremo superior del intervalo continuo."
            }
        ],
        "support": r"R_X = [a, b]",
        "pmf_pdf_title": "Función de Densidad de Probabilidad (PDF)",
        "pmf_pdf_latex": r"f_X(x) = \frac{1}{b - a} \quad \text{para } x \in [a, b]",
        "cdf_latex": r"""F_X(x) = \begin{cases} 
0, & x < a \\ 
\frac{x - a}{b - a}, & a \le x \le b \\ 
1, & x > b 
\end{cases}""",
        "mean_latex": r"E[X] = \frac{a + b}{2}",
        "var_latex": r"\text{Var}(X) = \frac{(b - a)^2}{12}",
        "mgf_latex": r"M_X(t) = \frac{e^{tb} - e^{ta}}{t(b - a)}, \quad t \neq 0; \quad M_X(0) = 1",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Simulación Estocástica:**
- **Pilar del Método de Transformación Inversa:** Si $U \sim \text{Unif}(0, 1)$, entonces $X = F^{-1}(U)$ tiene función de distribución acumulada $F$. Esto permite simular siniestros de cualquier distribución continua en computadoras.
- **Hipótesis de Fracción de Año:** En matemáticas actuariales de vida, para edades fraccionarias se asume con frecuencia la hipótesis UDD (*Uniform Distribution of Deaths*), donde el tiempo fraccionario de fallecimiento dentro del año se modela uniforme.
""",
        "scipy_dist": lambda params: stats.uniform(loc=params["a"], scale=params["b"] - params["a"]),
        "theoretical_moments": lambda params: (
            (params["a"] + params["b"]) / 2.0,
            ((params["b"] - params["a"]) ** 2) / 12.0,
            0.0,
            -1.2
        ),
        "get_x_range": lambda params: (
            np.linspace(params["a"] - 0.2 * (params["b"] - params["a"]), params["b"] + 0.2 * (params["b"] - params["a"]), 400),
            np.linspace(params["a"] - 0.2 * (params["b"] - params["a"]), params["b"] + 0.2 * (params["b"] - params["a"]), 400)
        )
    },

    "exponential": {
        "id": "exponential",
        "name": "Exponencial",
        "category": "Continua",
        "notation": r"X \sim \text{Exp}(\lambda)",
        "params": [
            {
                "id": "lambda",
                "label": "Tasa de riesgo / parámetro de escala (λ)",
                "latex": r"\lambda > 0 \quad (\text{media } \beta = 1/\lambda)",
                "min": 0.05,
                "max": 5.0,
                "default": 0.5,
                "step": 0.05,
                "type": "float",
                "help": "Tasa instantánea de ocurrencia o recíproco de la pérdida media esperada."
            }
        ],
        "support": r"R_X = [0, \infty)",
        "pmf_pdf_title": "Función de Densidad de Probabilidad (PDF)",
        "pmf_pdf_latex": r"f_X(x) = \lambda e^{-\lambda x} \quad \text{para } x \ge 0",
        "cdf_latex": r"F_X(x) = 1 - e^{-\lambda x} \quad \text{para } x \ge 0",
        "mean_latex": r"E[X] = \frac{1}{\lambda}",
        "var_latex": r"\text{Var}(X) = \frac{1}{\lambda^2}",
        "mgf_latex": r"M_X(t) = \frac{\lambda}{\lambda - t} = \left(1 - \frac{t}{\lambda}\right)^{-1}, \quad \text{para } t < \lambda",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos de Riesgo:**
- **Propiedad de Falta de Memoria:** $P(X > s + t \mid X > s) = P(X > t)$. En seguros, implica que el tiempo que un equipo o póliza ya ha sobrevivido no altera la distribución del tiempo de vida remanente.
- **Función de Supervivencia y Fuerza de Mortalidad:** $S(x) = e^{-\lambda x}$. La fuerza de mortalidad (hazard rate) $\mu(x) = \frac{f(x)}{S(x)} = \lambda$ es constante.
- **Deducibles y Reaseguro de Exceso de Pérdida:** Debido a la falta de memoria, el monto de siniestro que excede un deducible $d$ condicional a que $X > d$ sigue teniendo idéntica distribución exponencial: $X - d \mid X > d \sim \text{Exp}(\lambda)$.
""",
        "scipy_dist": lambda params: stats.expon(scale=1.0 / params["lambda"]),
        "theoretical_moments": lambda params: (
            1.0 / params["lambda"],
            1.0 / (params["lambda"] ** 2),
            2.0,
            6.0
        ),
        "get_x_range": lambda params: (
            np.linspace(0, stats.expon.ppf(0.999, scale=1.0 / params["lambda"]), 400),
            np.linspace(0, stats.expon.ppf(0.999, scale=1.0 / params["lambda"]), 400)
        )
    },

    "normal": {
        "id": "normal",
        "name": "Normal (Gaussiana)",
        "category": "Continua",
        "notation": r"X \sim \mathcal{N}(\mu, \sigma^2)",
        "params": [
            {
                "id": "mu",
                "label": "Media (μ)",
                "latex": r"\mu \in \mathbb{R}",
                "min": -50.0,
                "max": 50.0,
                "default": 0.0,
                "step": 1.0,
                "type": "float",
                "help": "Centro o valor esperado de la distribución."
            },
            {
                "id": "sigma",
                "label": "Desviación estándar (σ)",
                "latex": r"\sigma > 0",
                "min": 0.1,
                "max": 20.0,
                "default": 2.0,
                "step": 0.1,
                "type": "float",
                "help": "Dispersión o volatilidad de la variable."
            }
        ],
        "support": r"R_X = (-\infty, \infty)",
        "pmf_pdf_title": "Función de Densidad de Probabilidad (PDF)",
        "pmf_pdf_latex": r"f_X(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)",
        "cdf_latex": r"F_X(x) = \Phi\left(\frac{x - \mu}{\sigma}\right) = \frac{1}{2}\left[1 + \text{erf}\left(\frac{x - \mu}{\sigma \sqrt{2}}\right)\right]",
        "mean_latex": r"E[X] = \mu",
        "var_latex": r"\text{Var}(X) = \sigma^2",
        "mgf_latex": r"M_X(t) = \exp\left(\mu t + \frac{1}{2}\sigma^2 t^2\right), \quad \forall t \in \mathbb{R}",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos de Riesgo:**
- **Teorema del Límite Central (TLC):** La suma total de siniestros de una cartera grande de pólizas independientes $S_n = \sum_{i=1}^n X_i$ converge asintóticamente a una distribución normal.
- **Principio de Varianza y Margen de Solvencia:** Tarificación con recargo de seguridad: $\text{Prima} = E[S] + \theta \sqrt{\text{Var}(S)} = \mu + z_{1-\alpha}\sigma$.
- **Finanzas Cuantitativas:** Base del movimiento browniano geométrico y la fórmula de Black-Scholes para opciones financieras (rendimientos logarítmicos normales).
""",
        "scipy_dist": lambda params: stats.norm(loc=params["mu"], scale=params["sigma"]),
        "theoretical_moments": lambda params: (
            params["mu"],
            params["sigma"] ** 2,
            0.0,
            0.0
        ),
        "get_x_range": lambda params: (
            np.linspace(params["mu"] - 4 * params["sigma"], params["mu"] + 4 * params["sigma"], 400),
            np.linspace(params["mu"] - 4 * params["sigma"], params["mu"] + 4 * params["sigma"], 400)
        )
    },

    "gamma": {
        "id": "gamma",
        "name": "Gamma",
        "category": "Continua",
        "notation": r"X \sim \text{Gamma}(\alpha, \beta)",
        "params": [
            {
                "id": "alpha",
                "label": "Parámetro de forma (α)",
                "latex": r"\alpha > 0",
                "min": 0.2,
                "max": 20.0,
                "default": 3.0,
                "step": 0.2,
                "type": "float",
                "help": "Controla la forma y asimetría de la curva de severidad."
            },
            {
                "id": "beta",
                "label": "Parámetro de tasa (β)",
                "latex": r"\beta > 0 \quad (\text{escala } \theta = 1/\beta)",
                "min": 0.1,
                "max": 10.0,
                "default": 1.0,
                "step": 0.1,
                "type": "float",
                "help": "Parámetro de tasa (inverso de la escala)."
            }
        ],
        "support": r"R_X = (0, \infty)",
        "pmf_pdf_title": "Función de Densidad de Probabilidad (PDF)",
        "pmf_pdf_latex": r"f_X(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x} = \frac{1}{\Gamma(\alpha)\theta^\alpha} x^{\alpha - 1} e^{-x/\theta}",
        "cdf_latex": r"F_X(x) = \frac{\gamma(\alpha, \beta x)}{\Gamma(\alpha)} = P(\alpha, \beta x) \quad (\text{gamma incompleta regularizada})",
        "mean_latex": r"E[X] = \frac{\alpha}{\beta} = \alpha \theta",
        "var_latex": r"\text{Var}(X) = \frac{\alpha}{\beta^2} = \alpha \theta^2",
        "mgf_latex": r"M_X(t) = \left(1 - \frac{t}{\beta}\right)^{-\alpha}, \quad \text{para } t < \beta",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos de Riesgo:**
- **Modelo de Severidad de Siniestros:** Ajusta de manera flexible pérdidas positivas sesgadas a la derecha en seguros de daños (daños a terceros, colisión de autos, incendios medianos).
- **Suma de Exponenciales (Erlang):** Si $\alpha = k \in \mathbb{N}$, el tiempo hasta el $k$-ésimo siniestro en un proceso Poisson es $\text{Gamma}(k, \lambda)$.
- **Conjugada a Priori en Credibilidad Bayesiana:** Es la distribución conjugada a priori para el parámetro de Poisson en modelos de credibilidad de Bühlmann-Straub.
""",
        "scipy_dist": lambda params: stats.gamma(a=params["alpha"], scale=1.0 / params["beta"]),
        "theoretical_moments": lambda params: (
            params["alpha"] / params["beta"],
            params["alpha"] / (params["beta"] ** 2),
            2.0 / np.sqrt(params["alpha"]),
            6.0 / params["alpha"]
        ),
        "get_x_range": lambda params: (
            np.linspace(0.001, stats.gamma.ppf(0.999, a=params["alpha"], scale=1.0 / params["beta"]), 400),
            np.linspace(0.001, stats.gamma.ppf(0.999, a=params["alpha"], scale=1.0 / params["beta"]), 400)
        )
    },

    "beta": {
        "id": "beta",
        "name": "Beta",
        "category": "Continua",
        "notation": r"X \sim \text{Beta}(\alpha, \beta)",
        "params": [
            {
                "id": "alpha",
                "label": "Parámetro de forma α",
                "latex": r"\alpha > 0",
                "min": 0.2,
                "max": 15.0,
                "default": 2.0,
                "step": 0.2,
                "type": "float",
                "help": "Primer parámetro de forma."
            },
            {
                "id": "beta",
                "label": "Parámetro de forma β",
                "latex": r"\beta > 0",
                "min": 0.2,
                "max": 15.0,
                "default": 5.0,
                "step": 0.2,
                "type": "float",
                "help": "Segundo parámetro de forma."
            }
        ],
        "support": r"R_X = (0, 1)",
        "pmf_pdf_title": "Función de Densidad de Probabilidad (PDF)",
        "pmf_pdf_latex": r"f_X(x) = \frac{1}{\text{B}(\alpha, \beta)} x^{\alpha - 1} (1 - x)^{\beta - 1} = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}",
        "cdf_latex": r"F_X(x) = I_x(\alpha, \beta) = \frac{\text{B}(x; \alpha, \beta)}{\text{B}(\alpha, \beta)} \quad (\text{beta incompleta regularizada})",
        "mean_latex": r"E[X] = \frac{\alpha}{\alpha + \beta}",
        "var_latex": r"\text{Var}(X) = \frac{\alpha \beta}{(\alpha + \beta)^2 (\alpha + \beta + 1)}",
        "mgf_latex": r"M_X(t) = 1 + \sum_{k=1}^\infty \left(\prod_{r=0}^{k-1} \frac{\alpha + r}{\alpha + \beta + r}\right) \frac{t^k}{k!} = \,_1F_1(\alpha; \alpha + \beta; t)",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos de Riesgo:**
- **Severidad Proporcional o Porcentaje de Pérdida:** Modela el porcentaje de destrucción de un bien o la tasa de recuperación en créditos y bonos bancarios (Loss Given Default - LGD en Basilea III / Solvencia II).
- **Distribución a Priori Conjugada de la Binomial:** En inferencia bayesiana, modela la incertidumbre sobre la probabilidad de siniestro $p$.
- **Flexibilidad de Formas:** Dependiendo de $(\alpha, \beta)$ puede ser simétrica en campana ($\alpha=\beta>1$), uniforme ($\alpha=\beta=1$), en forma de U ($\alpha,\beta<1$) o asimétrica hacia cualquier extremo.
""",
        "scipy_dist": lambda params: stats.beta(a=params["alpha"], b=params["beta"]),
        "theoretical_moments": lambda params: (
            params["alpha"] / (params["alpha"] + params["beta"]),
            (params["alpha"] * params["beta"]) / (((params["alpha"] + params["beta"]) ** 2) * (params["alpha"] + params["beta"] + 1)),
            (2 * (params["beta"] - params["alpha"]) * np.sqrt(params["alpha"] + params["beta"] + 1)) / ((params["alpha"] + params["beta"] + 2) * np.sqrt(params["alpha"] * params["beta"])),
            (6 * ((params["alpha"] - params["beta"]) ** 2 * (params["alpha"] + params["beta"] + 1) - params["alpha"] * params["beta"] * (params["alpha"] + params["beta"] + 2))) /
            (params["alpha"] * params["beta"] * (params["alpha"] + params["beta"] + 2) * (params["alpha"] + params["beta"] + 3))
        ),
        "get_x_range": lambda params: (
            np.linspace(0.001, 0.999, 400),
            np.linspace(0.001, 0.999, 400)
        )
    },

    "chi2": {
        "id": "chi2",
        "name": "Chi-cuadrada (χ²)",
        "category": "Continua",
        "notation": r"X \sim \chi^2(\nu)",
        "params": [
            {
                "id": "df",
                "label": "Grados de libertad (ν)",
                "latex": r"\nu \in \{1, 2, \dots\}",
                "min": 1,
                "max": 50,
                "default": 4,
                "step": 1,
                "type": "int",
                "help": "Número de variables normales estándar independientes elevadas al cuadrado."
            }
        ],
        "support": r"R_X = (0, \infty)",
        "pmf_pdf_title": "Función de Densidad de Probabilidad (PDF)",
        "pmf_pdf_latex": r"f_X(x) = \frac{1}{2^{\nu/2} \Gamma(\nu/2)} x^{\nu/2 - 1} e^{-x/2} \quad \text{para } x > 0",
        "cdf_latex": r"F_X(x) = \frac{\gamma(\nu/2, x/2)}{\Gamma(\nu/2)} = P(\nu/2, x/2)",
        "mean_latex": r"E[X] = \nu",
        "var_latex": r"\text{Var}(X) = 2\nu",
        "mgf_latex": r"M_X(t) = (1 - 2t)^{-\nu/2}, \quad \text{para } t < 1/2",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Estadística:**
- **Pruebas de Bondad de Ajuste:** La estadística $\chi^2$ de Pearson es el instrumento fundamental para verificar si las tablas de mortalidad observadas corresponden a la ley de Gompertz-Makeham o si los siniestros de una cartera siguen una distribución Poisson.
- **Relación con la Normal:** Si $Z_1, \dots, Z_\nu \stackrel{iid}{\sim} \mathcal{N}(0, 1)$, entonces $\sum_{i=1}^\nu Z_i^2 \sim \chi^2(\nu)$.
- **Estimación de Varianzas:** La varianza muestral $S^2$ en poblaciones normales satisface $\frac{(n-1)S^2}{\sigma^2} \sim \chi^2(n-1)$.
""",
        "scipy_dist": lambda params: stats.chi2(df=params["df"]),
        "theoretical_moments": lambda params: (
            float(params["df"]),
            2.0 * params["df"],
            np.sqrt(8.0 / params["df"]),
            12.0 / params["df"]
        ),
        "get_x_range": lambda params: (
            np.linspace(0.01, stats.chi2.ppf(0.999, df=params["df"]), 400),
            np.linspace(0.01, stats.chi2.ppf(0.999, df=params["df"]), 400)
        )
    },

    "t_student": {
        "id": "t_student",
        "name": "t de Student",
        "category": "Continua",
        "notation": r"X \sim t(\nu)",
        "params": [
            {
                "id": "df",
                "label": "Grados de libertad (ν)",
                "latex": r"\nu > 0",
                "min": 1,
                "max": 60,
                "default": 5,
                "step": 1,
                "type": "int",
                "help": "Grados de libertad que regulan el grosor de las colas."
            }
        ],
        "support": r"R_X = (-\infty, \infty)",
        "pmf_pdf_title": "Función de Densidad de Probabilidad (PDF)",
        "pmf_pdf_latex": r"f_X(x) = \frac{\Gamma\left(\frac{\nu + 1}{2}\right)}{\sqrt{\nu \pi} \, \Gamma\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-\frac{\nu + 1}{2}}",
        "cdf_latex": r"F_X(x) = \frac{1}{2} + x \, \Gamma\left(\frac{\nu+1}{2}\right) \frac{\,_2F_1\left(\frac{1}{2}, \frac{\nu+1}{2}; \frac{3}{2}; -\frac{x^2}{\nu}\right)}{\sqrt{\pi \nu} \, \Gamma(\nu/2)}",
        "mean_latex": r"E[X] = 0 \quad (\text{para } \nu > 1; \text{indefinido para } \nu \le 1)",
        "var_latex": r"\text{Var}(X) = \frac{\nu}{\nu - 2} \quad (\text{para } \nu > 2; \, \infty \text{ si } 1 < \nu \le 2)",
        "mgf_latex": r"M_X(t) \text{ no existe para ningún } t \neq 0 \quad (\text{distribución de colas pesadas})",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Riesgos Financieros:**
- **Distribución de Colas Pesadas (*Fat Tails*):** Tiene mayor probabilidad de pérdidas extremas (cisnes negros) que la distribución Normal.
- **Riesgo de Mercado y VaR:** El cálculo de Valor en Riesgo (VaR) bajo supuestos normales subestima severamente el riesgo financiero de catástrofe; la $t$ de Student captura adecuadamente la leptocurtosis de los activos.
- **Inexistencia de FGM:** Refleja matemáticamente el decaimiento potencial (no exponencial) en las colas. Para $\nu \le 1$ la media no converge (ej. Cauchy cuando $\nu=1$), y para $\nu \le 2$ la varianza es infinita.
- **Inferencia en Muestras Pequeñas:** Base de los intervalos de confianza para la media de pérdidas cuando la varianza es desconocida: $T = \frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t(n-1)$.
""",
        "scipy_dist": lambda params: stats.t(df=params["df"]),
        "theoretical_moments": lambda params: (
            0.0 if params["df"] > 1 else np.nan,
            params["df"] / (params["df"] - 2.0) if params["df"] > 2 else (np.inf if params["df"] > 1 else np.nan),
            0.0 if params["df"] > 3 else np.nan,
            6.0 / (params["df"] - 4.0) if params["df"] > 4 else (np.inf if params["df"] > 2 else np.nan)
        ),
        "get_x_range": lambda params: (
            np.linspace(stats.t.ppf(0.001, df=params["df"]), stats.t.ppf(0.999, df=params["df"]), 400),
            np.linspace(stats.t.ppf(0.001, df=params["df"]), stats.t.ppf(0.999, df=params["df"]), 400)
        )
    },

    "f_fisher": {
        "id": "f_fisher",
        "name": "F de Fisher-Snedecor",
        "category": "Continua",
        "notation": r"X \sim F(d_1, d_2)",
        "params": [
            {
                "id": "df1",
                "label": "Grados de libertad numerador (d₁)",
                "latex": r"d_1 > 0",
                "min": 1,
                "max": 50,
                "default": 5,
                "step": 1,
                "type": "int",
                "help": "Grados de libertad de la varianza del numerador."
            },
            {
                "id": "df2",
                "label": "Grados de libertad denominador (d₂)",
                "latex": r"d_2 > 0",
                "min": 5,
                "max": 60,
                "default": 10,
                "step": 1,
                "type": "int",
                "help": "Grados de libertad de la varianza del denominador."
            }
        ],
        "support": r"R_X = (0, \infty)",
        "pmf_pdf_title": "Función de Densidad de Probabilidad (PDF)",
        "pmf_pdf_latex": r"f_X(x) = \frac{1}{\text{B}\left(\frac{d_1}{2}, \frac{d_2}{2}\right)} \left(\frac{d_1}{d_2}\right)^{d_1/2} x^{d_1/2 - 1} \left(1 + \frac{d_1}{d_2} x\right)^{-\frac{d_1 + d_2}{2}}",
        "cdf_latex": r"F_X(x) = I_{\frac{d_1 x}{d_1 x + d_2}}\left(\frac{d_1}{2}, \frac{d_2}{2}\right)",
        "mean_latex": r"E[X] = \frac{d_2}{d_2 - 2} \quad (\text{para } d_2 > 2)",
        "var_latex": r"\text{Var}(X) = \frac{2 d_2^2 (d_1 + d_2 - 2)}{d_1 (d_2 - 2)^2 (d_2 - 4)} \quad (\text{para } d_2 > 4)",
        "mgf_latex": r"M_X(t) \text{ no existe en forma cerrada para } t > 0 \quad (\text{cola pesada})",
        "actuarial_notes": r"""
**Relevancia en Actuaría y Modelos Estadísticos:**
- **Comparación de Volatilidad entre Carteras:** Permite contrastar si dos carteras de seguros presentan la misma variabilidad de siniestros mediante la razón de varianzas muestrales: $F = \frac{S_1^2 / \sigma_1^2}{S_2^2 / \sigma_2^2} \sim F(n_1-1, n_2-1)$.
- **Análisis de Varianza (ANOVA) y Modelos Lineales Generalizados (GLM):** Utilizada extensamente para probar si diferentes factores de riesgo (ej. edad del conductor, tipo de vehículo) tienen un impacto estadísticamente significativo en la prima de riesgo.
""",
        "scipy_dist": lambda params: stats.f(dfn=params["df1"], dfd=params["df2"]),
        "theoretical_moments": lambda params: (
            params["df2"] / (params["df2"] - 2.0) if params["df2"] > 2 else np.nan,
            (2.0 * (params["df2"] ** 2) * (params["df1"] + params["df2"] - 2.0)) /
            (params["df1"] * ((params["df2"] - 2.0) ** 2) * (params["df2"] - 4.0)) if params["df2"] > 4 else np.nan,
            ((2.0 * params["df1"] + params["df2"] - 2.0) * np.sqrt(8.0 * (params["df2"] - 4.0))) /
            ((params["df2"] - 6.0) * np.sqrt(params["df1"] * (params["df1"] + params["df2"] - 2.0))) if params["df2"] > 6 else np.nan,
            np.nan
        ),
        "get_x_range": lambda params: (
            np.linspace(0.001, stats.f.ppf(0.99, dfn=params["df1"], dfd=params["df2"]), 400),
            np.linspace(0.001, stats.f.ppf(0.99, dfn=params["df1"], dfd=params["df2"]), 400)
        )
    }
}
