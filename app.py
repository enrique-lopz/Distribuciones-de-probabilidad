"""
Aplicación Interactiva: Familias Paramétricas de Probabilidad en la Ciencia Actuarial
Diseñada para estudiantes de Licenciatura en Actuaría (3er Semestre) - FMAT, UADY.
"""

import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats

from distributions_catalog import DISTRIBUTIONS
from plotting_utils import (
    plot_pmf_pdf,
    plot_cdf,
    plot_comparison
)

# Configuración inicial de la página
#st.set_page_state = "expanded"
st.set_page_config(
    page_title="Familias Paramétricas | Actuaría FMAT",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inyección de estilos CSS personalizados para una apariencia académica y profesional
st.markdown("""
<style>
    /* Estilos generales */
    .main {
        background-color: #FAFAFB;
    }
    .main-title {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        color: #0D47A1;
        font-weight: 700;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    .sub-title {
        color: #455A64;
        font-size: 1.1rem;
        margin-top: 4px;
        margin-bottom: 20px;
    }
    .badge-fmat {
        background: linear-gradient(135deg, #1565C0, #0D47A1);
        color: white;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 12px;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px 6px 0px 0px;
        font-weight: 500;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
        font-weight: 700;
        color: #0D47A1;
    }
    /* Estilos para tablas Markdown con fórmulas matemáticas (Fondo oscuro uniforme) */
    div[data-testid="stMarkdownContainer"] table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 10px;
        margin-bottom: 20px;
        font-size: 0.95rem;
        background-color: #1E222B;
        border-radius: 6px;
    }
    div[data-testid="stMarkdownContainer"] th {
        background-color: #0D1B2A;
        color: #90CAF9;
        font-weight: 700;
        padding: 10px 14px;
        border: 1px solid #2E384D;
        text-align: left;
    }
    div[data-testid="stMarkdownContainer"] td {
        background-color: #1E222B;
        color: #E2E8F0;
        padding: 10px 14px;
        border: 1px solid #2E384D;
        vertical-align: middle;
    }
    div[data-testid="stMarkdownContainer"] tr {
        background-color: #1E222B;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# ENCABEZADO PRINCIPAL
# =============================================================================
st.markdown('<div class="badge-fmat">🎓 Facultad de Matemáticas — UADY | Licenciatura en Actuaría</div>', unsafe_allow_html=True)
st.markdown('<h1 class="main-title"> 🏛️ Probabilidad I</h1>', unsafe_allow_html=True)

# Navegación Superior
nav_tabs = st.tabs([
    "⚡ Módulo 1",
    "🔬 Módulo 2",
    "ℹ️ Módulo 3"
])


# =============================================================================
# MÓDULO 1: EXPLORADOR INDIVIDUAL DE FAMILIAS
# =============================================================================


with nav_tabs[0]:
    st.markdown("##⚡Explorador Individual de Familias Paramétricas")
    col_sel1, col_sel2 = st.columns([1, 2])
    
    num_disc = sum(1 for v in DISTRIBUTIONS.values() if v["category"] == "Discreta")
    num_cont = sum(1 for v in DISTRIBUTIONS.values() if v["category"] == "Continua")

    
    
    avail_dists = DISTRIBUTIONS
    dist_options = {f"{v['name']} ({v['category']})": k for k, v in avail_dists.items()} 
    
    
    selected_label = st.selectbox(
        "Selecciona una Familia Paramétrica:",
        list(dist_options.keys()),
        index=1 if "Binomial (Discreta)" in dist_options else 0
    )
    
    dist_key = dist_options[selected_label]
    dist_info = DISTRIBUTIONS[dist_key]
    
    st.markdown("---")
    
    # Barra lateral o columnas para parámetros
    col_params, col_main = st.columns([1, 2.5])
    
    with col_params:
        st.markdown(f"### ⚙️ Parámetros: {dist_info['name']}")
        st.markdown(f"**Notación:** ${dist_info['notation']}$")
        st.markdown(f"**Soporte:** ${dist_info['support']}$")
        
        # Recolector de parámetros dinámicos
        param_values = {}
        for p in dist_info["params"]:
            p_id = p["id"]
            label = f"{p['label']}"
            
            # Restricciones condicionales especiales
            min_val = p["min"]
            max_val = p["max"]
            default_val = p["default"]
            
            # Manejo de restricciones cruzadas (ej. b >= a en uniforme, K <= N, n <= N en hipergeométrica)
            if dist_key in ("discrete_uniform", "continuous_uniform") and p_id == "b":
                min_val = max(min_val, param_values["a"] + (1 if dist_key == "discrete_uniform" else 0.5))
                default_val = max(default_val, min_val)
            elif dist_key == "hypergeom" and p_id == "K":
                max_val = min(max_val, param_values["N"])
                default_val = min(default_val, max_val)
            elif dist_key == "hypergeom" and p_id == "n":
                max_val = min(max_val, param_values["N"])
                default_val = min(default_val, max_val)
                
            if p["type"] == "int":
                param_values[p_id] = st.slider(
                    label=label,
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default_val),
                    step=int(p["step"]),
                    help=p["help"],
                    key=f"{dist_key}_{p_id}"
                )
            else:
                param_values[p_id] = st.slider(
                    label=label,
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default_val),
                    step=float(p["step"]),
                    help=p["help"],
                    key=f"{dist_key}_{p_id}"
                )
            st.latex(p["latex"])
            
        st.markdown("---")
        st.markdown("#### 💡 Consejos de Uso")
        st.caption("Modifica los deslizadores para observar en tiempo real cómo cambia la forma de la distribución, su asimetría y el peso de sus colas.")

    with col_main:
        # Cálculo de momentos teóricos
        try:
            mean_val, var_val, skew_val, kurt_val = dist_info["theoretical_moments"](param_values)
            sd_val = np.sqrt(var_val) if not np.isnan(var_val) and not np.isinf(var_val) and var_val >= 0 else np.nan
        except Exception:
            mean_val, var_val, sd_val, skew_val, kurt_val = np.nan, np.nan, np.nan, np.nan, np.nan

        # Fila de métricas
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1:
            st.metric("Media E[X]", f"{mean_val:.3f}" if not np.isnan(mean_val) else "Indefinida")
        with m2:
            st.metric("Varianza Var(X)", f"{var_val:.3f}" if not np.isnan(var_val) and not np.isinf(var_val) else ("∞" if np.isinf(var_val) else "Indefinida"))
        with m3:
            st.metric("Desv. Estándar σ", f"{sd_val:.3f}" if not np.isnan(sd_val) else "Indefinida")
        with m4:
            st.metric("Asimetría γ₁", f"{skew_val:.3f}" if not np.isnan(skew_val) else "—")
        with m5:
            st.metric("Curtosis γ₂", f"{kurt_val:.3f}" if not np.isnan(kurt_val) and not np.isinf(kurt_val) else "—")

        # Subpestañas de contenido para la distribución
        sub_tab1, sub_tab2, sub_tab3, sub_tab4 = st.tabs([
            "📈 Gráficos Interactivos",
            "📐 Teoría y Fórmulas",
            "🧮 Calculadora & VaR Actuarial",
            "🏛️ Aplicación en Actuaría"
        ])

        # SUBPESTAÑA 1: GRÁFICOS INTERACTIVOS
        with sub_tab1:
            plot_choice = st.radio(
                "Seleccionar vista gráfica:",
                ["Función de Masa / Densidad (PMF/PDF)", "Función de Distribución Acumulada (CDF)", "Ambas lado a lado"],
                horizontal=True
            )
            
            if plot_choice == "Función de Masa / Densidad (PMF/PDF)":
                fig_pdf = plot_pmf_pdf(dist_info, param_values)
                st.plotly_chart(fig_pdf, use_container_width=True)
            elif plot_choice == "Función de Distribución Acumulada (CDF)":
                fig_cdf = plot_cdf(dist_info, param_values)
                st.plotly_chart(fig_cdf, use_container_width=True)
            else:
                c_p1, c_p2 = st.columns(2)
                with c_p1:
                    st.plotly_chart(plot_pmf_pdf(dist_info, param_values), use_container_width=True)
                with c_p2:
                    st.plotly_chart(plot_cdf(dist_info, param_values), use_container_width=True)

        # SUBPESTAÑA 2: TEORÍA Y FÓRMULAS
        with sub_tab2:
            st.markdown(f"### Fórmulas Principales — {dist_info['name']}")
            
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                st.markdown(f"#### {dist_info['pmf_pdf_title']}")
                st.latex(dist_info["pmf_pdf_latex"])
                
                st.markdown("#### Función de Distribución Acumulada $F_X(x)$")
                st.latex(dist_info["cdf_latex"])
                
            with c_f2:
                st.markdown("#### Valor Esperado y Varianza")
                st.latex(dist_info["mean_latex"])
                st.latex(dist_info["var_latex"])
                
                st.markdown("#### Función Generadora de Momentos (FGM) $M_X(t)$")
                st.latex(dist_info["mgf_latex"])
                
            st.info(r"""
            **Nota de Cálculo de Momentos:**
            Recordar que la varianza se recupera mediante $\text{Var}(X) = E[X^2] - (E[X])^2$.
            """)

        # SUBPESTAÑA 3: CALCULADORA Y VAR ACTUARIAL
        with sub_tab3:
            st.markdown("### 🧮 Calculadora de Probabilidades y Métricas de Riesgo")
            st.markdown("Utilizada en actuaría para tarificación de deducibles, probabilidades de ruina y capital de solvencia.")
            
            frozen = dist_info["scipy_dist"](param_values)
            is_disc = dist_info["category"] == "Discreta"
            
            c_calc1, c_calc2 = st.columns([1.2, 1.8])
            
            with c_calc1:
                calc_mode = st.selectbox(
                    "Tipo de cálculo actuarial:",
                    [
                        "Función de Distribución Acumulada: P(X ≤ x)",
                        "Función de Supervivencia / Deducible: P(X > d)",
                        "Probabilidad en Intervalo: P(a ≤ X ≤ b)",
                        "Cuantil / Value at Risk: VaR_α(X)"
                    ] + (["Probabilidad puntual: P(X = k)"] if is_disc else [])
                )
                
                prob_calc_dict = None
                
                if calc_mode == "Función de Distribución Acumulada: P(X ≤ x)":
                    default_x = float(mean_val) if not np.isnan(mean_val) else 0.0
                    val_x = st.number_input("Valor de corte (x):", value=default_x, step=0.5 if not is_disc else 1.0)
                    prob_res = frozen.cdf(val_x)
                    st.success(f"**P(X ≤ {val_x}) = {prob_res:.5f}** ({prob_res*100:.3f}%)")
                    prob_calc_dict = {"type": "leq", "a": val_x}
                    
                elif calc_mode == "Función de Supervivencia / Deducible: P(X > d)":
                    st.caption("En reaseguro y seguros de daños, representa la probabilidad de que un siniestro supere el deducible o límite de retención $d$.")
                    default_d = float(mean_val) if not np.isnan(mean_val) else 0.0
                    val_d = st.number_input("Deducible / Límite (d):", value=default_d, step=0.5 if not is_disc else 1.0)
                    if is_disc:
                        prob_res = 1.0 - frozen.cdf(val_d)
                    else:
                        prob_res = frozen.sf(val_d)
                    st.success(f"**S({val_d}) = P(X > {val_d}) = {prob_res:.5f}** ({prob_res*100:.3f}%)")
                    prob_calc_dict = {"type": "gt", "a": val_d}
                    
                elif calc_mode == "Probabilidad en Intervalo: P(a ≤ X ≤ b)":
                    def_a = float(mean_val - sd_val) if not np.isnan(mean_val) and not np.isnan(sd_val) else 0.0
                    def_b = float(mean_val + sd_val) if not np.isnan(mean_val) and not np.isnan(sd_val) else 2.0
                    c_a, c_b = st.columns(2)
                    with c_a:
                        val_a = st.number_input("Límite inferior (a):", value=def_a, step=0.5 if not is_disc else 1.0)
                    with c_b:
                        val_b = st.number_input("Límite superior (b):", value=def_b, step=0.5 if not is_disc else 1.0)
                    
                    if val_b < val_a:
                        st.error("El límite superior (b) debe ser mayor o igual al inferior (a).")
                    else:
                        if is_disc:
                            # P(a <= X <= b) = F(b) - F(a - 1)
                            prob_res = frozen.cdf(val_b) - frozen.cdf(val_a - 1)
                        else:
                            prob_res = frozen.cdf(val_b) - frozen.cdf(val_a)
                        st.success(f"**P({val_a} ≤ X ≤ {val_b}) = {prob_res:.5f}** ({prob_res*100:.3f}%)")
                        prob_calc_dict = {"type": "between", "a": val_a, "b": val_b}
                        
                elif calc_mode == "Cuantil / Value at Risk: VaR_α(X)":
                    st.caption("El cuantil $F^{-1}(\\alpha)$ representa el nivel de pérdida que no será superado con una probabilidad de confianza $\\alpha$ (métrica Solvencia II).")
                    alpha_level = st.slider("Nivel de confianza (α):", min_value=0.01, max_value=0.999, value=0.95, step=0.01)
                    var_quantile = frozen.ppf(alpha_level)
                    st.success(f"**VaR_{{{alpha_level:.3f}}}(X) = {var_quantile:.4f}**")
                    st.info(f"Interpretación actuarial: El {alpha_level*100:.1f}% de las veces, la variable no superará el valor de {var_quantile:.4f}.")
                    
                elif calc_mode == "Probabilidad puntual: P(X = k)":
                    val_k = st.number_input("Punto entero (k):", value=int(mean_val) if not np.isnan(mean_val) else 0, step=1)
                    prob_res = frozen.pmf(val_k)
                    st.success(f"**P(X = {val_k}) = {prob_res:.5f}** ({prob_res*100:.3f}%)")
                    prob_calc_dict = {"type": "eq", "a": val_k}
                    
            with c_calc2:
                if calc_mode == "Cuantil / Value at Risk: VaR_α(X)":
                    st.plotly_chart(plot_cdf(dist_info, param_values, quantile_val=var_quantile, prob_level=alpha_level), use_container_width=True)
                else:
                    st.plotly_chart(plot_pmf_pdf(dist_info, param_values, prob_calc=prob_calc_dict), use_container_width=True)

        # SUBPESTAÑA 5: APLICACIÓN ACTUARIAL
        with sub_tab4:
            st.markdown(f"### 🏛️ Aplicación en Actuaría — {dist_info['name']}")
            st.markdown(dist_info["actuarial_notes"])


# =============================================================================
# MÓDULO 2: LABORATORIO DE COMPARACIÓN & CONVERGENCIAS ACTUARIALES
# =============================================================================

with nav_tabs[1]:
    st.markdown("## 🔬 Laboratorio de Comparación y Convergencias")
    st.markdown("En la práctica actuarial, la selección de la distribución adecuada define la solvencia de una compañía de seguros o fondo de pensiones.")
    
    lab_mode = st.radio(
        "Selecciona el experimento actuarial:",
        [
            "1. Detección de Sobredispersión: Poisson vs Binomial Negativa",
            "2. Colas Pesadas y Riesgo Catastrófico: Normal vs t de Student",
            "3. Ley de Sucesos Raros: Binomial → Poisson",
            "4. Comparador Libre entre dos distribuciones"
        ],
        horizontal=True
    )
    
    st.markdown("---")
    
    # CASO 1: SOBREDISPERSIÓN POISSON VS BINOMIAL NEGATIVA
    if "1. Detección de Sobredispersión" in lab_mode:
        st.markdown("### 🚗 Caso 1: Frecuencia de Siniestros y Sobredispersión")
        st.markdown("""
        **Contexto Actuarial:** Supón una cartera de pólizas de automóviles. En un modelo homogéneo se asume $N \\sim \\text{Poisson}(\\mu)$.
        Sin embargo, los asegurados conducen diferente (jóvenes vs adultos, ciudad vs carretera). Esta heterogeneidad genera que la varianza
        real sea mayor que la media (sobredispersión actuarial). La **Binomial Negativa** calibra con la misma media $\\mu$, pero mayor varianza.
        """)
        
        c_l1, c_l2 = st.columns([1, 2])
        with c_l1:
            mu_target = st.slider("Media común deseada de siniestros (μ):", min_value=0.5, max_value=10.0, value=3.0, step=0.5)
            r_param = st.slider("Parámetro de forma de la heterogeneidad (r):", min_value=0.5, max_value=15.0, value=2.0, step=0.5,
                                help="Valores pequeños de r implican mayor varianza y heterogeneidad extrema.")
            
            # Para que la Binomial Negativa tenga E[Y] = r(1-p)/p = mu:
            # mu * p = r - r*p => p*(mu + r) = r => p = r / (mu + r)
            p_nb = r_param / (mu_target + r_param)
            var_nb = r_param * (1 - p_nb) / (p_nb ** 2)
            
            st.metric("Varianza Poisson", f"{mu_target:.2f}")
            st.metric("Varianza Binomial Negativa", f"{var_nb:.2f}", delta=f"+{((var_nb/mu_target)-1)*100:.1f}% sobredispersión")
            st.caption(f"Parámetro p calibrado para NB: {p_nb:.3f}")
            
        with c_l2:
            fig_comp = plot_comparison(
                DISTRIBUTIONS["poisson"], {"lambda": mu_target},
                DISTRIBUTIONS["nbinom"], {"r": r_param, "p": p_nb},
                label1=f"Poisson (μ={mu_target:.1f}, Var={mu_target:.1f})",
                label2=f"Binomial Negativa (μ={mu_target:.1f}, Var={var_nb:.1f})"
            )
            st.plotly_chart(fig_comp, use_container_width=True)
            
        st.warning("""
        **Conclusión para el Actuario:** Observa cómo la Binomial Negativa tiene mayor probabilidad en $x=0$ (conductores sin siniestros) 
        y simultáneamente una **cola derecha más gruesa** (mayor probabilidad de conductores con 6 o más accidentes). ¡Asumir erróneamente Poisson 
        subestima la probabilidad de siniestros múltiples!
        """)

    # CASO 2: NORMAL VS T-STUDENT (COLAS PESADAS)
    elif "2. Colas Pesadas" in lab_mode:
        st.markdown("### 📉 Caso 2: Rendimientos Financieros y Colas Pesadas (*Fat Tails*)")
        st.markdown("""
        **Contexto Actuarial:** Los modelos clásicos (como Black-Scholes o VaR regulatorio básico) asumen normalidad.
        No obstante, en crisis financieras las pérdidas extremas ocurren con frecuencia mucho mayor a la predicha por la curva normal.
        La distribución $t$ de Student modela este fenómeno mediante grados de libertad $\\nu$.
        """)
        
        c_t1, c_t2 = st.columns([1, 2])
        with c_t1:
            nu_df = st.slider("Grados de libertad de la t-Student (ν):", min_value=2, max_value=30, value=3, step=1)
            sigma_norm = np.sqrt(nu_df / (nu_df - 2)) if nu_df > 2 else 1.5
            
            st.markdown(f"**Comparación con desv. estándar idéntica:** $\\sigma = {sigma_norm:.3f}$")
            
            # Cálculo de probabilidad de evento extremo (pérdida > 3 desviaciones estándar)
            prob_norm_3sd = 1.0 - stats.norm.cdf(3 * sigma_norm, scale=sigma_norm)
            prob_t_3sd = 1.0 - stats.t.cdf(3 * sigma_norm, df=nu_df)
            ratio_risk = prob_t_3sd / prob_norm_3sd if prob_norm_3sd > 0 else np.nan
            
            st.metric("P(Pérdida > 3σ) Normal", f"{prob_norm_3sd:.5f}")
            st.metric("P(Pérdida > 3σ) t-Student", f"{prob_t_3sd:.5f}", delta=f"{ratio_risk:.1f}x más probable!")
            
        with c_t2:
            fig_t = plot_comparison(
                DISTRIBUTIONS["normal"], {"mu": 0.0, "sigma": sigma_norm},
                DISTRIBUTIONS["t_student"], {"df": nu_df},
                label1=f"Normal (σ={sigma_norm:.2f})",
                label2=f"t de Student (ν={nu_df})"
            )
            st.plotly_chart(fig_t, use_container_width=True)
            
        st.error("""
        **Impacto Actuarial en Solvencia II:** 
        Un evento de 3 o 4 desviaciones estándar ("imposible" bajo la normal) ocurre decenas o cientos de veces más a menudo bajo una distribución de colas pesadas.
        No contemplar colas pesadas conduce a la subestimación del capital de solvencia requerido y al riesgo de ruina.
        """)

    # CASO 3: LEY DE SUCESOS RAROS BINOMIAL A POISSON
    elif "3. Ley de Sucesos Raros" in lab_mode:
        st.markdown("### 🎲 Caso 3: Teorema del Límite de Poisson (Ley de Sucesos Raros)")
        st.markdown("""
        Si $n \\to \\infty$ y $p \\to 0$ tal que $\\lambda = np$ permanece constante:
        $$\\lim_{n \\to \\infty} \\binom{n}{x} p^x (1 - p)^{n - x} = \\frac{\\lambda^x e^{-\\lambda}}{x!}$$
        """)
        
        c_b1, c_b2 = st.columns([1, 2])
        with c_b1:
            lambda_val = st.slider("Tasa constante deseada (λ = np):", min_value=1.0, max_value=10.0, value=3.0, step=0.5)
            n_val = st.slider("Número de pólizas en la cartera (n):", min_value=10, max_value=250, value=20, step=5)
            p_val = lambda_val / n_val
            
            st.metric("Probabilidad individual p = λ / n", f"{p_val:.4f}")
            st.caption("A medida que aumentas n, la probabilidad individual p se vuelve muy pequeña y la aproximación se vuelve casi indistinguible.")
            
        with c_b2:
            fig_pois = plot_comparison(
                DISTRIBUTIONS["binomial"], {"n": n_val, "p": p_val},
                DISTRIBUTIONS["poisson"], {"lambda": lambda_val},
                label1=f"Binomial (n={n_val}, p={p_val:.3f})",
                label2=f"Poisson (λ={lambda_val:.1f})"
            )
            st.plotly_chart(fig_pois, use_container_width=True)

    # CASO 4: COMPARADOR LIBRE
    else:
        st.markdown("### ⚖️ Comparador Libre de Familias")
        st.markdown("Superpón cualquier par de distribuciones para comparar formas, dispersión y colas.")
        
        c_sel_a, c_sel_b = st.columns(2)
        
        with c_sel_a:
            st.markdown("#### Distribución 1")
            d1_key = st.selectbox("Selecciona Distribución 1:", list(DISTRIBUTIONS.keys()), index=8, key="free_d1")
            d1_info = DISTRIBUTIONS[d1_key]
            d1_params = {}
            for p in d1_info["params"]:
                d1_params[p["id"]] = st.slider(
                    f"{p['label']} (D1):",
                    min_value=float(p["min"]),
                    max_value=float(p["max"]),
                    value=float(p["default"]),
                    step=float(p["step"]),
                    key=f"free_d1_{p['id']}"
                )
                
        with c_sel_b:
            st.markdown("#### Distribución 2")
            d2_key = st.selectbox("Selecciona Distribución 2:", list(DISTRIBUTIONS.keys()), index=9, key="free_d2")
            d2_info = DISTRIBUTIONS[d2_key]
            d2_params = {}
            for p in d2_info["params"]:
                d2_params[p["id"]] = st.slider(
                    f"{p['label']} (D2):",
                    min_value=float(p["min"]),
                    max_value=float(p["max"]),
                    value=float(p["default"]),
                    step=float(p["step"]),
                    key=f"free_d2_{p['id']}"
                )
                
        fig_free = plot_comparison(d1_info, d1_params, d2_info, d2_params)
        st.plotly_chart(fig_free, use_container_width=True)


# =============================================================================
# MÓDULO 3: TABLA MAESTRA & GLOSARIO ACTUARIAL
# =============================================================================

with nav_tabs[2]:
    st.markdown(f"## 📊 Formulario de las {len(DISTRIBUTIONS)} Familias Paramétricas")
    #st.markdown("Guía condensada de referencia rápida con formulación matemática rigurosa para el estudiante de actuaría.")
    
    # Filtro opcional por categoría para mayor comodidad visual
    n_disc_master = sum(1 for v in DISTRIBUTIONS.values() if v["category"] == "Discreta")
    n_cont_master = sum(1 for v in DISTRIBUTIONS.values() if v["category"] == "Continua")
    
    table_filter = st.radio(
        "Filtrar familias en la tabla maestra:",
        [f"Todas ({len(DISTRIBUTIONS)})", f"Discretas ({n_disc_master})", f"Continuas ({n_cont_master})"],
        horizontal=True,
        key="master_cat_filter"
    )
    
    if table_filter.startswith("Discretas"):
        selected_dists = {k: v for k, v in DISTRIBUTIONS.items() if v["category"] == "Discreta"}
    elif table_filter.startswith("Continuas"):
        selected_dists = {k: v for k, v in DISTRIBUTIONS.items() if v["category"] == "Continua"}
    else:
        selected_dists = DISTRIBUTIONS

    # Construcción de la tabla en Markdown para renderizado matemático completo con KaTeX
    table_md = """| Distribución | Tipo | Notación | Soporte | Media $E[X]$ | Varianza $\\text{Var}(X)$ | FGM $M_X(t)$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for k, v in selected_dists.items():
        table_md += f"| **{v['name']}** | {v['category']} | ${v['notation']}$ | ${v['support']}$ | ${v['mean_latex']}$ | ${v['var_latex']}$ | ${v['mgf_latex']}$ |\n"

    st.markdown(table_md)

    with st.expander("📥 Exportar o descargar datos en CSV (sin formato LaTeX)"):
        master_records = []
        for k, v in selected_dists.items():
            master_records.append({
                "Distribución": v["name"],
                "Tipo": v["category"],
                "Notación": v["notation"],
                "Soporte": v["support"],
                "Media E[X]": v["mean_latex"],
                "Varianza Var(X)": v["var_latex"],
                "FGM M_X(t)": v["mgf_latex"]
            })
        df_export = pd.DataFrame(master_records)
        csv_data = df_export.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📄 Descargar esta vista en CSV",
            data=csv_data,
            file_name="tabla_maestra_familias_probabilidad.csv",
            mime="text/csv"
        )
    
    st.markdown("---")
    st.markdown("## 📖 Glosario de Conceptos Actuariales")
    
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        st.markdown(r"""
        ### 1. Función de Supervivencia $S(x)$
        $$S(x) = P(X > x) = 1 - F_X(x)$$
        - En **seguros de vida**: Probabilidad de que un individuo de edad 0 sobreviva hasta la edad $x$.
        - En **seguros de daños y reaseguro**: Probabilidad de que el monto de una pérdida supere el deducible o límite de retención $d$.
        
        ### 2. Deducibles Ordinarios y Monto Pagado
        Si $X$ es el monto de siniestro y $d$ es el deducible:
        $$Y = (X - d)_+ = \max(0, X - d)$$
        El costo medio esperado para la aseguradora se calcula integrando la función de supervivencia:
        $$E[Y] = \int_d^\infty S(x) \, dx$$
        
        ### 3. Sobredispersión (*Overdispersion*)
        Ocurre cuando la varianza de los datos es estrictamente mayor que su media:
        $$\text{Var}(X) > E[X]$$
        Es la razón principal por la que la distribución **Binomial Negativa** reemplaza a la **Poisson** en la tarificación de seguros generales de autos y gastos médicos.
        """)
        
    with col_g2:
        st.markdown(r"""
        ### 4. Distribuciones de Colas Pesadas (*Fat/Heavy Tails*)
        Una distribución es de cola pesada si su función de supervivencia decae más lentamente que una exponencial, es decir:
        $$\lim_{x \to \infty} e^{tx} P(X > x) = \infty, \quad \forall t > 0$$
        Esto implica que la **FGM $M_X(t)$ no existe para ningún $t > 0$**. Ejemplos: $t$ de Student, Pareto, Lognormal. En actuaría, estas familias modelan riesgos catastróficos y siniestros de alta severidad.
        
        ### 5. Value at Risk (VaR)
        Dado un nivel de confianza $\alpha \in (0, 1)$, el $\text{VaR}_\alpha(X)$ es el cuantil correspondiente:
        $$\text{VaR}_\alpha(X) = F_X^{-1}(\alpha) = \inf \{ x \in \mathbb{R} : F_X(x) \ge \alpha \}$$
        Bajo los marcos regulatorios de **Solvencia II** (Europa) y la **CNSF** (México), el requerimiento de capital de solvencia anual de una aseguradora se calibra al cuantil $\alpha = 99.5\%$.
        
        ### 6. Función Generadora de Momentos y Convolución
        Para la suma de pérdidas independientes $S_n = X_1 + \dots + X_n$:
        $$M_{S_n}(t) = \prod_{i=1}^n M_{X_i}(t)$$
        Si los riesgos son i.i.d., $M_{S_n}(t) = [M_X(t)]^n$. Esta propiedad algebraica permite encontrar la distribución exacta del monto agregado de reclamaciones en carteras aseguradas.
        """)

# Pie de página
st.markdown("---")
st.caption("🏛️ Universidad Autónoma de Yucatán | Facultad de Matemáticas (FMAT) — Licenciatura en Actuaría | Diseñado como recurso docente interactivo de Probabilidad y Modelado Estocástico.")
