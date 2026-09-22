"""
Utilidades de visualización interactiva con Plotly para Familias Paramétricas.
Diseñado para la cátedra de Probabilidad y Modelos Actuariales (FMAT - UADY).
"""

import numpy as np
import plotly.graph_objects as go
from scipy import stats


PRIMARY_COLOR = "#1565C0"      # Azul marino actuarial
ACCENT_COLOR = "#E65100"       # Ámbar / naranja de resalte
SECONDARY_COLOR = "#00897B"    # Verde esmeralda
SHADE_COLOR = "rgba(21, 101, 192, 0.28)"
SHADE_ACCENT = "rgba(230, 81, 0, 0.35)"


def plot_pmf_pdf(dist_info, params, prob_calc=None):
    """
    Genera el gráfico interactivo de la PMF (discreta) o PDF (continua),
    con resaltado del área o barras correspondientes al intervalo de probabilidad.
    
    prob_calc: dict opcional con:
        'type': 'leq' (X <= x), 'geq' (X >= x), 'gt' (X > x), 'between' (a <= X <= b), 'eq' (X == x)
        'a': float/int,
        'b': float/int
    """
    is_discrete = dist_info["category"] == "Discreta"
    frozen_dist = dist_info["scipy_dist"](params)
    x_vals, _ = dist_info["get_x_range"](params)
    
    fig = go.Figure()
    
    if is_discrete:
        # Calcular masa de probabilidad
        pmf_vals = frozen_dist.pmf(x_vals)
        
        # Determinar colores de barras según intervalo
        bar_colors = [PRIMARY_COLOR] * len(x_vals)
        if prob_calc is not None:
            calc_type = prob_calc.get("type")
            a = prob_calc.get("a", 0)
            b = prob_calc.get("b", 0)
            for i, x in enumerate(x_vals):
                selected = False
                if calc_type == "leq" and x <= a:
                    selected = True
                elif calc_type == "geq" and x >= a:
                    selected = True
                elif calc_type == "gt" and x > a:
                    selected = True
                elif calc_type == "between" and a <= x <= b:
                    selected = True
                elif calc_type == "eq" and x == a:
                    selected = True
                if selected:
                    bar_colors[i] = ACCENT_COLOR
                    
        fig.add_trace(go.Bar(
            x=x_vals,
            y=pmf_vals,
            marker=dict(
                color=bar_colors,
                line=dict(color="#0D47A1", width=1.5)
            ),
            width=0.45 if len(x_vals) < 30 else None,
            name="P(X = x)",
            hovertemplate="<b>x = %{x}</b><br>P(X = x) = %{y:.5f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"<b>Función de Masa de Probabilidad (PMF) — {dist_info['name']}</b>",
            xaxis_title="Valor de la Variable Aleatoria (x)",
            yaxis_title="Probabilidad P(X = x)",
            yaxis=dict(range=[0, max(pmf_vals) * 1.18 if len(pmf_vals) > 0 and max(pmf_vals) > 0 else 1]),
            template="plotly_white",
            hovermode="x",
            height=460,
            margin=dict(l=40, r=40, t=50, b=40)
        )
    else:
        # Distribución Continua: Densidad PDF
        pdf_vals = frozen_dist.pdf(x_vals)
        
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=pdf_vals,
            mode="lines",
            line=dict(color=PRIMARY_COLOR, width=3),
            name="f(x)",
            hovertemplate="<b>x = %{x:.3f}</b><br>f(x) = %{y:.5f}<extra></extra>"
        ))
        
        # Sombreado de intervalo si está activo
        if prob_calc is not None:
            calc_type = prob_calc.get("type")
            a = prob_calc.get("a", 0)
            b = prob_calc.get("b", 0)
            
            x_min, x_max = float(x_vals[0]), float(x_vals[-1])
            if calc_type == "leq":
                shade_x = np.linspace(x_min, min(a, x_max), 200)
            elif calc_type in ("geq", "gt"):
                shade_x = np.linspace(max(a, x_min), x_max, 200)
            elif calc_type == "between":
                shade_x = np.linspace(max(a, x_min), min(b, x_max), 200)
            else:
                shade_x = np.array([])
                
            if len(shade_x) > 1 and shade_x[-1] >= shade_x[0]:
                shade_y = frozen_dist.pdf(shade_x)
                fig.add_trace(go.Scatter(
                    x=np.concatenate([[shade_x[0]], shade_x, [shade_x[-1]]]),
                    y=np.concatenate([[0], shade_y, [0]]),
                    fill="toself",
                    fillcolor=SHADE_ACCENT,
                    line=dict(color="rgba(230,81,0,0)"),
                    hoverinfo="skip",
                    name="Área de Probabilidad"
                ))
                
        fig.update_layout(
            title=f"<b>Función de Densidad de Probabilidad (PDF) — {dist_info['name']}</b>",
            xaxis_title="x",
            yaxis_title="Densidad f(x)",
            yaxis=dict(range=[0, max(pdf_vals) * 1.18 if len(pdf_vals) > 0 and max(pdf_vals) > 0 else 1]),
            template="plotly_white",
            hovermode="x unified",
            height=460,
            margin=dict(l=40, r=40, t=50, b=40)
        )
        
    return fig


def plot_cdf(dist_info, params, quantile_val=None, prob_level=None):
    """
    Genera el gráfico interactivo de la Función de Distribución Acumulada F(x).
    Incluye líneas punteadas del cuantil / VaR si se proporciona.
    """
    is_discrete = dist_info["category"] == "Discreta"
    frozen_dist = dist_info["scipy_dist"](params)
    x_vals, _ = dist_info["get_x_range"](params)
    
    fig = go.Figure()
    
    if is_discrete:
        # Gráfico escalonado riguroso para discretas
        x_min = x_vals[0] - 1
        x_max = x_vals[-1] + 1
        dense_x = np.linspace(x_min, x_max, 600)
        cdf_vals = frozen_dist.cdf(dense_x)
        
        fig.add_trace(go.Scatter(
            x=dense_x,
            y=cdf_vals,
            mode="lines",
            line=dict(color=SECONDARY_COLOR, width=2.5, shape="hv"),
            name="F(x) = P(X ≤ x)",
            hovertemplate="<b>x = %{x:.2f}</b><br>F(x) = %{y:.5f}<extra></extra>"
        ))
        
        # Puntos de masa en los enteros
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=frozen_dist.cdf(x_vals),
            mode="markers",
            marker=dict(color=SECONDARY_COLOR, size=6),
            name="F(k)",
            hoverinfo="skip"
        ))
    else:
        # Distribución Continua
        dense_x = x_vals
        cdf_vals = frozen_dist.cdf(dense_x)
        
        fig.add_trace(go.Scatter(
            x=dense_x,
            y=cdf_vals,
            mode="lines",
            line=dict(color=SECONDARY_COLOR, width=3),
            name="F(x) = P(X ≤ x)",
            hovertemplate="<b>x = %{x:.3f}</b><br>F(x) = %{y:.5f}<extra></extra>"
        ))
        
    # Marcador de cuantil / VaR
    if quantile_val is not None and prob_level is not None:
        fig.add_trace(go.Scatter(
            x=[dense_x[0], quantile_val, quantile_val],
            y=[prob_level, prob_level, 0],
            mode="lines+markers",
            line=dict(color=ACCENT_COLOR, width=2, dash="dash"),
            marker=dict(size=[0, 8, 0], color=ACCENT_COLOR),
            name=f"Cuantil VaR({prob_level:.2f}) = {quantile_val:.3f}",
            hovertemplate=f"Nivel p = {prob_level:.4f}<br>Cuantil x = {quantile_val:.4f}<extra></extra>"
        ))
        
    fig.update_layout(
        title=f"<b>Función de Distribución Acumulada (CDF) — {dist_info['name']}</b>",
        xaxis_title="x",
        yaxis_title="F(x) = P(X ≤ x)",
        yaxis=dict(range=[-0.02, 1.05]),
        template="plotly_white",
        height=460,
        margin=dict(l=40, r=40, t=50, b=40)
    )
    return fig


def plot_monte_carlo(dist_info, params, n_samples=1000):
    """
    Genera un histograma empírico de simulación Monte Carlo
    comparado directamente con la curva o barras teóricas.
    """
    is_discrete = dist_info["category"] == "Discreta"
    frozen_dist = dist_info["scipy_dist"](params)
    
    # Generar muestra pseudoaleatoria con semilla reproducible pero dinámica
    sample = frozen_dist.rvs(size=n_samples, random_state=None)
    x_vals, _ = dist_info["get_x_range"](params)
    
    fig = go.Figure()
    
    if is_discrete:
        # Frecuencias relativas empíricas
        unique, counts = np.unique(sample, return_counts=True)
        emp_prob = counts / n_samples
        
        # Histograma / Barras empíricas
        fig.add_trace(go.Bar(
            x=unique,
            y=emp_prob,
            name=f"Muestra Empírica (n={n_samples})",
            opacity=0.6,
            marker_color="#90CAF9",
            marker_line=dict(color="#1565C0", width=1.5),
            hovertemplate="<b>x = %{x}</b><br>Frec. relativa = %{y:.5f}<extra></extra>"
        ))
        
        # Puntos teóricos
        pmf_teorica = frozen_dist.pmf(x_vals)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=pmf_teorica,
            mode="markers+lines",
            line=dict(color=ACCENT_COLOR, width=2),
            marker=dict(color=ACCENT_COLOR, size=8),
            name="Teórico P(X = x)",
            hovertemplate="<b>x = %{x}</b><br>P(X=x) teórico = %{y:.5f}<extra></extra>"
        ))
    else:
        # Histograma continuo normalizado a densidad
        fig.add_trace(go.Histogram(
            x=sample,
            histnorm="probability density",
            name=f"Muestra Empírica (n={n_samples})",
            opacity=0.55,
            marker_color="#80CBC4",
            marker_line=dict(color="#00695C", width=1),
            nbinsx=min(60, max(20, int(np.sqrt(n_samples)))),
            hovertemplate="Intervalo: %{x}<br>Densidad empírica: %{y:.5f}<extra></extra>"
        ))
        
        # Curva teórica de densidad
        pdf_teorica = frozen_dist.pdf(x_vals)
        fig.add_trace(go.Scatter(
            x=x_vals,
            y=pdf_teorica,
            mode="lines",
            line=dict(color=ACCENT_COLOR, width=3),
            name="Densidad Teórica f(x)",
            hovertemplate="<b>x = %{x:.3f}</b><br>f(x) teórico = %{y:.5f}<extra></extra>"
        ))
        
    fig.update_layout(
        title=f"<b>Simulación Monte Carlo vs Teoría — {dist_info['name']}</b>",
        xaxis_title="x",
        yaxis_title="Densidad / Probabilidad Relativa",
        template="plotly_white",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
    )
    
    # Calcular estadísticas empíricas vs teóricas
    emp_mean = float(np.mean(sample))
    emp_var = float(np.var(sample, ddof=1))
    
    return fig, sample, emp_mean, emp_var


def plot_comparison(dist1_info, params1, dist2_info, params2, label1=None, label2=None):
    """
    Superpone dos distribuciones para análisis comparativo directo en actuaría
    (e.g., Poisson vs Binomial Negativa para sobredispersión, o Normal vs t-Student).
    """
    froz1 = dist1_info["scipy_dist"](params1)
    froz2 = dist2_info["scipy_dist"](params2)
    
    name1 = label1 if label1 else dist1_info["name"]
    name2 = label2 if label2 else dist2_info["name"]
    
    is_disc1 = dist1_info["category"] == "Discreta"
    is_disc2 = dist2_info["category"] == "Discreta"
    
    fig = go.Figure()
    
    # Determinar rango común de x
    x1, _ = dist1_info["get_x_range"](params1)
    x2, _ = dist2_info["get_x_range"](params2)
    
    if is_disc1 and is_disc2:
        # Ambas discretas
        min_x = min(x1[0], x2[0])
        max_x = max(x1[-1], x2[-1])
        common_x = np.arange(min_x, max_x + 1)
        
        fig.add_trace(go.Bar(
            x=common_x - 0.15,
            y=froz1.pmf(common_x),
            name=name1,
            width=0.3,
            marker_color=PRIMARY_COLOR,
            hovertemplate=f"<b>{name1}</b><br>x = %{{x:.0f}}<br>P(X=x) = %{{y:.5f}}<extra></extra>"
        ))
        fig.add_trace(go.Bar(
            x=common_x + 0.15,
            y=froz2.pmf(common_x),
            name=name2,
            width=0.3,
            marker_color=ACCENT_COLOR,
            hovertemplate=f"<b>{name2}</b><br>x = %{{x:.0f}}<br>P(X=x) = %{{y:.5f}}<extra></extra>"
        ))
        fig.update_layout(
            barmode="group",
            xaxis_title="x (Conteo de eventos / siniestros)",
            yaxis_title="Probabilidad P(X = x)"
        )
    else:
        # Continuas (o una y una)
        min_x = min(x1[0], x2[0])
        max_x = max(x1[-1], x2[-1])
        common_x = np.linspace(min_x, max_x, 500)
        
        y1 = froz1.pmf(common_x) if is_disc1 else froz1.pdf(common_x)
        y2 = froz2.pmf(common_x) if is_disc2 else froz2.pdf(common_x)
        
        fig.add_trace(go.Scatter(
            x=common_x,
            y=y1,
            mode="lines",
            line=dict(color=PRIMARY_COLOR, width=3),
            name=name1,
            hovertemplate=f"<b>{name1}</b><br>x = %{{x:.3f}}<br>f(x) = %{{y:.5f}}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=common_x,
            y=y2,
            mode="lines",
            line=dict(color=ACCENT_COLOR, width=3, dash="dash"),
            name=name2,
            hovertemplate=f"<b>{name2}</b><br>x = %{{x:.3f}}<br>f(x) = %{{y:.5f}}<extra></extra>"
        ))
        fig.update_layout(
            xaxis_title="x",
            yaxis_title="Densidad / Probabilidad"
        )
        
    fig.update_layout(
        title=f"<b>Comparación Directa: {name1} vs {name2}</b>",
        template="plotly_white",
        height=480,
        margin=dict(l=40, r=40, t=50, b=40),
        legend=dict(yanchor="top", y=0.98, xanchor="right", x=0.98)
    )
    
    return fig

