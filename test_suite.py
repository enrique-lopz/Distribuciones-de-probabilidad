"""
Suite de pruebas automatizadas para verificar las 14 familias paramétricas y componentes de la app.
"""

import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np

from distributions_catalog import DISTRIBUTIONS
from plotting_utils import plot_pmf_pdf, plot_cdf, plot_monte_carlo, plot_comparison

def run_tests():
    print(f"Verificando {len(DISTRIBUTIONS)} familias paramétricas...")
    assert len(DISTRIBUTIONS) == 15, f"Se esperaban 15 distribuciones, encontradas {len(DISTRIBUTIONS)}"

    for dist_id, dist_info in DISTRIBUTIONS.items():
        print(f"Probando {dist_info['name']} ({dist_id})...")
        params = {p["id"]: p["default"] for p in dist_info["params"]}
        
        # Frozen scipy distribution
        frozen = dist_info["scipy_dist"](params)
        
        # X range
        x_vals, _ = dist_info["get_x_range"](params)
        
        # PMF/PDF
        if dist_info["category"] == "Discreta":
            pmf = frozen.pmf(x_vals)
            assert np.all(pmf >= 0)
        else:
            pdf = frozen.pdf(x_vals)
            assert np.all(pdf >= 0)
            
        # CDF
        cdf = frozen.cdf(x_vals)
        assert np.all(cdf >= -1e-6)
        
        # Theoretical moments
        m = dist_info["theoretical_moments"](params)
        assert len(m) == 4
        
        # Sampling
        sample = frozen.rvs(size=50)
        assert len(sample) == 50
        
        # Plotting
        fig_pmf = plot_pmf_pdf(dist_info, params)
        fig_cdf = plot_cdf(dist_info, params)
        fig_mc, _, _, _ = plot_monte_carlo(dist_info, params, n_samples=100)
        assert fig_pmf is not None and fig_cdf is not None and fig_mc is not None

    print("\n✅ ¡Todas las pruebas matemáticas y de gráficos pasaron exitosamente!")

if __name__ == "__main__":
    run_tests()

