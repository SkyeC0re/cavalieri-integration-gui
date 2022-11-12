import streamlit as st

st.write("# Cavalieri Integration")

HOME_TEXT = r"""
Cavalieri Integration in $\mathbb{R}^n$  presents a novel visualization mechanism for weighted integration and
challenges the notion of strictly rectangular integration strips. It does so by concealing the in-
tegrator inside the boundary curves of the integral. A superset of Riemann-integration in $\mathbb{R}^{n−1}$,
the Cavalieri integral is defined by a translational region $\mathcal{C} = {f, c, R, S}$, which uniquely defines
the integrand, integrator and integration region. In $\mathbb{R}^2$ it allows for the visualization of Riemann-
Stieltjes integrals along with other forms of weighted integration such as the Riemann–Liouville
fractional integral and convolution operator. Programmatic implementation of such visualizations
and computation of integral values relies on knowledge relating to numeric integration, algorithmic
differentiation and numeric root finding. For the $\mathbb{R}^3$ case, such visualizations over polygonal regions
requires mechanism for triangulating a set of nested polygons and transformations which allow for
the use of repeated integration to solve the integration value over a triangular region using standard
1-dimensional integration routines.
"""

st.write(HOME_TEXT)