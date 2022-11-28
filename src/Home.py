import streamlit as st

st.write("# Cavalieri Integration")

HOME_TEXT = r"""
Cavalieri Integration in $\mathbb{R}^n$ presents a novel visualization mechanism for weighted integration 
and challenges the notion of strictly rectangular integration strips. It does so by concealing the integrator
inside the boundary curves of the integral. This paper investigates the Cavalieri integral as a superset of
Riemann-integration in $\mathbb{R}^{n-1}$, whereby the integral is defined by a translational region in 
$\mathbb{R}^{n-1}$, which uniquely defines the integrand, integrator and integration region. In $\mathbb{R}^2$,
this refined translational region definition allows for the visualization of Riemann-Stieltjes integrals
along with other forms of weighted integration such as the Riemann–Liouville fractional integral and
convolution operator. Programmatic implementation of such visualizations and computation of integral values are
also investigated and relies on knowledge relating to numeric integration, algorithmic differentiation and 
numeric root finding. For the $\mathbb{R}^3$ case, such visualizations over polygonal regions requires a 
mechanism for the triangulation of a set of nested polygons and transformations which allow for the use of 
repeated integration to solve the integration value over the produced triangular regions using standard 
1-dimensional integration routines.
"""

st.write(HOME_TEXT)
