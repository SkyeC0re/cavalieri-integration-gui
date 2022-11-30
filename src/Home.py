import streamlit as st

st.write("# Cavalieri Integration")

HOME_TEXT = r"""
Cavalieri Integration in $\mathbb{R}^n$ presents a novel visualization mechanism for weighted integration 
and challenges the notion of strictly rectangular integration strips. It does so by concealing the integrator
inside the boundary curves of the integral. This sandbox allows the user to create visualizations of this
integral in $\mathbb{R}^2$ and $\mathbb{R}^3$ as well as creating Cavalieri integral representations for
the Riemann-Stieltjes integral.
"""

st.write(HOME_TEXT)
