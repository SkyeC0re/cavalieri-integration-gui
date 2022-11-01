import streamlit as st


def st_or_default(key: str, default):
    v = st.session_state.get(key)
    if v is None:
        return default
    return v


st.markdown("## Config")

tab1, tab2, tab3 = st.tabs(
    ("Integration and Rootfinding", "Cavalieri Display 2D", "Cavalieri Display 3D"))

with tab1:

    rf_iters = st.number_input("Maximum Root Finding Iterations",
                               help="Sets the maximum root finding iterations before failure.", min_value=10, max_value=1000, value=st_or_default("rf_iters", 100))
    if rf_iters:
        st.session_state["rf_iters"] = rf_iters

    integ_iters = st.number_input("Maximum Integration Iterations",
                                  help="Sets the maximum adaptive Gauss-Kronrod iterations before failure.", min_value=10, max_value=1000, value=st_or_default("integ_iters", 100))
    if integ_iters:
        st.session_state["integ_iters"] = integ_iters
    else:
        st.markdown("# WHYYY")

    tol = st.number_input("Integration Tolerance", help="Sets the absolute integration tolerance.",
                          min_value=1e-11, max_value=1.0, value=st_or_default("tol", 1e-9))
    if tol:
        st.session_state["tol"] = tol

with tab2:
    x_res2d = st.number_input("X Resolution", help="Sets the x resolution for each produced interval.",
                              min_value=50, max_value=150, value=st_or_default("x_res2d", 100))
    if x_res2d:
        st.session_state["x_res2d"] = x_res2d

    y_res2d = st.number_input("Y Resolution", help="Sets the y resolution for each produced interval.",
                              min_value=50, max_value=150, value=st_or_default("y_res2d", 100))
    if y_res2d:
        st.session_state["y_res2d"] = y_res2d

with tab3:
    radial_res3d = st.number_input("Triangle Radial Resolution",
                                   help="Sets the radial resolution for the top bottom meshes of each produced triangle.", min_value=30, max_value=100, value=st_or_default("radial_res3d", 50))
    if radial_res3d:
        st.session_state["radial_res3d"] = radial_res3d

    x_res3d = st.number_input("Triangle X Resolution", help="Sets the xy resolution for the sides of each produced triangle.",
                              min_value=30, max_value=100, value=st_or_default("x_res3d", 50))
    if x_res3d:
        st.session_state["x_res3d"] = x_res3d

    y_res3d = st.number_input("Triangle Y Resolution", help="Sets the z resolution for the sides of each produced triangle.",
                              min_value=30, max_value=100, value=st_or_default("y_res3d", 50))
    if y_res3d:
        st.session_state["y_res3d"] = y_res3d
