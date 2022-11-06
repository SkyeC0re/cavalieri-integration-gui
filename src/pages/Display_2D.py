import time
import streamlit as st
from cavint import display_cav2d, display_cav2d_rs
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
pd.set_option('display.float_format', '{:.3e}'.format)

LGROUP_F = 1
LGROUP_INTERMEDIATE = 2
LGROUP_SIDES = 3
LGROUP_BASE = 4


def dictionary_inherit(child, parent):
    for key in parent:
        if key in child:
            if isinstance(child[key], dict) and isinstance(parent[key], dict):
                dictionary_inherit(child[key], parent[key])
        else:
            child[key] = parent[key]

    return child


F_COLOR = '#eb5e28'
BASE_COLOR_OLD = '#252422'
BASE_COLOR = '#4287f5'
SIDES_COLOR = '#403d39'
LINE_COLOR = '#000000'

PTLY_MESH_DEF = dict(
    opacity=1.0,
    color='#FFFFFF',
    hoverinfo='x+y+z',
)

PTLY_MINOR_GRID_DEF = dict(
    opacity=1,
    mode='lines',
    line=dict(color='#000000', dash='dash',  width=1.2, shape='spline'),
    hoverinfo='x+y+z',
)

PTLY_MAJOR_GRID_DEF = dict(
    opacity=1,
    mode='lines',
    line=dict(color='#000000', width=2.5, shape='spline'),
    hoverinfo='x+y+z',
)

PTLY_F_MESH_DEF = dict(
    color=F_COLOR,
)
dictionary_inherit(PTLY_F_MESH_DEF, PTLY_MESH_DEF)


PTLY_F_MINOR_GRID_DEF = dict(
    line=dict(color=LINE_COLOR),
)
dictionary_inherit(PTLY_F_MINOR_GRID_DEF, PTLY_MINOR_GRID_DEF)


PTLY_F_MAJOR_GRID_DEF = dict(
    line=dict(color=F_COLOR),
    name='Function curve'
)
dictionary_inherit(PTLY_F_MAJOR_GRID_DEF, PTLY_MAJOR_GRID_DEF)


PTLY_F_PACKAGE_SPEC = dict(
    spec=PTLY_F_MESH_DEF,
    minor_spec=PTLY_F_MINOR_GRID_DEF,
    major_spec=PTLY_F_MAJOR_GRID_DEF,
)


PTLY_BASE_MESH_DEF = dict(
    color=BASE_COLOR,
)
dictionary_inherit(PTLY_BASE_MESH_DEF, PTLY_MESH_DEF)


PTLY_BASE_MINOR_GRID_DEF = dict(
    line=dict(color=LINE_COLOR),
)
dictionary_inherit(PTLY_BASE_MINOR_GRID_DEF, PTLY_MINOR_GRID_DEF)


PTLY_BASE_MAJOR_GRID_DEF = dict(
    line=dict(color=BASE_COLOR),
    name='Integral Base'
)
dictionary_inherit(PTLY_BASE_MAJOR_GRID_DEF, PTLY_MAJOR_GRID_DEF)

PTLY_BASE_PACKAGE_SPEC = dict(
    spec=PTLY_BASE_MESH_DEF,
    minor_spec=PTLY_BASE_MINOR_GRID_DEF,
    major_spec=PTLY_BASE_MAJOR_GRID_DEF,
)


PTLY_SIDES_MESH_DEF = dict(
    color=SIDES_COLOR,
)
dictionary_inherit(PTLY_SIDES_MESH_DEF, PTLY_MESH_DEF)


PTLY_SIDES_MINOR_GRID_DEF = dict(
    line=dict(color=SIDES_COLOR),
    name='Intermediate curves'
)
dictionary_inherit(PTLY_SIDES_MINOR_GRID_DEF, PTLY_MINOR_GRID_DEF)


PTLY_SIDES_MAJOR_GRID_DEF = dict(
    line=dict(color=SIDES_COLOR),
    name='Boundary Curves',
)
dictionary_inherit(PTLY_SIDES_MAJOR_GRID_DEF, PTLY_MAJOR_GRID_DEF)


PTLY_SIDES_PACKAGE_SPEC = dict(
    spec=PTLY_SIDES_MESH_DEF,
    minor_spec=PTLY_SIDES_MINOR_GRID_DEF,
    major_spec=PTLY_SIDES_MAJOR_GRID_DEF,
)

st.set_page_config(layout="wide")

input_cav_tab, input_rs_tab, config_tab = st.tabs(
    ['Cavalieri Input', 'Riemann-Stieltjes Input', 'Config']
)

with config_tab:
    st.write('# Config')
    x_res2d = st.number_input("X Resolution", help="Sets the x resolution for each produced interval.", key='cfg2d_x_res',
                              min_value=50, max_value=500, value=100)

    y_res2d = st.number_input("Y Resolution", help="Sets the y resolution for each produced interval.", key='cfg2d_y_res',
                              min_value=50, max_value=500, value=100)

    interm_cs = st.number_input("Intermediate Curves", help="How many intermediate c_x(y) curves to plot.", key='cfg2d_interm_cs',
                                min_value=0, max_value=20, value=0)

    rf_iters = st.number_input("Maximum Root Finding Iterations",
                               help="Sets the maximum root finding iterations before failure.", key='cfg2d_rf_iters', min_value=10, max_value=1000, value=100)

    integ_iters = st.number_input("Maximum Integration Iterations",
                                  help="Sets the maximum adaptive Gauss-Kronrod iterations before failure.", key='cfg2d_integ_iters', min_value=10, max_value=1000, value=100)

    tol = 10 ** st.number_input("Integration Tolerance Base 10 Exponent", help="Sets the absolute integration tolerance exponent.", key='cfg2d_tol',
                                min_value=-12, max_value=0, value=-9)


with input_cav_tab:
    st.write('# Parameters')
    f_label_col_cav, f_input_col_cav, c_label_col_cav, c_input_col_cav = \
        st.columns([1, 5, 1, 5])

    with f_label_col_cav:
        st.write('###')
        st.latex('f(x)=')

    with f_input_col_cav:
        f_input_cav = st.text_input(
            '*', label_visibility="hidden", value='x + 2', key='f_input_cav')

    with c_label_col_cav:
        st.write('###')
        st.latex('c(y)=')

    with c_input_col_cav:
        c_input_cav = st.text_input(
            '*', label_visibility="hidden", value='-y', key='c_input_cav')

    x_intervals_label_col_cav, x_intervals_input_col_cav = st.columns([1, 11])

    with x_intervals_label_col_cav:
        st.write('###')
        st.latex('S=')

    with x_intervals_input_col_cav:
        x_intervals_input_cav = st.text_area('*', label_visibility="hidden", value="[0, 1]",
                                             key='x_intervals_input_cav')

    gen_button_cav = st.button('Generate', key='gen_button_cav')

    if gen_button_cav:
        f_expr = str(f_input_cav)
        c_expr = str(c_input_cav)
        intervals_expr = str(x_intervals_input_cav)

        displays = display_cav2d(f_expr, c_expr, intervals_expr,
                                 True,
                                 x_res2d,
                                 y_res2d,
                                 interm_cs,
                                 rf_iters,
                                 integ_iters,
                                 tol,
                                 )


with input_rs_tab:
    st.write('# Parameters')
    f_label_col_rs, f_input_col_rs, g_label_col_rs, g_input_col_rs = \
        st.columns([1, 5, 1, 5])

    with f_label_col_rs:
        st.write('###')
        st.latex('f(x)=')

    with f_input_col_rs:
        f_input_rs = st.text_input(
            '*', label_visibility="hidden", value='x + 2', key='f_input_rs')

    with g_label_col_rs:
        st.write('###')
        st.latex('g(x)=')

    with g_input_col_rs:
        g_input_rs = st.text_input(
            '*', label_visibility="hidden", value='2*x', key='g_input_rs')

    x_intervals_label_col_rs, x_intervals_input_col_rs = st.columns([1, 11])

    with x_intervals_label_col_rs:
        st.write('###')
        st.latex('S=')

    with x_intervals_input_col_rs:
        x_intervals_input_rs = st.text_area('*', label_visibility="hidden", value="[0, 1]",
                                            key='x_intervals_input_rs')

    gen_button_rs = st.button('Generate', key='gen_button_rs')

    if gen_button_rs:
        f_expr = str(f_input_rs)
        g_expr = str(g_input_rs)
        intervals_expr = str(x_intervals_input_rs)
        start_time = time.time()
        displays = display_cav2d_rs(f_expr, g_expr, intervals_expr,
                                    True,
                                    x_res2d,
                                    y_res2d,
                                    interm_cs,
                                    rf_iters,
                                    integ_iters,
                                    tol,
                                    )
        end_time = time.time()
        st.write(f"Finished Rust in seconds: {end_time - start_time}")


if 'displays' in locals():
    cav_integ_fig = make_subplots(2, 2, shared_xaxes='all', shared_yaxes='all',
                                  subplot_titles=("Cavalieri Integral",
                                                  "f(h(x))",
                                                  "f(x*)g'(x*)", ""),
                                  horizontal_spacing=0.05, vertical_spacing=0.1
                                  )
    cav_integ_fig['layout']['xaxis2_showticklabels'] = True
    cav_integ_fig['layout']['height'] = 800
    cav_integ_fig.update_annotations(font_size=24)

    g_graphs_fig = make_subplots(2, 1, shared_xaxes=True,
                                 subplot_titles=("g(x*)", "g'(x*)"),
                                 vertical_spacing=0.1)
    g_graphs_fig['layout']['height'] = 800
    g_graphs_fig.update_annotations(font_size=24)

    cav_f = []
    cav_base = []
    cav_inter = []
    cav_sides = []
    rs_f = []
    rs_base = []
    rs_inter = []
    rs_sides = []
    r_f = []
    r_base = []
    r_inter = []
    r_sides = []
    lg_i = 0
    interval_integ_df = []
    accu_sum = 0.0
    accu_err = 0.0
    for display in displays:
        iv, ierr = display.integ_value
        accu_sum += iv
        accu_err += ierr

        interval_integ_df.append([
            f"[{display.a:.2e}, {display.b:.2e}]",
            iv,
            ierr
        ])
        xv = np.array(display.xv)
        fv = np.array(display.fv)
        gv = np.array(display.gv)
        dgv = np.array(display.dgv)
        cvs = display.cvs
        st.write(f"[{display.a}, {display.b}]")
        for i in range(len(cvs)):
            index, cv = cvs[i]
            cvs[i] = (index, np.array(cv))
        lg_i += 1
        group_title = f"[{xv[0]:.2e}, {xv[-1]:.2e}]"
        spec_f = dictionary_inherit(
            dict(
                legendgroup=f"f{lg_i}",
                legendgrouptitle=dict(text=group_title)
            ),
            PTLY_F_MAJOR_GRID_DEF
        )
        spec_base = dictionary_inherit(
            dict(
                legendgroup=f"f{lg_i}",
                legendgrouptitle=dict(text=group_title),
            ),
            PTLY_BASE_MAJOR_GRID_DEF
        )
        spec_sides = dictionary_inherit(
            dict(
                legendgroup=f"f{lg_i}",
                legendgrouptitle=dict(text=group_title)
            ),
            PTLY_SIDES_MAJOR_GRID_DEF
        )
        spec_inter = dictionary_inherit(
            dict(
                legendgroup=f"f{lg_i}",
                legendgrouptitle=dict(text=group_title)
            ),
            PTLY_SIDES_MINOR_GRID_DEF
        )

        make_legend = True
        # Intermediate c(y) graphs
        for i in range(1, len(cvs) - 1):
            _, cv = cvs[i]
            cav_inter.append(
                go.Scatter(
                    x=cv[:, 1],
                    y=cv[:, 0],
                    showlegend=make_legend,
                    **spec_inter
                )
            )
            make_legend = False

        # R x {0}
        cav_base.append(go.Scatter(
            x=gv, y=np.zeros_like(gv), showlegend=True, **spec_base
        ))

        # Boundary curve at `a`
        _, cv = cvs[0]
        cav_sides.append(go.Scatter(
            x=cv[:, 1], y=cv[:, 0], showlegend=True, **spec_sides
        ))

        # Boundary curve at `b`
        _, cv = cvs[-1]
        cav_sides.append(go.Scatter(
            x=cv[:, 1], y=cv[:, 0], showlegend=False, **spec_sides
        ))

        # S x f(S)
        cav_f.append(go.Scatter(
            x=xv, y=fv, showlegend=True, **spec_f
        ))

        # Equivalent Riemann-Stieltjes integral over S.

        # f(S)*g'(S)
        rs_f.append(go.Scatter(
            x=xv,
            y=fv * dgv,
            showlegend=False,
            **spec_f
        ))

        # Intermediate graphs
        for i in range(1, len(cvs) - 1):
            xindex, _ = cvs[i]
            rs_inter.append(
                go.Scatter(
                    x=xv[[xindex, xindex]],
                    y=[0, fv[xindex] * dgv[xindex]],
                    showlegend=False,
                    **spec_inter
                )
            )

        # Boundary at `a`
        rs_sides.append(go.Scatter(
            x=xv[[0, 0]],
            y=[0, fv[0] * dgv[0]],
            showlegend=False,
            **spec_sides
        ))

        # Boundary at `b`
        rs_sides.append(go.Scatter(
            x=xv[[-1, -1]],
            y=[0, fv[-1] * dgv[-1]],
            showlegend=False,
            **spec_sides
        ))

        # Base
        rs_base.append(
            go.Scatter(
                x=xv,
                y=np.zeros_like(xv),
                showlegend=False,
                **spec_base
            )
        )

        # Equivalent Riemann integral over R.

        # f(h(R))
        r_f.append(
            go.Scatter(
                x=gv,
                y=fv,
                showlegend=False,
                **spec_f
            )
        )

       # Intermediate graphs
        for i in range(1, len(cvs) - 1):
            xindex, _ = cvs[i]
            r_inter.append(
                go.Scatter(
                    x=gv[[xindex, xindex]],
                    y=[0, fv[xindex]],
                    showlegend=False,
                    **spec_inter
                )
            )

        r_sides.append(
            go.Scatter(
                x=gv[[0, 0]],
                y=[0, fv[0]],
                showlegend=False,
                **spec_sides,
            )
        )

        r_sides.append(
            go.Scatter(
                x=gv[[-1, -1]],
                y=[0, fv[-1]],
                showlegend=False,
                **spec_sides
            )
        )

        r_base.append(
            go.Scatter(
                x=gv,
                y=np.zeros_like(gv),
                showlegend=False,
                **spec_base
            )
        )

        # g graph
        g_graphs_fig.add_trace(
            go.Scatter(
                x=xv,
                y=gv,
                **PTLY_F_MAJOR_GRID_DEF
            ),
            row=1,
            col=1
        )

        # dg graph
        g_graphs_fig.add_trace(
            go.Scatter(
                x=xv,
                y=dgv,
                **PTLY_F_MAJOR_GRID_DEF
            ),
            row=2,
            col=1
        )

    interval_integ_df.append(
        ["Total", accu_sum, accu_err])
    interval_integ_df = pd.DataFrame(interval_integ_df,
                                     columns=['Interval', 'Integration Value', 'Estimated Error'], index=None)

    cav_integ_fig.add_traces(cav_inter + cav_sides + cav_base + cav_f, 1, 1)
    cav_integ_fig.add_traces(rs_inter + rs_sides + rs_base + rs_f, 2, 1)
    cav_integ_fig.add_traces(r_inter + r_sides + r_base + r_f, 1, 2)

    cav_int_tab, misc_graph_tab, integ_tab = st.tabs(
        ['Integral Graphs', 'Misc Graphs', 'Interval Integration Values'])
    with cav_int_tab:
        if cav_integ_fig:
            st.plotly_chart(cav_integ_fig, True)

    with misc_graph_tab:
        if g_graphs_fig:
            st.plotly_chart(g_graphs_fig, True)

    with integ_tab:
        st.dataframe(interval_integ_df.style.format('{:.3e}', subset=[
                     "Integration Value", "Estimated Error"]), use_container_width=True)
