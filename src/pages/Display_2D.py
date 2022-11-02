import streamlit as st
from cavint import display_cav2d, display_cav2d_rs
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd

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

input_cav_tab, input_rs_tab = st.tabs(
    ['Cavalieri Input', 'Riemann-Stieltjes Input']
)


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
                                 st.session_state["x_res2d"],
                                 st.session_state["y_res2d"],
                                 st.session_state["rf_iters"],
                                 st.session_state["integ_iters"],
                                 st.session_state["tol"],
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

        displays = display_cav2d_rs(f_expr, g_expr, intervals_expr,
                                    True,
                                    st.session_state["x_res2d"],
                                    st.session_state["y_res2d"],
                                    st.session_state["rf_iters"],
                                    st.session_state["integ_iters"],
                                    st.session_state["tol"],
                                    )


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
    g_graphs_fig['layout']['height'] = 500

    intermediate_step = 10
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
        xy_grid = np.array(display.cav_grid)
        dgv = np.array(display.dgv)
        lg_i += 1
        group_title = f"[{xy_grid[-1, 0, 0]:.2e}, {xy_grid[-1, -1, 0]:.2e}]"
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
        for i in range(intermediate_step, len(xy_grid[0])-1, intermediate_step):
            cav_inter.append(go.Scatter(
                x=xy_grid[:, i, 0], y=xy_grid[:, i, 1], showlegend=make_legend, **spec_inter
            ))
            make_legend = False

        cav_base.append(go.Scatter(
            x=xy_grid[0, :, 0], y=np.zeros_like(xy_grid[0, :, 1]), showlegend=True, **spec_base
        ))

        cav_sides.append(go.Scatter(
            x=xy_grid[:, 0, 0], y=xy_grid[:, 0, 1], showlegend=True, **spec_sides
        ))

        cav_sides.append(go.Scatter(
            x=xy_grid[:, -1, 0], y=xy_grid[:, -1, 1], showlegend=False, **spec_sides
        ))

        cav_f.append(go.Scatter(
            x=xy_grid[-1, :, 0], y=xy_grid[-1, :, 1], showlegend=True, **spec_f
        ))

        # Equivalent Riemann-Stieltjes integral over S.

        # f(x*)*g'(x*) graph
        rs_f.append(go.Scatter(
            x=xy_grid[-1, :, 0],
            y=xy_grid[-1, :, 1] * dgv,
            showlegend=False,
            **spec_f
        ))

        for i in range(intermediate_step, len(xy_grid[0])-1, intermediate_step):
            rs_inter.append(
                go.Scatter(
                    x=xy_grid[-1, [i, i], 0],
                    y=[0, xy_grid[-1, i, 1] * dgv[i]],
                    showlegend=False,
                    **spec_inter
                )
            )

        # Left boundary
        rs_sides.append(go.Scatter(
            x=xy_grid[-1, [0, 0], 0],
            y=[0, xy_grid[-1, 0, 1] * dgv[0]],
            showlegend=False,
            **spec_sides
        ))

        # Right boundary
        rs_sides.append(go.Scatter(
            x=xy_grid[-1, [-1, -1], 0],
            y=[0, xy_grid[-1, -1, 1] * dgv[-1]],
            showlegend=False,
            **spec_sides
        ))

        # Base
        rs_base.append(
            go.Scatter(
                x=xy_grid[-1, :, 0],
                y=np.zeros_like(dgv),
                showlegend=False,
                **spec_base
            )
        )

        # Equivalent Riemann integral over R.

        # f(h(x)) graph
        r_f.append(
            go.Scatter(
                x=xy_grid[0, :, 0],
                y=xy_grid[-1, :, 1],
                showlegend=False,
                **spec_f
            )
        )

        for i in range(intermediate_step, len(xy_grid[0])-1, intermediate_step):
            r_inter.append(
                go.Scatter(
                    x=xy_grid[0, [i, i], 0],
                    y=[0, xy_grid[-1, i, 1]],
                    showlegend=False,
                    **spec_inter
                )
            )

        r_sides.append(
            go.Scatter(
                x=xy_grid[0, [0, 0], 0],
                y=[0, xy_grid[-1, 0, 1]],
                showlegend=False,
                **spec_sides,
            )
        )

        r_sides.append(
            go.Scatter(
                x=xy_grid[0, [-1, -1], 0],
                y=[0, xy_grid[-1, -1, 1]],
                showlegend=False,
                **spec_sides
            )
        )

        r_base.append(
            go.Scatter(
                x=xy_grid[0, :, 0],
                y=np.zeros_like(xy_grid[0, :, 0]),
                showlegend=False,
                **spec_base
            )
        )

        # g graph
        g_graphs_fig.add_trace(
            go.Scatter(
                x=xy_grid[-1, :, 0],
                y=xy_grid[-1, :, 1],
                **PTLY_F_MAJOR_GRID_DEF
            ),
            row=1,
            col=1
        )

        # dg graph
        g_graphs_fig.add_trace(
            go.Scatter(
                x=xy_grid[-1, :, 0],
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
        st.dataframe(interval_integ_df)
