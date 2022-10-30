import streamlit as st
from cavint import display_cav2d
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

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


with input_cav_tab:
    st.write('# Parameters')
    f_label_col_cav, f_input_col_cav, c_label_col_cav, c_input_col_cav = \
        st.columns([1, 5, 1, 5])

    with f_label_col_cav:
        st.write('###')
        st.latex('f(x)=')

    with f_input_col_cav:
        f_input_cav = st.text_input('', value='x + 2', key='f_input_cav')

    with c_label_col_cav:
        st.write('###')
        st.latex('c(y)=')

    with c_input_col_cav:
        c_input_cav = st.text_input('', value='-y', key='c_input_cav')

    x_intervals_label_col_cav, x_intervals_input_col_cav = st.columns([1, 11])

    with x_intervals_label_col_cav:
        st.write('###')
        st.latex('S=')

    with x_intervals_input_col_cav:
        x_intervals_input_cav = st.text_area('', value="[0, 1]",
                                             key='x_intervals_input_cav')

    gen_button_cav = st.button('Generate', key='gen_button_cav')

    if gen_button_cav:
        f_expr = str(f_input_cav)
        c_expr = str(c_input_cav)
        intervals_expr = str(x_intervals_input_cav)

        cav_displays = display_cav2d(f_expr, c_expr, intervals_expr,
                                     True,
                                     100,
                                     50,
                                     100,
                                     100,
                                     1e-9
                                     )


with input_rs_tab:
    st.write('# Parameters')
    f_label_col_rs, f_input_col_rs, g_label_col_rs, g_input_col_rs = \
        st.columns([1, 5, 1, 5])

    with f_label_col_rs:
        st.write('###')
        st.latex('f(x)=')

    with f_input_col_rs:
        f_input_rs = st.text_input('', value='x + 2', key='f_input_rs')

    with g_label_col_rs:
        st.write('###')
        st.latex('g(x)=')

    with g_input_col_rs:
        g_input_rs = st.text_input('', value='2*x', key='g_input_rs')

    x_intervals_label_col_rs, x_intervals_input_col_rs = st.columns([1, 11])

    with x_intervals_label_col_rs:
        st.write('###')
        st.latex('S=')

    with x_intervals_input_col_rs:
        x_intervals_input_rs = st.text_area('', value="[0, 1]",
                                            key='x_intervals_input_rs')

    gen_button_rs = st.button('Generate', key='gen_button_rs')

    if gen_button_rs:
        def f_expr(x):
            return eval(str(f_input_rs), {}, {
                'x': x,
                'sin': anp.sin,
                'cos': anp.cos,
                'pi': anp.pi,
                'ln': anp.log
            })

        def g(x):
            return eval(str(g_input_rs), {}, {
                'x': x,
                'sin': anp.sin,
                'cos': anp.cos,
                'pi': anp.pi,
                'ln': anp.log
            })
        intervals_expr = anp.array(
            eval('[' + str(x_intervals_input_rs) + ']', {}, {})
        )

        cav_displays = display.generate_integ_rs(f_expr, g, intervals_expr,
                                                 make_rs=True,
                                                 make_r=True, make_g=True,
                                                 make_deriv_g=True,
                                                 compute_integ=True)


with config_tab:
    st.write('# Config')


if 'cav_displays' in locals():
    cav_integ_fig = make_subplots(2, 2, shared_xaxes='all', shared_yaxes='all',
                                  subplot_titles=("Cavalieri Integral",
                                                  "f(h(x))",
                                                  "f(x*)g'(x*)", ""),
                                  horizontal_spacing=0.05, vertical_spacing=0.1
                                  )
    cav_integ_fig['layout']['xaxis2_showticklabels'] = True
    cav_integ_fig['layout']['height'] = 800

    g_graphs_fig = make_subplots(2, 1, shared_xaxes=True,
                                 subplot_titles=("g(x*)", "g'(x*)"),
                                 vertical_spacing=0.1)
    g_graphs_fig['layout']['height'] = 500

    intermediate_step = 10
    integ_accu_value = 0.0
    integ_accu_err = 0.0
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
    for cav_display in cav_displays:
        integ_accu_value += cav_display.integ_value[0]
        integ_accu_err += cav_display.integ_value[1]
        xy_grid = np.array(cav_display.cav_grid)
        dgv = np.array(cav_display.dgv)
        lg_i += 1
        spec_f = dictionary_inherit(
            dict(
                legendgroup=f"f{lg_i}",
                legendgrouptitle=dict(text=f"Interval {lg_i}")
            ),
            PTLY_F_MAJOR_GRID_DEF
        )
        spec_base = dictionary_inherit(
            dict(
                legendgroup=f"f{lg_i}",
                legendgrouptitle=dict(text=f"Interval {lg_i}"),
            ),
            PTLY_BASE_MAJOR_GRID_DEF
        )
        spec_sides = dictionary_inherit(
            dict(
                legendgroup=f"f{lg_i}",
                legendgrouptitle=dict(text=f"Interval {lg_i}")
            ),
            PTLY_SIDES_MAJOR_GRID_DEF
        )
        spec_inter = dictionary_inherit(
            dict(
                legendgroup=f"f{lg_i}",
                legendgrouptitle=dict(text=f"Interval {lg_i}")
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

        # g_graphs_fig.add_trace(
        #     go.Scatter(
        #         x=cav_display.g_interval[0],
        #         y=cav_display.g_interval[1],
        #         **PTLY_F_MAJOR_GRID_DEF
        #     ),
        #     row=1,
        #     col=1
        # )

        # g_graphs_fig.add_trace(
        #     go.Scatter(
        #         x=cav_display.deriv_g_interval[0],
        #         y=cav_display.deriv_g_interval[1],
        #         **PTLY_F_MAJOR_GRID_DEF
        #     ),
        #     row=2,
        #     col=1
        # )

    cav_integ_fig.add_traces(cav_inter + cav_sides + cav_base + cav_f, 1, 1)
    cav_integ_fig.add_traces(rs_inter + rs_sides + rs_base + rs_f, 2, 1)
    cav_integ_fig.add_traces(r_inter + r_sides + r_base + r_f, 1, 2)
    st.markdown(f"""\
# Output

## Integral Information:

| Parameter | Value |
| --- | --- |
| Accumilative Integration Value | {integ_accu_value} |
| Accumilative Integration Error | {integ_accu_err} |
| Intervals | {len(cav_displays)} |

## Cavalieri Integral Display:
""")
    st.plotly_chart(cav_integ_fig, True)
    st.plotly_chart(g_graphs_fig, True)
