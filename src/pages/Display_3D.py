import plotly
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from cavint import display_cav3d
from cmath import cos, sin
import streamlit as st
import pandas as pd


def triangulate_grid_surf(rows, cols):
    bot_l_y, bot_l_x = np.mgrid[0:rows-1, 0:cols-1]
    top_r_y, top_r_x = np.mgrid[1:rows, 1:cols]
    top_l_y, top_l_x = np.mgrid[1:rows, 0:cols-1]
    bot_r_y, bot_r_x = np.mgrid[0:rows-1, 1:cols]

    bot_l_y = bot_l_y.flatten()
    bot_l_x = bot_l_x.flatten()
    top_r_y = top_r_y.flatten()
    top_r_x = top_r_x.flatten()
    top_l_y = top_l_y.flatten()
    top_l_x = top_l_x.flatten()
    bot_r_y = bot_r_y.flatten()
    bot_r_x = bot_r_x.flatten()

    i_set = np.concatenate((bot_l_y*cols + bot_l_x, bot_l_y*cols + bot_l_x))
    j_set = np.concatenate((top_r_y*cols + top_r_x, top_l_y*cols + top_l_x))
    k_set = np.concatenate((bot_r_y*cols + bot_r_x, top_r_y*cols + top_r_x))

    return (i_set, j_set, k_set)


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
    opacity=0.49,
    color='#FFFFFF',
    hoverinfo='x+y+z',
    showlegend=False,
)

PTLY_MINOR_LINE_DEF = dict(
    opacity=1,
    mode='lines',
    line=dict(color='#000000', width=1.5),
    hoverinfo='x+y+z',
    showlegend=False,
)

PTLY_MAJOR_LINE_DEF = dict(
    opacity=1,
    mode='lines',
    line=dict(color='#000000', width=5.0),
    hoverinfo='x+y+z',
    showlegend=False,
)

PTLY_2D_LINE_DEF = dict(
    opacity=1,
    mode='lines',
    line=dict(color='#000000', width=2.5),
    hoverinfo='x+y+z',
    showlegend=False,
)

PTLY_F_MESH_DEF = dict(
    color=F_COLOR,
)
dictionary_inherit(PTLY_F_MESH_DEF, PTLY_MESH_DEF)


PTLY_F_MINOR_LINE_DEF = dict(
    line=dict(color=LINE_COLOR),
)
dictionary_inherit(PTLY_F_MINOR_LINE_DEF, PTLY_MINOR_LINE_DEF)


PTLY_F_MAJOR_LINE_DEF = dict(
    line=dict(color=LINE_COLOR),
)
dictionary_inherit(PTLY_F_MAJOR_LINE_DEF, PTLY_MAJOR_LINE_DEF)


PTLY_BASE_MESH_DEF = dict(
    color=BASE_COLOR,
)
dictionary_inherit(PTLY_BASE_MESH_DEF, PTLY_MESH_DEF)


PTLY_BASE_MINOR_LINE_DEF = dict(
    line=dict(color=LINE_COLOR),
)
dictionary_inherit(PTLY_BASE_MINOR_LINE_DEF, PTLY_MINOR_LINE_DEF)


PTLY_BASE_MAJOR_LINE_DEF = dict(
    line=dict(color=LINE_COLOR),
)
dictionary_inherit(PTLY_BASE_MAJOR_LINE_DEF, PTLY_MAJOR_LINE_DEF)


PTLY_SIDES_MESH_DEF = dict(
    color=SIDES_COLOR,
)
dictionary_inherit(PTLY_SIDES_MESH_DEF, PTLY_MESH_DEF)


PTLY_SIDES_MINOR_LINE_DEF = dict(
    line=dict(color=LINE_COLOR),
)
dictionary_inherit(PTLY_SIDES_MINOR_LINE_DEF, PTLY_MINOR_LINE_DEF)


PTLY_SIDES_MAJOR_LINE_DEF = dict(
    line=dict(color=LINE_COLOR),
)
dictionary_inherit(PTLY_SIDES_MAJOR_LINE_DEF, PTLY_MAJOR_LINE_DEF)

PTLY_2D_S_LINE_DEF = dict(
    line=dict(color=F_COLOR),
)
dictionary_inherit(PTLY_2D_S_LINE_DEF, PTLY_2D_LINE_DEF)

PTLY_2D_R_LINE_DEF = dict(
    line=dict(color=BASE_COLOR),
)
dictionary_inherit(PTLY_2D_R_LINE_DEF, PTLY_2D_LINE_DEF)

PTLY_2D_TRIANGLE_LINE_DEF = dict(
    line=dict(color=LINE_COLOR, width=1.5),
    legendgroup='2DTriagLines'
)
dictionary_inherit(PTLY_2D_TRIANGLE_LINE_DEF, PTLY_2D_LINE_DEF)

st.set_page_config(layout="wide")

input_tab, config_tab = st.tabs(['Parameters', 'Config'])


with input_tab:
    st.write('# Parameters')
    f_label_col, f_input_col, c1_label_col, c1_input_col, c2_label_col, c2_input_col = st.columns([
                                                                                                  1, 5, 1, 5, 1, 5])

    with f_label_col:
        st.write('###')
        st.latex('f(x, y)=')

    with f_input_col:
        f_ti = st.text_input('*', label_visibility="hidden",
                             value='x + 2', key="f_input_3d")

    with c1_label_col:
        st.write('###')
        st.latex('c_x(z)=')

    with c1_input_col:
        c1_ti = st.text_input('*', label_visibility="hidden",
                              value='-z', key="c1_input_3d")

    with c2_label_col:
        st.write('###')
        st.latex('c_y(z)=')

    with c2_input_col:
        c2_ti = st.text_input('*', label_visibility="hidden",
                              value='0', key="c2_input_3d")

    polygons_label_col, polygons_input_col = st.columns([1, 11])

    with polygons_label_col:
        st.write('###')
        st.write('Polygons:')

    initial_polygon_set = r'''[
    [-1,-1],
    [-1,1],
    [1,1],
    [1,-1]
],
[
    [-0.5, -0.5],
    [-0.5, 0.5],
    [0.5, 0.5],
    [0.5, -0.5]
]'''

    with polygons_input_col:
        polygons_input = st.text_area(
            '*', label_visibility="hidden", value=initial_polygon_set, key="poly_input_3d")


with config_tab:
    st.write('# Config')

f_expr = str(f_ti)
c1_expr = str(c1_ti)
c2_expr = str(c2_ti)
poly_set_expr = str(polygons_input)

gen_button = st.button('Generate')

if gen_button:
    fig_3d_integral = make_subplots(
        1,
        1,
        specs=[[{'type': 'mesh3d'}]],
        subplot_titles=['3D Cavalieri Integral'],
    )

    fig_2d_regions = make_subplots(
        1,
        2,
        shared_xaxes='all',
        shared_yaxes='all',
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}]
        ],
        subplot_titles=['S Region', 'R Region'],
    )
    displays = display_cav3d(f_expr, c1_expr, c2_expr,
                             poly_set_expr, True, st.session_state["radial_res3d"],
                             st.session_state["x_res3d"], st.session_state["y_res3d"],
                             st.session_state["integ_iters"], st.session_state["tol"])
    make_triangle_legend = True
    curtain_id_map = dict()
    triag_integ_df = []
    accu_sum = 0
    accu_err = 0
    for display in displays:
        iv, ierr = display.integ_value
        accu_sum += iv
        accu_err += ierr
        triag = np.array(display.triag)

        triag_integ_df.append([
            f"({triag[0, 0]:.2e}, {triag[0, 1]:.2e})--({triag[1, 0]:.2e}, {triag[1, 1]:.2e})--({triag[2, 0]:.2e}, {triag[2, 1]:.2e})",
            iv,
            ierr
        ])
        top_mesh = np.array(display.top_mesh)
        bot_mesh = np.array(display.bot_mesh)
        i, j, k = triangulate_grid_surf(bot_mesh.shape[0], bot_mesh.shape[1])
        # Top mesh
        fig_3d_integral.add_trace(
            go.Mesh3d(x=top_mesh[..., 0].flatten(), y=top_mesh[..., 1].flatten(
            ), z=top_mesh[..., 2].flatten(), i=i, j=j, k=k, **PTLY_F_MESH_DEF),
            row=1,
            col=1
        )

        # Top mesh tracing
        fig_3d_integral.add_trace(
            go.Scatter3d(x=top_mesh[-1, :, 0], y=top_mesh[-1, :, 1],
                         z=top_mesh[-1, :, 2], **PTLY_F_MINOR_LINE_DEF),
            row=1,
            col=1
        )

        xv = bot_mesh[..., 0].flatten()
        # Bottom mesh
        fig_3d_integral.add_trace(
            go.Mesh3d(x=xv, y=bot_mesh[..., 1].flatten(
            ), z=np.zeros_like(xv), i=i, j=j, k=k, **PTLY_BASE_MESH_DEF),
            row=1,
            col=1
        )

        # Bottom mesh tracing
        fig_3d_integral.add_trace(
            go.Scatter3d(x=bot_mesh[-1, :, 0], y=bot_mesh[-1, :, 1], z=np.zeros_like(
                bot_mesh[-1, :, 0]), **PTLY_BASE_MINOR_LINE_DEF),
            row=1,
            col=1
        )

        curtains = np.array(display.curtains)
        for curtain in curtains:
            a = curtain[-1, 0, :]
            b = curtain[-1, -1, :]

            if a[0] < b[0]:
                curtain_id = str([a, b])
            elif a[0] > b[0]:
                curtain_id = str([b, a])
            elif a[1] < b[1]:
                curtain_id = str([a, b])
            else:
                curtain_id = str([b, a])
            if curtain_id in curtain_id_map:
                curtain_id_map[curtain_id][0] += 1
            else:
                curtain_id_map[curtain_id] = [0, curtain]

    triag_integ_df.append(
        ["Total", accu_sum, accu_err])
    triag_integ_df = pd.DataFrame(triag_integ_df,
                                  columns=['Triangle', 'Integration Value', 'Estimated Error'], index=None)
    for repititions, curtain in curtain_id_map.values():
        i, j, k = triangulate_grid_surf(curtain.shape[0], curtain.shape[1])
        if repititions == 0:
            # Curtain mesh
            fig_3d_integral.add_trace(
                go.Mesh3d(x=curtain[..., 0].flatten(), y=curtain[..., 1].flatten(
                ), z=curtain[..., 2].flatten(), i=i, j=j, k=k, **PTLY_SIDES_MESH_DEF)
            )

            # Curtain top side major line
            fig_3d_integral.add_trace(
                go.Scatter3d(x=curtain[:, 0, 0], y=curtain[:, 0, 1],
                             z=curtain[:, 0, 2], **PTLY_SIDES_MAJOR_LINE_DEF)
            )

            # Curtain right side major line
            fig_3d_integral.add_trace(
                go.Scatter3d(x=curtain[-1, :, 0], y=curtain[-1, :, 1],
                             z=curtain[-1, :, 2], **PTLY_F_MAJOR_LINE_DEF)
            )

            # Curtain left side major line
            fig_3d_integral.add_trace(
                go.Scatter3d(x=curtain[0, :, 0], y=curtain[0, :, 1],
                             z=curtain[0, :, 2], **PTLY_F_MAJOR_LINE_DEF)
            )

            # S Edge Line
            fig_2d_regions.add_trace(
                go.Scatter(x=curtain[-1, :, 0],
                           y=curtain[-1, :, 1], **PTLY_2D_S_LINE_DEF),
                row=1,
                col=1
            )

            # R Edge Line
            fig_2d_regions.add_trace(
                go.Scatter(x=curtain[0, :, 0], y=curtain[0, :, 1],
                           **PTLY_2D_R_LINE_DEF),
                row=1,
                col=2
            )
        else:
            # S Triangulation Line
            fig_2d_regions.add_trace(
                go.Scatter(x=curtain[-1, :, 0], y=curtain[-1, :, 1],
                           name='Triangulation', **PTLY_2D_TRIANGLE_LINE_DEF),
                row=1,
                col=1
            )

            # R Triangulation Line
            fig_2d_regions.add_trace(
                go.Scatter(x=curtain[0, :, 0], y=curtain[0, :, 1], **dictionary_inherit(dict(
                    name='Triangulation', showlegend=make_triangle_legend), PTLY_2D_TRIANGLE_LINE_DEF)),
                row=1,
                col=2
            )

            make_triangle_legend = False

    fig_3d_integral['layout']['height'] = 800

cav_int_tab, regions_tab, integ_tab = st.tabs(
    ['3D Integral', 'R and S', 'Triangle Integration Values'])

with cav_int_tab:
    if not "fig_3d_integral" in locals():
        fig_3d_integral = st.session_state.get("fig_3d_integral")

    if fig_3d_integral:
        st.plotly_chart(fig_3d_integral, True)
        st.session_state["fig_3d_integral"] = fig_3d_integral

with regions_tab:
    if not "fig_2d_regions" in locals():
        fig_2d_regions = st.session_state.get("fig_2d_regions")

    if fig_2d_regions:
        st.plotly_chart(fig_2d_regions, True)
        st.session_state["fig_2d_regions"] = fig_2d_regions

with integ_tab:
    if not "triag_integ_df" in locals():
        triag_integ_df = st.session_state.get("triag_integ_df")

    if not triag_integ_df is None:
        st.dataframe(triag_integ_df)
        st.session_state["triag_integ_df"] = triag_integ_df
