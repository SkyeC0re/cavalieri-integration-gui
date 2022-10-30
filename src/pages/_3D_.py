import plotly
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly
from cavInt.cav3d import display
from cmath import cos, sin
import streamlit as st

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
    color = F_COLOR,
)
dictionary_inherit(PTLY_F_MESH_DEF, PTLY_MESH_DEF)


PTLY_F_MINOR_LINE_DEF = dict(
    line=dict(color = LINE_COLOR),
)
dictionary_inherit(PTLY_F_MINOR_LINE_DEF, PTLY_MINOR_LINE_DEF)


PTLY_F_MAJOR_LINE_DEF = dict(
    line=dict(color = LINE_COLOR),
)
dictionary_inherit(PTLY_F_MAJOR_LINE_DEF, PTLY_MAJOR_LINE_DEF)


PTLY_BASE_MESH_DEF = dict(
    color = BASE_COLOR,
)
dictionary_inherit(PTLY_BASE_MESH_DEF, PTLY_MESH_DEF)


PTLY_BASE_MINOR_LINE_DEF = dict(
    line=dict(color = LINE_COLOR),
)
dictionary_inherit(PTLY_BASE_MINOR_LINE_DEF, PTLY_MINOR_LINE_DEF)


PTLY_BASE_MAJOR_LINE_DEF = dict(
    line=dict(color = LINE_COLOR),
)
dictionary_inherit(PTLY_BASE_MAJOR_LINE_DEF, PTLY_MAJOR_LINE_DEF)


PTLY_SIDES_MESH_DEF = dict(
    color = SIDES_COLOR,
)
dictionary_inherit(PTLY_SIDES_MESH_DEF, PTLY_MESH_DEF)


PTLY_SIDES_MINOR_LINE_DEF = dict(
    line=dict(color = LINE_COLOR),
)
dictionary_inherit(PTLY_SIDES_MINOR_LINE_DEF, PTLY_MINOR_LINE_DEF)


PTLY_SIDES_MAJOR_LINE_DEF = dict(
    line=dict(color = LINE_COLOR),
)
dictionary_inherit(PTLY_SIDES_MAJOR_LINE_DEF, PTLY_MAJOR_LINE_DEF)

PTLY_2D_S_LINE_DEF = dict(
    line=dict(color = F_COLOR),
)
dictionary_inherit(PTLY_2D_S_LINE_DEF, PTLY_2D_LINE_DEF)

PTLY_2D_R_LINE_DEF = dict(
    line=dict(color = BASE_COLOR),
)
dictionary_inherit(PTLY_2D_R_LINE_DEF, PTLY_2D_LINE_DEF)

PTLY_2D_TRIANGLE_LINE_DEF = dict(
    line=dict(color = LINE_COLOR, width=1.5),
    legendgroup='2DTriagLines'
)
dictionary_inherit(PTLY_2D_TRIANGLE_LINE_DEF, PTLY_2D_LINE_DEF)

st.set_page_config(layout="wide")

input_tab, config_tab = st.tabs(['Parameters', 'Config'])


with input_tab:
    st.write('# Parameters')
    f_label_col, f_input_col, c1_label_col, c1_input_col, c2_label_col, c2_input_col = st.columns([1, 5, 1, 5, 1, 5])

    with f_label_col:
        st.write('###')
        st.latex('f(x, y)=')
    
    with f_input_col:
        f_ti = st.text_input('', value='x + 2')

    with c1_label_col:
        st.write('###')
        st.latex('c_x(z)=')

    with c1_input_col:
        c1_ti = st.text_input('', value='-z')

    with c2_label_col:
        st.write('###')
        st.latex('c_y(z)=')

    with c2_input_col:
        c2_ti = st.text_input('', value='0')

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
        polygons_input = st.text_area('', value=initial_polygon_set)

    

with config_tab:
    st.write('# Config')

def f(x):
    return eval(str(f_ti), {}, {
        'x':x[0],
        'y':x[1],
        'sin': np.sin,
        'cos':np.cos,
        'pi': np.pi
    })

def c_1(z):
    return eval(str(c1_ti), {}, {
        'z':z,
        'sin':np.sin,
        'cos':np.cos,
        'pi': np.pi
    })

def c_2(z):
    return eval(str(c2_ti), {}, {
        'z':z,
        'sin':np.sin,
        'cos':np.cos,
        'pi': np.pi
    })

def c(z):
    return np.array([c_1(z), c_2(z)])

gen_button = st.button('Generate')

cav_int_tab, regions_tab = st.tabs(['3D Integral', 'R and S'])

if gen_button:

    polygons = []
    for polygon in eval('[' + polygons_input + ']', {}, {}):
        polygons.append(np.array(polygon, dtype=np.float64))
    
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
    side_curtains, triangle_info = display.generate_integral(f, c, polygons)
    make_triangle_legend = True
    for x_grid_bot, x_grid, y_grid in triangle_info:
        i, j, k = triangulate_grid_surf(*y_grid.shape)
        fig_3d_integral.add_trace(
            go.Mesh3d(x=x_grid[..., 0].flatten(), y=x_grid[..., 1].flatten(), z=y_grid.flatten(), i=i, j=j, k=k, **PTLY_F_MESH_DEF),
                row=1,
                col=1
        )
        fig_3d_integral.add_trace(
            go.Mesh3d(x=x_grid_bot[..., 0].flatten(), y=x_grid_bot[..., 1].flatten(), z=np.zeros_like(y_grid.flatten()), i=i, j=j, k=k, **PTLY_BASE_MESH_DEF),
                row=1,
                col=1
        )
        fig_3d_integral.add_trace(
            go.Scatter3d(x=x_grid_bot[0, :, 0], y=x_grid_bot[0, :, 1], z=np.zeros_like(y_grid[0, :]), **PTLY_BASE_MINOR_LINE_DEF),
            row = 1,
            col = 1
        )
        fig_3d_integral.add_trace(
            go.Scatter3d(x=x_grid[0, :, 0], y=x_grid[0, :, 1], z=y_grid[0, :], **PTLY_F_MINOR_LINE_DEF),
            row = 1,
            col = 1
        )
        
        fig_2d_regions.add_trace(
            go.Scatter(x=x_grid[0, :, 0], y=x_grid[0, :, 1], name='Triangulation', **PTLY_2D_TRIANGLE_LINE_DEF),
            row=1, 
            col=1
        )

        fig_2d_regions.add_trace(
            go.Scatter(x=x_grid_bot[0, :, 0], y=x_grid_bot[0, :, 1], **dictionary_inherit(dict(name='Triangulation', showlegend=make_triangle_legend), PTLY_2D_TRIANGLE_LINE_DEF)),
            row=1,
            col=2
        )

        make_triangle_legend = False
    for x_grid, y_grid in side_curtains:
        i, j, k = triangulate_grid_surf(*y_grid.shape)
        fig_3d_integral.add_trace(
        go.Mesh3d(x=x_grid[..., 0].flatten(), y=x_grid[..., 1].flatten(), z=y_grid.flatten(), i=i, j=j, k=k, **PTLY_SIDES_MESH_DEF)
        )
        fig_3d_integral.add_trace(
            go.Scatter3d(x=x_grid[:, 0, 0], y=x_grid[:, 0, 1], z=y_grid[:, 0], **PTLY_SIDES_MAJOR_LINE_DEF)
        )
        fig_3d_integral.add_trace(
            go.Scatter3d(x=x_grid[-1, :, 0], y=x_grid[-1, :, 1], z=y_grid[-1, :], **PTLY_F_MAJOR_LINE_DEF)
        )
        fig_3d_integral.add_trace(
            go.Scatter3d(x=x_grid[0, :, 0], y=x_grid[0, :, 1], z=y_grid[0, :], **PTLY_F_MAJOR_LINE_DEF)
        )

        fig_2d_regions.add_trace(
            go.Scatter(x=x_grid[-1, :, 0], y=x_grid[-1, :, 1], **PTLY_2D_S_LINE_DEF),
            row=1,
            col=1
        )

        fig_2d_regions.add_trace(
            go.Scatter(x=x_grid[0, :, 0], y=x_grid[0, :, 1], **PTLY_2D_R_LINE_DEF),
            row=1,
            col=2
        )

    fig_3d_integral['layout']['height']=800

    
    with cav_int_tab:
        st.plotly_chart(fig_3d_integral, True)

    with regions_tab:
        st.plotly_chart(fig_2d_regions, True)