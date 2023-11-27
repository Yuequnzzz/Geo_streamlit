import streamlit as st
import geopandas as gpd
import numpy as np
import pydeck as pdk
import os


def load_data(path, file_nodes, file_edges):
    """
    This function loads the data from the input path and files
    :param path: the path to the data
    :param file_nodes: the file name of the nodes
    :param file_edges: the file name of the edges
    :return: the nodes and edges as geopandas dataframes
    """
    nodes_gdf, edges_gdf = gpd.read_file(path + file_nodes), gpd.read_file(path + file_edges)
    # rename the x and y columns
    nodes_gdf.rename(columns={'x': 'longitude', 'y': 'latitude'}, inplace=True)
    # convert the x y (epsg=2056) to lat long with geopandas
    edges_gdf['geometry'] = edges_gdf['geometry'].to_crs(epsg=4326)
    nodes_gdf['geometry'] = nodes_gdf['geometry'].to_crs(epsg=4326)
    # get the lat long from the geometry
    nodes_gdf['latitude'] = nodes_gdf['geometry'].y
    nodes_gdf['longitude'] = nodes_gdf['geometry'].x

    return nodes_gdf, edges_gdf


if __name__ == '__main__':

    # set the title of the page
    st.title("The MV network")

    # specify the path to the data
    path = "data/MV/"
    # get all the files in the path
    list_files = os.listdir(path)
    # get all the possible test IDs, that is remove the "_nodes" and "_edges" from the file names
    list_ids = [i[:-6] for i in list_files]
    # create a box to type in the ID of the test case
    st.subheader("Please enter the ID of the test case")
    test_case = st.text_input("test case ID", "1_0")
    # check if the ID is valid
    if test_case not in list_ids:
        st.write("The test case ID is not valid, please enter a valid ID")
        st.stop()
    # load the data
    file_n, file_e = test_case + "_nodes", test_case + "_edges"

    # list_files = ['1_0', '2_0', '2_1', '3_0', '4_0', '5_0', '6_0', '7_0', '111-1_1_4', '111-1_2_5', '111-2_0_3', '111-2_1_5', '111-3_0_4', '111-3_1_5']
    # test_case = 9
    # file_n, file_e = list_files[test_case] + "_nodes", list_files[test_case] + "_edges"
    nodes, edges = load_data(path, file_n, file_e)

    # data preprocessing
    midpoint = (np.average(nodes['latitude']), np.average(nodes['longitude']))
    nodes['size'] = nodes['el_dmd'] * 2000
    nodes['r'], nodes['g'], nodes['b'] = 255, 0, 0
    nodes['name'] = 1
    # mark the substation
    osmid_sub = nodes[nodes['source']].index[0]
    nodes.loc[osmid_sub, 'size'] = nodes['size'].max() + 10
    nodes.loc[osmid_sub, 'g'] = 140
    nodes.loc[osmid_sub, 'name'] = 'substation'

    # add a layer to show the edges and nodes
    st.write(pdk.Deck(
        map_style="mapbox://styles/mapbox/light-v9",
        initial_view_state={
            "latitude": midpoint[0],
            "longitude": midpoint[1],
            "zoom": 11,
            "pitch": 50,
        },
        layers=[
            pdk.Layer(
                "PathLayer",
                data=[list(i.coords) for i in edges.geometry],
                get_filled_color=[0, 255, 0],
                pickable=True,
                width_min_pixels=2,
                get_width=0.1,
                get_path='-'

            ),
            pdk.Layer(
                "ScatterplotLayer",
                data=nodes,
                get_position=['longitude', 'latitude'],
                opacity=0.8,
                scale=20,
                pickable=True,
                stroked=True,
                filled=True,
                get_radius='size',
                get_fill_color=['r', 'g', 'b'],
            ),
            # pdk.Layer(
            #     "TextLayer",
            #     data=nodes,
            #     get_position=['longitude', 'latitude'],
            #     get_text='name',
            #     get_size=12,
            #     get_color=[0, 0, 0],
            #     get_angle=0,
            #     pickable=True,
            #     getTextAnchor='"middle"',
            #     get_alignment_baseline='"bottom"'
            # ),


        ]
    ))
    # # draw the nodes, and set the proper node size
    # st.map(nodes[['latitude', 'longitude']], size=1)

    # show the raw data
    if st.checkbox("show raw data", False):
        st.subheader('Raw nodes')
        st.write(nodes)
        st.subheader('Raw edges')
        st.write(edges)  # todo: cannot show all the edges, too many








