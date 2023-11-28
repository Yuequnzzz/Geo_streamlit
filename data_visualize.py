import streamlit as st
import geopandas as gpd
import numpy as np
import pydeck as pdk
import os
import pandas as pd
from shapely import wkt
from copy import deepcopy



def get_all_test_id(grid_type):
    """
    This function gets all the test IDs
    :return: list, all the test IDs
    """
    # specify the path to the data
    path = "data/" + grid_type + "/"
    # get all the files in the path
    list_files = os.listdir(path)
    # get all the possible test IDs, that is remove the "_nodes" and "_edges" from the file names
    list_ids = [i[:-6] for i in list_files]
    # remove the duplicates
    list_ids = list(set(list_ids))
    # sort the list
    list_ids.sort()
    return list_ids


# create a class containing the functions to visualize the data
class GridVisualize:
    def __init__(self, grid_type, test_id):
        self.grid_type = grid_type
        self.test_id = test_id
        self.nodes, self.edges = self.select_test_id()

    def select_test_id(self):
        """
        This function creates a box to type in the ID of the test case
        :return: selected nodes and edges
        """
        path = "data/" + self.grid_type + "/"
        file_n, file_e = self.test_id + "_nodes", self.test_id + "_edges"
        nodes_gdf, edges_gdf = gpd.read_file(path + file_n), gpd.read_file(path + file_e)
        # rename the x and y columns
        nodes_gdf.rename(columns={'x': 'longitude', 'y': 'latitude'}, inplace=True)
        # convert the x y (epsg=2056) to lat long with geopandas
        edges_gdf['geometry'] = edges_gdf['geometry'].to_crs(epsg=4326)
        nodes_gdf['geometry'] = nodes_gdf['geometry'].to_crs(epsg=4326)
        # get the lat long from the geometry
        nodes_gdf['latitude'] = nodes_gdf['geometry'].y
        nodes_gdf['longitude'] = nodes_gdf['geometry'].x
        return nodes_gdf, edges_gdf

    def get_routed_nodes_in_edges(self):
        """
        This function gets the nodes in the edges
        :return: list, the nodes in the edges
        """
        list_path = [list(i.coords) for i in self.edges.geometry]
        list_routed_nodes = [item for sublist in list_path for item in sublist]
        return list_routed_nodes

    def get_initial_middle_point(self):
        """
        This function gets the initial middle point of the network
        :return: tuple, (lat, long), the initial middle point of the network
        """
        # find the max and min of the lat and long in the nodes
        max_lat_n, min_lat_n = self.nodes['latitude'].max(), self.nodes['latitude'].min()
        max_lon_n, min_lon_n = self.nodes['longitude'].max(), self.nodes['longitude'].min()
        # find the max and min of the lat and long in the edges
        list_routed_nodes = self.get_routed_nodes_in_edges()
        max_lat_e, min_lat_e = max([i[1] for i in list_routed_nodes]), min([i[1] for i in list_routed_nodes])
        max_lon_e, min_lon_e = max([i[0] for i in list_routed_nodes]), min([i[0] for i in list_routed_nodes])
        # get the midpoint
        max_lat, min_lat = max(max_lat_n, max_lat_e), min(min_lat_n, min_lat_e)
        max_lon, min_lon = max(max_lon_n, max_lon_e), min(min_lon_n, min_lon_e)
        midpoint = (np.average([max_lat, min_lat]), np.average([max_lon, min_lon]))
        return midpoint

    def get_initial_zoom(self):
        # find nodes in the edges
        list_routed_nodes = self.get_routed_nodes_in_edges()
        # add the nodes in the edges to the nodes dataframe
        nodes_in_edges = pd.DataFrame(list_routed_nodes, columns=['longitude', 'latitude'])
        nodes_all = pd.concat([self.nodes[['longitude', 'latitude']], nodes_in_edges])
        cv = pdk.data_utils.compute_view(points=nodes_all)
        initial_zoom = cv.zoom
        return initial_zoom

    def data_preprocessing_for_drawing(self):
        """
        This function preprocesses the data, including adding the size of the nodes, the color of the nodes, and mark
        the substation
        :return: the preprocessed nodes and edges
        """
        # data preprocessing
        if self.grid_type == 'MV':
            size_scale = 50  # todo: automatically adjust the size scale
        else:
            size_scale = 500
        self.nodes['size'] = self.nodes['el_dmd'] * size_scale
        self.nodes['r'], self.nodes['g'], self.nodes['b'] = 255, 0, 0
        # mark the substation
        osmid_sub = self.nodes[self.nodes['source']].index[0]
        self.nodes.loc[osmid_sub, 'size'] = self.nodes['size'].max() * 1.1
        self.nodes.loc[osmid_sub, 'r'] = 105
        self.nodes.loc[osmid_sub, 'g'] = 204
        self.nodes.loc[osmid_sub, 'b'] = 164
        return self.nodes, self.edges

    def draw_layers(self, pitch=None):
        """
        This function draws the layers, including ScatterplotLayer and PathLayer
        :return:
        """
        # data preprocessing
        self.nodes, self.edges = self.data_preprocessing_for_drawing()
        # # provide the map styles TODO: CANNOT CHANGE THE MAP STYLE
        # available_map_styles = {'road': pdk.map_styles.ROAD, 'satellite': "mapbox://styles/mapbox/satellite-v9"}
        # # add buttons to choose the map style
        # map_style = st.selectbox("Map style", list(available_map_styles.keys()))
        # get the map style
        mapstyle = st.sidebar.selectbox(
            "Choose Map Style for %s:" % self.test_id,
            options=["light", "dark", "satellite", "road"],
            format_func=str.capitalize,
            key=self.grid_type,
        )
        # add a layer to show the edges and nodes
        pydeck_layers = pdk.Deck(
            map_style=f"{mapstyle}",  # pdk.map_styles.ROAD,
            initial_view_state={
                "latitude": self.get_initial_middle_point()[0],
                "longitude": self.get_initial_middle_point()[1],
                "zoom": self.get_initial_zoom(),
                "pitch": pitch,
            },
            layers=[
                pdk.Layer(
                    "PathLayer",
                    data=[list(i.coords) for i in self.edges.geometry],
                    get_filled_color=[0, 255, 0],
                    pickable=True,
                    width_min_pixels=2,
                    get_width=0.1,
                    get_path='-'

                ),
                pdk.Layer(
                    "ScatterplotLayer",
                    data=self.nodes,
                    pickable=True,
                    opacity=0.8,
                    stroked=True,
                    filled=True,
                    radius_scale=10,
                    radius_min_pixels=1,
                    radius_max_pixels=20,
                    line_width_min_pixels=0.2,
                    get_position=['longitude', 'latitude'],
                    get_radius="size",
                    get_fill_color=['r', 'g', 'b'],
                    get_line_color=[0, 0, 0],
                ),
            ]
        )
        st.write(pydeck_layers)
        # add a button to make the map back to the initial view
        if st.button('Reset the map for %s' % self.test_id):
            pydeck_layers.update()
        return pydeck_layers

    def get_statistics(self):
        """
        This function gets the statistics of the demand
        :return:
        """
        # get the min, max, average and standard deviation of the demand, among which remove the non-demand nodes
        self.nodes['el_dmd'] = self.nodes['el_dmd'].replace(0, np.nan)
        min_dmd, max_dmd, avg_dmd, std_dmd = self.nodes['el_dmd'].min(), self.nodes['el_dmd'].max(), \
                                             self.nodes['el_dmd'].mean(), self.nodes['el_dmd'].std()
        # get the total demand and the number of demand nodes
        total_dmd, num_dmd = self.nodes['el_dmd'].sum(), self.nodes['el_dmd'].count()
        return min_dmd, max_dmd, avg_dmd, std_dmd, total_dmd, num_dmd

    def show_statistics(self):
        """
        This function shows the statistics of the demand
        :return:
        """
        # get the statistics of the demand in the unit of MW
        min_dmd, max_dmd, avg_dmd, std_dmd, total_dmd, num_dmd = self.get_statistics()
        # show the statistics in a table
        if self.grid_type=='MV':
            st.write("The statistics of the demand (MW)")
            df = {'min': [min_dmd], 'max': [max_dmd], 'average': [avg_dmd], 'standard deviation': [std_dmd],
                  'total': [total_dmd],
                  'number of loads': [num_dmd]}
        else:
            st.write("The statistics of the demand (kW)")
            df = {'min': [min_dmd*1000], 'max': [max_dmd*1000], 'average': [avg_dmd*1000],
                  'standard deviation': [std_dmd*1000],
                  'total': [total_dmd*1000],
                  'number of loads': [num_dmd]}
        df = pd.DataFrame(df)
        st.write(df)
        return


if __name__ == '__main__':
    # --------------------------- MV network ---------------------------
    # set the title of the page
    st.title("The MV network")
    # get all the test IDs
    list_ids = get_all_test_id('MV')
    # create a text field to type in the ID of the test case
    st.subheader("Please enter the ID of the test case")
    test_case = st.text_input("test case ID", list_ids[0])
    # check if the test case ID is valid
    if test_case not in list_ids:
        st.write("Please enter a valid test case ID")
        st.stop()

    # create a checkbox that can be clicked to show all possible test IDs
    if st.checkbox('Show all possible test IDs'):

        # write them into a table of ten columns
        if len(list_ids) % 5 != 0:
            # add some empty elements to make the length of the list a multiple of 10
            list_ids = list_ids + [''] * (5 - len(list_ids) % 5)
        st.table(pd.DataFrame(np.array(list_ids).reshape(-1, 5), columns=['col 1', 'col 2', 'col 3', 'col 4', 'col 5']))
        # darken the background of the table
        st.markdown(""" <style>
        table td:nth-child(1) {
            background-color: #e6e6e6;
        }
        table td:nth-child(2) {
            background-color: #e6e6e6;
        }
        table td:nth-child(3) {
            background-color: #e6e6e6;
        }
        table td:nth-child(4) {
            background-color: #e6e6e6;
        }
        table td:nth-child(5) {
            background-color: #e6e6e6;
        }
        table td:nth-child(6) {
        background-color: #e6e6e6;
        }
        </style> """, unsafe_allow_html=True)

    # create a object of the class
    mv = GridVisualize('MV', test_case)
    # add a slider to adjust the pitch
    pitch = st.slider('Pitch', 0, 60, 30)
    # draw the layers
    mv_layers = mv.draw_layers(pitch=pitch)
    # add some legend for substation and demand nodes, including a logo with the color
    st.write("🟢Substation  🔴Demand node")
    # show the statistics in a table
    mv.show_statistics()
    # process the shown nodes, transform the geometry to wkt
    mv_show_nodes = deepcopy(mv.nodes)
    mv_show_nodes.geometry.to_wkt()  # todo: not solved
    # add a checkbox that can be clicked to show the raw data
    if st.checkbox('Show raw data of MV %s' % test_case):

        # mv.nodes['str_geo'] = mv.nodes.geometry.apply(lambda x: wkt.dumps(x))
        st.dataframe(pd.DataFrame(mv_show_nodes))  # todo: cannot scroll down
        # st.write(mv.edges)

    # --------------------------- LV network ---------------------------
    # set the title of the page
    st.title("The LV network")
    # get all the test IDs
    list_ids_lv = get_all_test_id('LV')
    # create a box to type in the ID of the test case
    st.subheader("Please enter the ID of the test case")
    test_case_lv = st.selectbox("test case ID", list_ids_lv)
    # create a object of the class
    lv = GridVisualize('LV', test_case_lv)
    # draw the layers
    lv_layers = lv.draw_layers()
    # add some legend for substation and demand nodes, including a logo with the color
    st.write("🟢Substation  🔴Demand node")
    # show the statistics in a table
    lv.show_statistics()
    # add a checkbox that can be clicked to show the raw data
    if st.checkbox('Show raw data of MV %s' % test_case_lv):
        st.write(lv.nodes)
        st.write(lv.edges)

