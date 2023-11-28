import streamlit as st
import geopandas as gpd
import numpy as np
import pydeck as pdk
import os
import pandas as pd


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

    def draw_layers(self):
        """
        This function draws the layers, including ScatterplotLayer and PathLayer
        :return:
        """
        # data preprocessing
        self.nodes, self.edges = self.data_preprocessing_for_drawing()
        # add a layer to show the edges and nodes
        st.write(pdk.Deck(
            map_style="mapbox://styles/mapbox/light-v9",
            initial_view_state={
                "latitude": self.get_initial_middle_point()[0],
                "longitude": self.get_initial_middle_point()[1],
                "zoom": self.get_initial_zoom(),
                # "pitch": 80,
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
        ))
        return

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
        # get the statistics of the demand
        min_dmd, max_dmd, avg_dmd, std_dmd, total_dmd, num_dmd = self.get_statistics()
        # show the statistics in a table
        st.write("The statistics of the demand (kW)")
        df = {'min': [min_dmd], 'max': [max_dmd], 'average': [avg_dmd], 'standard deviation': [std_dmd],
              'total': [total_dmd],
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
    # create a box to type in the ID of the test case
    st.subheader("Please enter the ID of the test case")
    test_case = st.selectbox("test case ID", list_ids)
    # create a object of the class
    mv = GridVisualize('MV', test_case)
    # draw the layers
    mv.draw_layers()
    # add some legend for substation and demand nodes, including a logo with the color
    st.write("🟢Substation  🔴Demand node")
    # show the statistics in a table
    mv.show_statistics()

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
    lv.draw_layers()
    # add some legend for substation and demand nodes, including a logo with the color
    st.write("🟢Substation  🔴Demand node")
    # show the statistics in a table
    lv.show_statistics()

