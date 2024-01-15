import folium
import streamlit as st
from streamlit_folium import st_folium, folium_static
import streamlit as st
import geopandas as gpd
import numpy as np
import pydeck as pdk
import os
import pandas as pd
from copy import deepcopy
import json
import ast
from IPython.display import display

# # center on Liberty Bell, add marker
# m = folium.Map(location=[39.949610, -75.150282], zoom_start=16)
# folium.Marker(
#     [39.949610, -75.150282], popup="Liberty Bell", tooltip="Liberty Bell"
# ).add_to(m)
#
# # call to render Folium map in Streamlit
# st_data = st_folium(m, width=725)
# # st.write(st_data)
# st.write(st_data['last_clicked'])
# if st_data['last_clicked']:
#     # just show the lat/lon we clicked
#     lat = st_data['last_clicked']['lat']
#     lon = st_data['last_clicked']['lng']
#     st.write('You clicked: ', lat, lon)


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
        if self.grid_type == 'MV':
            path = self.grid_type + "/"
        else:
            path = self.grid_type + "/" + dict_test_id_folder[self.test_id] + "/"

        if isinstance(self.test_id, str):
            file_n, file_e = self.test_id + "_nodes", self.test_id + "_edges"
            nodes_gdf, edges_gdf = gpd.read_file(path + file_n), gpd.read_file(path + file_e)
        elif isinstance(self.test_id, list) & (len(self.test_id) != 0):
            # consider the canton case
            nodes_gdf = gpd.GeoDataFrame()
            edges_gdf = gpd.GeoDataFrame()
            for i in self.test_id:
                sub_nodes = gpd.read_file(path + i + "_nodes")
                sub_edges = gpd.read_file(path + i + "_edges")
                nodes_gdf = pd.concat([nodes_gdf, sub_nodes])
                edges_gdf = pd.concat([edges_gdf, sub_edges])
        else:
            st.write("There is no such grid within the canton")
            raise ValueError

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
        # get the map style
        mapstyle = st.sidebar.selectbox(
            "Choose Map Style for %s:" % self.grid_type,
            options=["road", "satellite"],
            format_func=str.capitalize,
            key=self.grid_type,
        )
        available_map_styles = {'road': pdk.map_styles.ROAD, 'satellite': 'mapbox://styles/mapbox/satellite-v9'}
        # add a layer to show the edges and nodes
        pathlayer = pdk.Layer(
                    "PathLayer",
                    data=[list(i.coords) for i in self.edges.geometry],
                    get_filled_color=[0, 255, 0],
                    pickable=True,
                    width_min_pixels=2,
                    get_width=0.1,
                    get_path='-'

                )
        scatterplotlayer = pdk.Layer(
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
                )
        if self.grid_type == 'MV':
            polygon_plot = pd.read_csv('data_processing/canton_coordinates_plot.csv')
            polygon_plot['coordinates'] = polygon_plot['coordinates'].apply(ast.literal_eval)
        else:
            # todo
            polygon_plot = pd.read_csv('data_processing/canton_coordinates_plot_LV.csv')
            polygon_plot['coordinates'] = polygon_plot['coordinates'].apply(ast.literal_eval)
        polygonlayer = pdk.Layer(
                    "PolygonLayer",
                    polygon_plot,
                    # id="geojson",
                    # opacity=0.8,
                    # stroked=False,
                    get_polygon="coordinates",
                    # filled=True,
                    # extruded=True,
                    # wireframe=True,
                    get_fill_color="fill_color",
                    # get_line_color=[255, 255, 255],
                    auto_highlight=True,
                    pickable=True,
                )
        textlayer = pdk.Layer(
                    "TextLayer",
                    data=polygon_plot,
                    get_position=['centroid_x', 'centroid_y'],
                    get_text="NAME",
                    # get_color=[255, 255, 255],
                    get_angle=0,
                    get_size=16,
                    get_alignment_baseline="'center'",
                )
        pydeck_layers = pdk.Deck(
            map_style=f"{available_map_styles[mapstyle]}",
            initial_view_state={
                "latitude": self.get_initial_middle_point()[0],
                "longitude": self.get_initial_middle_point()[1],
                "zoom": self.get_initial_zoom(),
                "pitch": pitch,
            },
            layers=[
                pathlayer,
                scatterplotlayer,
                polygonlayer,
                textlayer,
            ]
        )
        st.write(pydeck_layers)
        # add a button to make the map back to the initial view
        if st.button('Reset the map for %s' % self.grid_type):
            pydeck_layers.update()
        return pydeck_layers

    def style_function_nodes(self, feature):
        props = feature.get('properties')
        # scale the size of the nodes between 0 and 10
        size = props.get('size')/self.nodes['size'].max()*10
        markup = f"""
                <div style="width: {size}px;
                            height: {size}px;
                            background-color: blue;">
                </div>
            </div>
            </a>
        """
        return {"html": markup}

    def draw_layers_folium(self):
        """
        This function draws the layers with folium
        """
        self.nodes, self.edges = self.data_preprocessing_for_drawing()
        # specify the initial location
        m = folium.Map(location=self.get_initial_middle_point(), zoom_start=self.get_initial_zoom(), opacity=0.1)
        # draw the nodes
        folium.GeoJson(
            self.nodes,
            name="Nodes",
            # marker=folium.Marker(icon=folium.DivIcon()),
            # style_function=self.style_function_nodes,
            marker=folium.Circle(radius=40, fill_color="red", fill_opacity=0.4, weight=1),
            style_function=lambda x: {"radius": (x['properties']['size']),
                                      },
            zoom_on_click=True,
        ).add_to(m)
        # draw the edges
        self.edges['geometry'] = self.edges['geometry'].to_crs(epsg=4326)
        raw_edges = [list(i.coords) for i in self.edges.geometry]
        raw_edges = [[[i[1], i[0]] for i in j] for j in raw_edges]

        folium.PolyLine(
            locations=raw_edges,
            color="green",
            weight=2,
            opacity=1,
        ).add_to(m)

        # # check where the last click was
        # st_data = st_folium(m, key='nodes_edges', width=1200, height=600)
        # lat, lon = None, None
        # if st_data['last_clicked']:
        #     # just show the lat/lon we clicked
        #     lat = st_data['last_clicked']['lat']
        #     lon = st_data['last_clicked']['lng']
        #     st.write('You clicked: ', lat, lon)
        # # determine which canton the point is in
        # point = gpd.points_from_xy([lon], [lat])
        canton_boundary = gpd.read_file('data_processing/canton_union.geojson')
        canton_boundary['geometry'] = canton_boundary['geometry'].to_crs(epsg=4326)
        # add the canton centroid
        canton_boundary['centroid'] = canton_boundary['geometry'].centroid

        # # determine which canton the point is in
        # canton = None
        # for k, c in enumerate(canton_boundary['NAME']):
        #     if point.within(canton_boundary[canton_boundary['NAME'] == c].geometry[k])[0]:
        #         canton = c
        #         st.write(f'You are in {c}')
        #         break

        # if canton is None:
        #     st.write('You are not in any canton')
        # else:
        #     st.write('The canton boundary is:')
        #     canton_geo = canton_boundary[canton_boundary['NAME'] == canton]['geometry']
            # canton_geo = gpd.GeoSeries(canton_geo).simplify(tolerance=0.001)
            # canton_geo = canton_geo.to_json()
            # folium.GeoJson(
            #     canton_geo,
            #     name="Canton",
            #     style_function=lambda x: {"fillColor": "grey"},
            #     zoom_on_click=True,
            #     tooltip=f"{canton}",
            #     marker=folium.Marker(icon=folium.DivIcon()),
            # ).add_to(m)
            # # add the
            # st_folium(m, key='canton', width=1200, height=600)

        # add 'selected_canton' to the session state
        if 'selected_canton' not in st.session_state:
            st.session_state['selected_canton'] = None

        # update the selected canton with the last clicked object
        session_keys = list(st.session_state.keys())
        # remove the keys that include ['selected_canton', 'MV_text_input_id', 'MV_text_input_canton', 'MV_checkbox']
        object_name = [i for i in session_keys if i not in ['selected_canton', 'MV_text_input_id', 'MV_text_input_canton', 'MV_checkbox']]
        if len(object_name) != 0:
            st.session_state['selected_canton'] = st.session_state[object_name[0]]['last_object_clicked_tooltip']

        # add the feature group
        fg = folium.FeatureGroup(name="NAME")
        for n in canton_boundary['NAME']:
            fg.add_child(
                folium.Marker(
                    location=[canton_boundary[canton_boundary['NAME'] == n]['centroid'].y,
                              canton_boundary[canton_boundary['NAME'] == n]['centroid'].x],
                    popup=f"{n}",
                    tooltip=f"{n}",
                )
            )
            if n == st.session_state["selected_canton"]:
                fg.add_child(
                    folium.features.GeoJson(
                        canton_boundary[canton_boundary['NAME'] == n]['geometry']),
                    folium.Marker(
                        icon=folium.Icon(color="green"),
                    )
                )
        st_data = st_folium(
            m,
            feature_group_to_add=fg,
            width=1200,
            height=500,
        )

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

    def show_raw_data(self):
        """
        This function shows the raw data of the network
        :return:
        """
        # data preprocessing, transform the geometry to wkt
        # for nodes
        show_nodes = deepcopy(self.nodes)
        show_nodes['geo'] = show_nodes.geometry.to_wkt()
        # replace the geometry with the wkt
        show_nodes.drop(columns=['geometry'], inplace=True)
        show_nodes.rename(columns={'geo': 'geometry'}, inplace=True)
        # for edges
        show_edges = deepcopy(self.edges)
        show_edges['geo'] = show_edges.geometry.to_wkt()
        # replace the geometry with the wkt
        show_edges.drop(columns=['geometry'], inplace=True)
        show_edges.rename(columns={'geo': 'geometry'}, inplace=True)

        # show the raw data
        if st.checkbox('Show raw data of %s' % self.grid_type):
            st.subheader('Nodes')
            st.dataframe(pd.DataFrame(show_nodes))
            st.subheader('Edges')
            st.dataframe(pd.DataFrame(show_edges))
        return


if __name__ == '__main__':
    data_path = 'data_processing/'
    # ----------------------- Process LV ----------------------
    # load the dictionary connecting the test ID and the folder name
    with open(data_path + 'file_folder_lv.json') as json_file:
        dict_test_id_folder = json.load(json_file)

    # ----------------------- Process cantons ----------------------
    # load the dictionary connecting the canton and the grid
    with open(data_path + 'dict_canton_grid_MV.json') as json_file:
        dict_canton_grid_mv = json.load(json_file)
    with open(data_path + 'dict_canton_grid_LV.json') as json_file:
        dict_canton_grid_lv = json.load(json_file)
    list_canton_names = list(dict_canton_grid_mv.keys())

    # --------------------------- MV network ---------------------------
    # set the title of the page
    st.title("The MV network")
    # get all the test IDs
    with open(data_path + 'list_test_id_MV.json') as json_file:
        list_ids_mv = json.load(json_file)
    # get the ids and the corresponding canton names
    table_ids_canton = pd.read_csv(data_path + 'table_grid_canton_MV.csv')
    # create a text field to type in the ID of the test case
    st.subheader("Please choose what you want to show")
    # create a text field to type in canton name in the same line
    cols = st.columns(2)
    with cols[0]:
        test_id = st.text_input('test case ID', list_ids_mv[0], key='MV_text_input_id')
    with cols[1]:
        test_canton = st.selectbox('canton name', list_canton_names, key='MV_text_input_canton')
    # create a checkbox that can be clicked to show all possible test IDs
    if st.checkbox('Show all possible test IDs', key='MV_checkbox'):
        # show_all_possible_test_ids(list_ids_mv)
        st.dataframe(table_ids_canton)

    # add a single checkbox to choose the test case
    genre = st.radio(
        "Which one do you want to show?",
        ["***Single grid***", "***Canton region***"],
        captions=["show selected test id.", "show the whole canton region"])

    # check if the test case ID is valid
    if test_id not in list_ids_mv:
        st.write("Please enter a valid test case ID")
        st.stop()

    if genre == "***Single grid***":
        test_case = test_id
    else:
        # get the test IDs in the canton
        test_case = dict_canton_grid_mv[test_canton]

    # create an object of the class
    mv = GridVisualize('MV', test_case)
    # draw the layers
    mv_layers = mv.draw_layers_folium()
    # add some legend for substation and demand nodes, including a logo with the color
    st.write("🟢Substation  🔴Demand node")
    # show the statistics in a table
    mv.show_statistics()
    # add a checkbox that can be clicked to show the raw data
    mv.show_raw_data()


# todo:
# 1. let the user choose what to show
# either click or select bar
# 2. show the map
# 3. download

# todo: opacity of the background


