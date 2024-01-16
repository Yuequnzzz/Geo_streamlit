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
        elif isinstance(self.test_id, list) and len(self.test_id) != 0:
            # consider the canton case or the multiple grids
            nodes_gdf = gpd.GeoDataFrame()
            edges_gdf = gpd.GeoDataFrame()
            for i in self.test_id:
                sub_nodes = gpd.read_file(path + i + "_nodes")
                # add a column to mark the substation
                sub_nodes['source_id'] = None
                index_substation = sub_nodes[sub_nodes['source']].index[0]
                sub_nodes.loc[index_substation, 'source_id'] = i

                sub_edges = gpd.read_file(path + i + "_edges")
                nodes_gdf = pd.concat([nodes_gdf, sub_nodes])
                edges_gdf = pd.concat([edges_gdf, sub_edges])

        elif isinstance(self.test_id, list) and len(self.test_id) == 0:
            st.write("💥Warning: you haven't enter a test case ID")
            nodes_gdf, edges_gdf = None, None
            st.stop()
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
        # record the substation's test ID

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
        size = props.get('size') / self.nodes['size'].max() * 10
        markup = f"""
                <div style="width: {size}px;
                            height: {size}px;
                            background-color: blue;">
                </div>
            </div>
            </a>
        """
        return {"html": markup}

    def draw_layers_folium(self, substation_show=False, grid_show=False):
        """
        This function draws the layers with folium
        """
        self.nodes, self.edges = self.data_preprocessing_for_drawing()
        # specify the initial location
        m = folium.Map(location=self.get_initial_middle_point(), zoom_start=self.get_initial_zoom(), opacity=0.1)
        if substation_show:
            # mark the substation
            substation_nodes = self.nodes[self.nodes['source']]
            for k, s in substation_nodes.iterrows():
                folium.Marker(
                    location=[s['latitude'],
                              s['longitude']],
                    popup=f"{s['source_id']}",
                    tooltip=f"{s['source_id']}",
                    icon=folium.Icon(color='green', icon='flash')
                ).add_to(m)
        if grid_show:
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
        # remove the keys
        object_name = [i for i in session_keys if
                       i not in ['selected_canton', 'MV_text_input_id', 'MV_input_canton_by_click',
                                 'MV_text_input_canton', 'MV_test_id_checkbox', 'MV_show_canton_grids',
                                 'MV_multiselect_grid_id', 'MV_checkbox_show_multi_grid']]
        if len(object_name) != 0:  # means that the map is loaded for the first time
            last_clicked = st.session_state[object_name[0]]['last_object_clicked_tooltip']
            if last_clicked is not None:
                st.session_state['selected_canton'] = last_clicked
                st.write('You clicked %s' % st.session_state['selected_canton'])

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
                )

        st_data = st_folium(
            m,
            feature_group_to_add=fg,
            width=1200,
            height=500,
        )

        return st_data['last_object_clicked_tooltip']

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
        if self.grid_type == 'MV':
            st.write("The statistics of the demand (MW)")
            df = {'min': [min_dmd], 'max': [max_dmd], 'average': [avg_dmd], 'standard deviation': [std_dmd],
                  'total': [total_dmd],
                  'number of loads': [num_dmd]}
        else:
            st.write("The statistics of the demand (kW)")
            df = {'min': [min_dmd * 1000], 'max': [max_dmd * 1000], 'average': [avg_dmd * 1000],
                  'standard deviation': [std_dmd * 1000],
                  'total': [total_dmd * 1000],
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
    table_ids_canton = pd.read_csv(data_path + 'table_grid_canton_MV.csv')

    # --------------------------- MV network ---------------------------
    # set the title of the page
    st.title("The MV network")
    # add a single checkbox to choose the test case
    genre = st.radio(
        "Which one do you want to show?",
        ["***Select grids by IDs***", "***Select grids by canton***"],
        captions=["show selected test ids", "show the whole canton region"],
        horizontal=True)
    # add a divider
    st.markdown("---")

    # --------------------------- canton region --------------------------
    # canton part, if the user chooses to show the canton region, first show the map with markers of the cantons,
    # letting the user choose the canton either by clicking or selecting the canton name
    # then show the map of the selected canton in the first map
    # add a checkbox that can be clicked to show the grids within the canton

    if genre == "***Select grids by canton***":

        cols = st.columns(2)
        with cols[0]:
            test_canton = st.selectbox('canton name', list_canton_names, index=None, placeholder='Please select',
                                       key='MV_text_input_canton')
            # update the session state
            st.session_state['selected_canton'] = test_canton
        with cols[1]:
            if st.checkbox('Show all grids in this canton', key='MV_show_canton_grids'):
                grid_option = True
                # check if the canton name is vacant
                if test_canton is None:
                    st.write("💥Warning: please select a canton")
                    st.stop()
            else:
                grid_option = False
            if st.checkbox('Show substations', key='MV_test_id_checkbox'):
                substation_option = True
                if test_canton is None:
                    st.write("💥Warning: please select a canton")
                    st.stop()

                #st.dataframe(table_ids_canton[table_ids_canton['canton_name'] == test_canton])
            else:
                substation_option = False

        if test_canton is not None:
            # create an object of the class
            mv = GridVisualize('MV', dict_canton_grid_mv[test_canton])
            # draw the layers
            mv_layers = mv.draw_layers_folium(grid_show=grid_option, substation_show=substation_option)

            # show the statistics in a table
            mv.show_statistics()
            # add a checkbox that can be clicked to show the raw data
            mv.show_raw_data()

    # --------------------------- single grid --------------------------
    # single grid part, if the user chooses to show the single grid,
    # first show the text field to type in the test ID,
    # then show the map of the selected test ID
    else:

        test_id = st.multiselect('test case IDs', table_ids_canton['grid_id'].values, key='MV_multiselect_grid_id')
        # add a dataframe to show the canton name of the selected test IDs
        if st.checkbox('Show data of selected test IDs', key='MV_checkbox_show_multi_grid'):
            st.dataframe(table_ids_canton[table_ids_canton['grid_id'].isin(test_id)])

        # create an object of the class
        mv = GridVisualize('MV', test_id)
        # draw the layers
        mv_layers = mv.draw_layers_folium(grid_show=True, substation_show=True)
        # show the statistics in a table
        mv.show_statistics()
        # add a checkbox that can be clicked to show the raw data
        mv.show_raw_data()

        # I wanna download the data with the selected test ID
        # add a button to download the data
        if st.button('Download the data of selected test IDs'):
            # create a folder to store the data
            os.system('mkdir data_download')
            # create empty geojson data file
            for i in test_id:
                os.system('touch data_download/' + i + '_nodes')
                os.system('touch data_download/' + i + '_edges')

            # copy the data file in the MV folder to the data_processing folder

            for i in test_id:
                os.system('cp MV/' + i + '_nodes data_download/' + i + '_nodes')
                os.system('cp MV/' + i + '_edges data_download/' + i + '_edges')
            # # zip the files
            # os.system('zip -r data_grids.zip data_download/')
            # # download the zip file
            # st.download_button(label='Download the data', data='data_processing.zip', mime='application/zip',
            #                    file_name='data_grids.zip')
            # but the zip is empty, why?



# todo:
# 1. let the user choose what to show
# either click or select bar
# 2. show the map
# 3. download

# todo: opacity of the background
# todo: different color for different nodes
# todo: the initial view of the map
