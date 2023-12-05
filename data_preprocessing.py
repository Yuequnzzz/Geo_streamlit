"""
Generate necessary data for visualization
date: 05/12/2023
"""

import os
import json
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely import union_all


def get_all_test_id(grid_type):
    """
    This function gets all the test IDs
    :return: list, all the test IDs
    """
    if grid_type == 'MV':
        path_base = grid_type + "/"
        # get all the files in the path
        list_files = os.listdir(path_base)

    else:
        path_base = grid_type + "/"
        list_files = []
        list_folders = os.listdir(path_base)
        for j in range(len(list_folders)):
            path = path_base + list_folders[j] + "/"
            sub_files = os.listdir(path)
            list_files += sub_files

    # get all the possible test IDs, that is remove the "_nodes" and "_edges" from the file names
    list_ids = [i[:-6] for i in list_files]
    # remove the duplicates
    list_ids = list(set(list_ids))
    # sort the list
    list_ids.sort()

    # save the list
    with open('data_processing/list_test_id_' + grid_type + '.json', 'w') as fp:
        json.dump(list_ids, fp)

    return list_ids


def classify_grids(cantons, grid_type):
    """
    This function classifies the grids into different cantons
    """
    # for LV network, create a dictionary to store the test ID and the folder name
    path_base = "LV/"
    list_folders = os.listdir(path_base)
    dict_test_id_folder = {}
    for j in range(len(list_folders)):
        path = path_base + list_folders[j] + "/"
        sub_files = os.listdir(path)
        # create a dictionary, the key is the test ID, the value is the folder name
        sub_dict = {i[:-6]: list_folders[j] for i in sub_files}
        dict_test_id_folder.update(sub_dict)

    # get all the test IDs
    test_id_lists = get_all_test_id(grid_type)
    # create a dataframe to store the canton name and the test IDs in the canton
    df_record = pd.DataFrame(np.zeros((len(test_id_lists), len(cantons['NAME']))), columns=list(cantons['NAME']))
    df_record.index = test_id_lists

    # record the number of nodes in each canton
    for i in test_id_lists:
        # read the nodes
        if grid_type == 'MV':
            path = grid_type + "/"
        else:
            path = grid_type + "/" + dict_test_id_folder[i] + "/"
        file_name = i + '_nodes'
        nodes_gpd = gpd.read_file(path + file_name)
        # check if the nodes are in the canton
        for k, c in enumerate(cantons['NAME']):
            nodes_gpd['in_canton'] = nodes_gpd['geometry'].within(cantons[cantons['NAME'] == c].geometry[k])
            df_record.loc[i, c] = nodes_gpd['in_canton'].sum()

    # for each grid, find the canton with the largest number of nodes
    grid_canton_belongs = df_record.idxmax(axis=1)

    # return the index name of each canton, store it in a dictionary
    dict_canton_grid = {}
    for i in cantons['NAME']:
        dict_canton_grid[i] = list(grid_canton_belongs[grid_canton_belongs == i].index)

    # save the dictionary
    with open('data_processing/dict_canton_grid_' + grid_type + '.json', 'w') as fp:
        json.dump(dict_canton_grid, fp)
    with open('data_processing/file_folder_lv.json', 'w') as fp:
        json.dump(dict_test_id_folder, fp)
    return


if __name__ == '__main__':
    # get all the test IDs for MV and LV
    mv_test_list, lv_test_list = get_all_test_id('MV'), get_all_test_id('LV')

    # classify the grids into different cantons
    canton_gpd = gpd.read_file('cantons.geojson')
    # get all the canton names
    list_canton_names = list(canton_gpd['NAME'].drop_duplicates())
    list_canton_names.sort()
    # create a new dataframe keeping the canton name and the geometry
    canton_geo = pd.DataFrame(columns=['NAME', 'geometry'])
    canton_geo['NAME'] = list_canton_names
    # get the union canton boundary
    for i in list_canton_names:
        canton_geo.loc[canton_gpd['NAME'] == i, 'geometry'] = union_all(canton_gpd[canton_gpd['NAME'] == i].geometry)

    classify_grids(canton_geo, 'MV')
    classify_grids(canton_geo, 'LV')
