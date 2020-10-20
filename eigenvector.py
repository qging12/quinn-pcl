from pyntcloud import PyntCloud
import numpy as np
import pandas as pd
import time
import os
import sys
import pdb

def eigen_edges(npCloud, kN, thresh):
    clmns = ['x', 'y', 'z']
    cloud = pd.DataFrame(data=npCloud, columns=clmns)
    cloud = PyntCloud(cloud)
 #   if not os.path.exists(output_dir):
 #       os.makedirs(output_dir)

    # define hyperparameters
    #kN = 50
    #thresh = 0.2

    pcd_np = np.zeros((len(cloud.points), 6))
    # find neighbors
    kdtree_id = cloud.add_structure("kdtree")
    k_neighbors = cloud.get_neighbors(k=kN, kdtree=kdtree_id)
    # calculate eigenvalues

    cloud.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

    x = cloud.points['x'].values
    y = cloud.points['y'].values
    z = cloud.points['z'].values

    e1 = cloud.points['e3(' + str(kN + 1) + ')'].values
    e2 = cloud.points['e2(' + str(kN + 1) + ')'].values
    e3 = cloud.points['e1(' + str(kN + 1) + ')'].values

    sum_eg = np.add(np.add(e1, e2), e3)
    sigma = np.divide(e1, sum_eg)
    sigma_value = sigma

    # Save the edges and point cloud
    thresh_min = sigma_value < thresh
    sigma_value[thresh_min] = 0
    thresh_max = sigma_value > thresh
    sigma_value[thresh_max] = 255

    pcd_np[:, 0] = x
    pcd_np[:, 1] = y
    pcd_np[:, 2] = z
    pcd_np[:, 3] = sigma_value

    edge_np = np.delete(pcd_np, np.where(pcd_np[:, 3] == 0), axis=0)
    edge_np = np.delete(edge_np, 5, 1)
    edge_np = np.delete(edge_np, 4, 1)
    edge_np = np.delete(edge_np, 3, 1)
    #clmns = ['x', 'y', 'z', 'red', 'green', 'blue']
    #pcd_pd = pd.DataFrame(data=pcd_np, columns=clmns)
    #pcd_pd['red'] = sigma_value.astype(np.uint8)

    # pcd_points = PyntCloud(pd.DataFrame(data=pcd_np,columns=clmns))
    #pcd_points = PyntCloud(pcd_pd)
    #edge_points = PyntCloud(pd.DataFrame(data=edge_np, columns=clmns))
    return(edge_np)