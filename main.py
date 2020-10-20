# Author: Quinn Graehling
# File: test.py
# Purpose: Format and input 2 point clouds into
# the ICP algorithm for registration
# Date: 6/26/2020

#imports
import numpy as np
import pcl_icp
import eigenvector
import time
import sys
import laspy as lp
from pyntcloud import PyntCloud as pynt
import os
import pandas as pd
import math
import argparse
import open3d as o3d
import multiprocessing as mp
#global variables
N = 100                 #number of random points (if using random points)
dim = 3                 #dimensions within cloud
iter = 100              #number of iterations
translation = .1        #max translation of cloud B if creating random clouds
rotation = .1           #max rotation of cloud B if creating random clouds
tolerance = .00001      #error value for calculating stable icp state
noise = .01             #noise sigma for adding jitter



def file_to_np(filename):
    #convert file to numpy array
    #inputs:
    #filename - filename and directory for input file (see acceptable file types below)
    #outputs:
    #npArray - numpy array represenation of input point cloud
    #currently acceptable file types are:
    #.asc/.pts/.txt/.csv/.xyz (coordinate format must by x y z)
    #.las
    #.npy/npz
    #.obj
    #.off
    #.pcd
    #.ply
    lasFile = pynt.from_file(filename)
    npArray = np.zeros((len(lasFile.points), 3))
    npArray[:, 0] = lasFile.points['x'].values
    npArray[:, 1] = lasFile.points['y'].values
    npArray[:, 2] = lasFile.points['z'].values
    return(npArray)



def np_to_las(npArray, original, filename):
    #convert numpy array to .las file and save as filename (placeholder until np_to_file can support .las)
    #inputs:
    #npArray - numpy represenation of point cloud to be saved
    #original - original file that represents transformed file, used for header creation
    #filename - name you would like to save file as
    #outputs:
    headerFile = lp.file.File(original, mode="r")
    h = headerFile.header
    lasFile = lp.file.File(filename, mode="w", header=h)
    npArray = np.transpose(npArray)
    lasFile.x = npArray[0, :]
    lasFile.y = npArray[1, :]
    lasFile.z = npArray[2, :]
    lasFile.close()
    return



def np_to_file( points, fileType, filename):
    #converts numpy represenation of point cloud to point cloud file and saves it
    #inputs:
    #points - numpy representation of the cloud that is to be saved
    #fileType - the type of point cloud format that the cloud is being saved as (see acceptable file types below)
    #filename - the filename that the output cloud will be saved as
    #outputs:
    #currently acceptable file types are:
    #.asc/.pts/.txt/.csv/.xyz (coordinate format must by x y z)
    #.npy/npz
    #.obj
    #.off
    #.pcd
    #.ply
    outputDir = "./output clouds/"
    outputFilename = filename + '.' + fileType
    clmns = ['x', 'y', 'z']
    output = pynt(pd.DataFrame(data=points, columns=clmns))
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    pynt.to_file(output, outputDir + outputFilename)
    return


#TODO: get partial ICP working
def run_icp(npA, npB):
    #runs the ICP algorithm for cloud registration
    #inputs:
    #npA - numpy array for reference cloud
    #npA - numpy array for registration cloud
    #outputs
    #T - final transform for registration of npA to npB
    #npC - numpy array of npB after transform T applied
    total_time = 0
    start = time.time()
    T, distances, iterations = pcl_icp.icp(npB, npA, max_iterations=iter, error_tolerance=tolerance)
    total_time += time.time() - start
    # make C cloud representation of transformed B
    npC = np.ones((np.size(npB, 0), 4))
    npC[:, 0:3] = np.copy(npB)
    npC = np.dot(T, npC.T).T
    npC = np.delete(npC, 3, 1)
    print('ICP completed!')
    print('Total time: ', total_time)
    print('Time per iteration: ', total_time/iterations)
    return(T, npC)



def rand_cloud():
    #generates a random point cloud if there is no input cloud
    #inputs:
    #outputs:
    #npA - randomly generated reference point cloud
    #npB - npA point cloud after random translation and rotation
    npA = np.random.rand(N, dim)
    npB = np.copy(npA)
    t = np.random.rand(dim) * translation
    npB += t
    R = rotation_matrix(np.random.rand(dim), np.random.rand() * rotation)
    npB = np.dot(R, npB.T).T
    return(npA, npB)



def rotation_matrix(axis, theta):
    #generates the rotation matrix for cloud B if using random clouds
    #inputs:
    #axis - the axis to be rotate
    #theta - scalar value to determine degree of rotation
    #outputs:
    #numpy array of of rotation
    axis = axis/np.sqrt(np.dot(axis, axis))
    a = np.cos(theta/2.)
    b, c, d = -axis*np.sin(theta/2.)

    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])



def jitter(npCloud, noise_sigma):
    #randomly jitters cloud
    #inputs:
    #npCloud - numpy represenation of point cloud that jitter will be added to
    #noise_sigma - scalar value to determine level of jitter to be added to cloud
    #outputs:
    #npCloud - original numpy representation of cloud with randomly added noise
    gaussJitter = np.random.normal(0, .01, (np.size(npCloud, 0), dim))
    npCloud = npCloud + gaussJitter
    #npCloud += np.random.randn(np.size(npCloud, 0), dim) * noise_sigma
    return(npCloud)



def edge_extract(npCloud, kN, thresh):
    #function to extract edges from a point cloud
    #inputs:
    #npCloud - input numpy array
    #outputs:
    #outCloud - numpy representation of eigenvector cloud
    start = time.time()
    outCloud = eigenvector.eigen_edges(npCloud, kN, thresh)
    print('Edge extraction complete!')
    print('Total time: ', time.time()-start)
    return(outCloud)



def point_density_filter(npCloud, kN1, thresh1, kN2, thresh2):
    #Performs filter to remove points high edge density areas such as tree branches that otherwise do not provide any unique information
    #inputs:
    #npCloud - numpy represenation of cloud to be tested
    #thresh1 - threshold for lower extraction cloud (more edges will be extracted)
    #kN1 - number of nearest neighbors used for extraction on low thresh
    #thresh2 - threshold for higher extraction cloud (less edges will be extracted, mainly trees)
    #kN2 - number of nearest neighbors used for extraction on high thresh
    #outputs:
    #differenceCloud - cloud with difference taken between thresholds 1 and 2
    lowThreshCloud = edge_extract(npCloud, kN1, thresh1)
    highThreshCloud = edge_extract(npCloud, kN2, thresh2)
    lowThreshRows = lowThreshCloud.view([('', lowThreshCloud.dtype)] * lowThreshCloud.shape[1])
    highThreshRows = highThreshCloud.view([('', highThreshCloud.dtype)] * highThreshCloud.shape[1])
    differenceCloud = np.setdiff1d(lowThreshRows, highThreshRows).view(npCloud.dtype).reshape(-1, npCloud.shape[1])
    return(differenceCloud)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #add parser arguments
    parser.add_argument("sampleCloud", help="The sample point cloud file to be registered", type=str)
    parser.add_argument("referenceCloud", help="The reference point cloud file to be registered", type=str)
    parser.add_argument("-j", "--jitter", action="store_true", help="Jitter sample file points")
    parser.add_argument("-e", "--edge",  action="store_true", help="Perform eigenvector edge extraction")
    parser.add_argument("-d", "--edgeDifference", action="store_true", help="Perform eigenvector edge difference extraction")
    parser.add_argument("-o", "--output", action="store", help="Output file with name")
    parser.add_argument("-i", "--icp", action="store_true", help="Run ICP algorithm")
    parser.add_argument("-las", "--las", action="store", help="Output file with name as type .las")
    #parse arguements
    args = parser.parse_args()
    sampleCloud = args.sampleCloud
    referenceCloud = args.referenceCloud
    sampleNP = file_to_np(sampleCloud)
    #referenceNP = file_to_np(referenceCloud)

    if args.jitter:
        sampleNP = jitter(sampleNP, noise)

    if args.edge:
        sampleNP = edge_extract(sampleNP, 50, .03)

    if args.edgeDifference:
        sampleNP = point_density_filter(sampleNP, 50, .01, 50, .03)

    if args.icp:
        T, sampleNP = run_icp(sampleNP, referenceNP)

    if args.output:
        np_to_file(sampleNP, "ply", args.output)

    if args.las:
        np_to_las(sampleNP, sampleCloud, args.las)