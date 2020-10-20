#Author: Quinn Graehling
#Filename: main2.py
#Purpose: Perform various functions on
#point clouds for explicit purpose of
#registration and change detection on
#aerial point cloud data
#Date created: 9/29/2020
#Last modified: 9/29/2020

#imports
import numpy as np
import eigenvector
import time
import pcl_icp
import laspy as lp
from pyntcloud import PyntCloud as pynt
import os
import pandas as pd
import math
import argparse
import octree_cd
import sys
#Global variables

#TODO: Check approved inputs and update (specifically las)
def file_in(filename):
    fileSplit = filename.split('.')
    filePath = fileSplit[0]
    fileType = fileSplit[1]
    acceptedTypes = ['asc','pts','txt','csv','xyz','las','npy','npz','obj','off','pcd','ply']
    if fileType not in acceptedTypes:
        print('File format not supported')
        exit(0)
    file = pynt.from_file(filename)
    npArray = np.zeros((len(file.points), 3), dtype=np.float32)
    npArray[:, 0] = file.points['x'].values
    npArray[:, 1] = file.points['y'].values
    npArray[:, 2] = file.points['z'].values
    return (npArray)


#TODO: ensure that pcl clouds can be converted/outputted
def file_out(points, fileType, filename):
    outputDir = "./output clouds/"
    outputFilename = filename + '.' + fileType
    clmns = ['x', 'y', 'z']
    output = pynt(pd.DataFrame(data=points, columns=clmns))
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    pynt.to_file(output, outputDir + outputFilename)
    return


#TODO: add option for gaussian vs random
def jitter(npCloud, noise_sigma):
    gaussJitter = np.random.normal(0, .01, (np.size(npCloud, 0), 3))
    npCloud = npCloud + gaussJitter
    #npCloud += np.random.randn(np.size(npCloud, 0), dim) * noise_sigma
    return(npCloud)


#TODO: add code, check pcl for possible noise removal code
def noise_removal():
    return


#TODO: add code
def registration(sampleCloud, referenceCloud):
    pcl_icp.pcl_icp(sampleCloud, referenceCloud)
    return


def edge_extract(npCloud, kN, thresh):
    start = time.time()
    outCloud = eigenvector.eigen_edges(npCloud, kN, thresh)
    print('Edge extraction complete!')
    print('Total time: ', time.time()-start)
    return(outCloud)


def point_density_filter(npCloud, kN1, thresh1, kN2, thresh2):
    lowThreshCloud = edge_extract(npCloud, kN1, thresh1)
    highThreshCloud = edge_extract(npCloud, kN2, thresh2)
    lowThreshRows = lowThreshCloud.view([('', lowThreshCloud.dtype)] * lowThreshCloud.shape[1])
    highThreshRows = highThreshCloud.view([('', highThreshCloud.dtype)] * highThreshCloud.shape[1])
    differenceCloud = np.setdiff1d(lowThreshRows, highThreshRows).view(npCloud.dtype).reshape(-1, npCloud.shape[1])
    return(differenceCloud)


def change_detection(sampleCloud, referenceCloud, res):
    output = octree_cd.octree_cd(sampleCloud, referenceCloud, res)
    return(output)




def main():
    parser = argparse.ArgumentParser()
    # add parser arguments
    parser.add_argument("sampleCloud", help="The sample point cloud file to be registered", type=str)
    parser.add_argument("referenceCloud", help="The reference point cloud file to be registered", type=str)
    parser.add_argument("-j", "--jitter", action="store_true", help="Jitter sample file points")
    parser.add_argument("-e", "--edge", action="store_true", help="Perform eigenvector edge extraction")
    parser.add_argument("-d", "--edgeDifference", action="store_true",
                        help="Perform eigenvector edge difference extraction")
    parser.add_argument("-o", "--output", action="store", help="Output file with name")
    parser.add_argument("-i", "--icp", action="store_true", help="Run ICP algorithm")
    parser.add_argument("-las", "--las", action="store", help="Output file with name as type .las")
    parser.add_argument("-oc", "--octreeChange", action="store", help="Perform octree differencing")
    args = parser.parse_args()

    sampleCloud = args.sampleCloud
    referenceCloud = args.referenceCloud
    sampleNP = file_in(sampleCloud)
    referenceNP = file_in(referenceCloud)
    registration(sampleNP, referenceNP)
if __name__ == '__main__':
    main()