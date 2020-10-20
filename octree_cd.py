import pcl
import numpy as np
import os
from pyntcloud import PyntCloud as pynt
import pandas as pd
def main():
    resolution = 1
    cloudB = pcl.load("full_cloud_tester.pcd")
    cloudA = pcl.load("no_fences_jitter.pcd")

    octree = cloudA.make_octreeChangeDetector(resolution)
    octree.add_points_from_input_cloud()
    print(cloudA)
    print(cloudB)
    octree.switchBuffers()
    octree.set_input_cloud(cloudB)
    octree.add_points_from_input_cloud()
    newPointIdxVector = octree.get_PointIndicesFromNewVoxels()
    cloudB.extract(newPointIdxVector)
    if len(newPointIdxVector) == 0:
        print("No change detected, check resolution")
        exit(1)
    print((len(newPointIdxVector)))
    outPoints = np.zeros((len(newPointIdxVector), 3), dtype=np.float32)
    for i in range(0, (len(newPointIdxVector))):
        outPoints[i][0] = (cloudB[newPointIdxVector[i]][0])
        outPoints[i][1] = (cloudB[newPointIdxVector[i]][1])
        outPoints[i][2] = (cloudB[newPointIdxVector[i]][2])

    outCloud = pcl.PointCloud()
    outCloud.from_array(outPoints)
    #outputDir = "./output clouds/"
    outputFilename = 'octree_fences' + '.' + 'ply'
    clmns = ['x', 'y', 'z']
    output = pynt(pd.DataFrame(data=outPoints, columns=clmns))
    #if not os.path.exists(outputDir):
    #    os.makedirs(outputDir)
    pynt.to_file(output, outputFilename)
if __name__ == "__main__":
    main()

def octree_cd(sampleCloud, referenceCloud, res):
    resolution = res
    cloudA = pcl.PointCloud()
    cloudA.from_array(sampleCloud)
    cloudB = pcl.PointCloud()
    cloudB.from_array(referenceCloud)

    octree = cloudA.make_octreeChangeDetector(resolution)
    octree.add_points_from_input_cloud()

    print(cloudA)
    print(cloudB)

    octree.switchBuffers()
    octree.set_input_cloud(cloudB)
    octree.add_points_from_input_cloud()
    newPointIdxVector = octree.get_PointIndicesFromNewVoxels()
    cloudB.extract(newPointIdxVector)

    if len(newPointIdxVector) == 0:
        print("No change detected, check resolution")
        exit(1)
    print((len(newPointIdxVector)))
    outPoints = np.zeros((len(newPointIdxVector), 3), dtype=np.float32)
    for i in range(0, (len(newPointIdxVector))):
        outPoints[i][0] = (cloudB[newPointIdxVector[i]][0])
        outPoints[i][1] = (cloudB[newPointIdxVector[i]][1])
        outPoints[i][2] = (cloudB[newPointIdxVector[i]][2])
    return(outPoints)