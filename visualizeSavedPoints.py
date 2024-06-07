#!/usr/bin/python3
import open3d
import sys
import open3d.visualization
import pandas
import os
import time
import numpy as np

class CSVReader:
    def __init__(self, filepath):
        self.data = pandas.DataFrame()
        with open(filepath, "r") as file:
            self.data = pandas.read_csv(file)
        print(self.data)
    def np(self):
        return self.data.to_numpy()


def usage():
    print("usage: python3 visualizeSavedPoints.py <dir> <numFrames>")

def main():
    if len(sys.argv) != 3:
        usage()
        return
    pointData = []
    numFrames = int(sys.argv[2])
    for i in range(numFrames):
        pointData.append(CSVReader(os.path.join(os.path.join(os.getcwd(), sys.argv[1]), str(i))))
    
    pointCloud = open3d.geometry.PointCloud()
    while True:
        for i in pointData:
            pointCloud.points = open3d.utility.Vector3dVector(i.np())
            print(np.asarray(pointCloud.points))
            open3d.visualization.draw_geometries([pointCloud])
            time.sleep(1)

if __name__ == "__main__":
    main()