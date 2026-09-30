#!/usr/bin/python3
import open3d
import sys
import open3d.visualization
import pandas
import os
import time
import collections
import numpy as np

class CSVReader:
    def __init__(self, filepath):
        self.data = pandas.DataFrame()
        with open(filepath, "r") as file:
            self.data = pandas.read_csv(file)
        print(self.data)
    def np(self):
        return self.data.to_numpy()

class BinaryReader:     #<frame>.bin: float32 x, y, z per particle
    def __init__(self, filepath):
        self.data = np.fromfile(filepath, dtype=np.float32).reshape(-1, 3)
    def np(self):
        return self.data.astype(np.float64)


def usage():
    print("usage: python3 visualizeSavedPoints.py <dir> <numFrames>")
    print("plays the frames in realtime (24 fps, skipping frames if drawing can't keep up) on a loop; space pauses, closing the window quits")

def main():
    if len(sys.argv) != 3:
        usage()
        return
    pointData = []
    numFrames = int(sys.argv[2])
    for i in range(numFrames):
        path = os.path.join(os.path.join(os.getcwd(), sys.argv[1]), str(i))
        pointData.append(BinaryReader(path + ".bin") if os.path.exists(path + ".bin") else CSVReader(path))
    
    #outline everything the run covers, so the camera frames the whole run from the first frame
    lows = np.min([frame.np().min(axis=0) for frame in pointData], axis=0)
    highs = np.max([frame.np().max(axis=0) for frame in pointData], axis=0)
    bounds = open3d.geometry.LineSet.create_from_axis_aligned_bounding_box(open3d.geometry.AxisAlignedBoundingBox(lows, highs))

    paused = [False]
    def togglePause(visualizer):
        paused[0] = not paused[0]
        return False

    visualizer = open3d.visualization.VisualizerWithKeyCallback()
    visualizer.register_key_callback(ord(" "), togglePause)
    visualizer.create_window(window_name="flip2: " + sys.argv[1])
    visualizer.get_render_option().point_size = 1.5
    visualizer.get_render_option().point_color_option = open3d.visualization.PointColorOption.YCoordinate    #colour by height
    pointCloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pointData[0].np()))
    visualizer.add_geometry(bounds)
    visualizer.add_geometry(pointCloud)
    shown = -1
    playhead = 0.0      #simulated seconds played so far
    last = time.time()
    draws = collections.deque()     #when recent frames were drawn, for the status line
    while visualizer.poll_events():     #until the window is closed
        now = time.time()
        if not paused[0]:
            playhead += now - last
        last = now
        frame = int(playhead*24) % len(pointData)   #the frame due now: frames that can't be drawn in time are skipped, so playback stays realtime
        if frame != shown:
            pointCloud.points = open3d.utility.Vector3dVector(pointData[frame].np())
            visualizer.update_geometry(pointCloud)
            shown = frame
            draws.append(now)
            while draws[0] < now - 1:
                draws.popleft()
            print("\rframe %3d/%d, drawing %2d fps of 24" % (frame, len(pointData) - 1, len(draws)), end="", flush=True)
        visualizer.update_renderer()
        time.sleep(0.002)
    visualizer.destroy_window()
    print()

if __name__ == "__main__":
    main()