#!/usr/bin/python3
import open3d
import sys
import open3d.visualization
import pandas
import os
import json
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

class CacheReader:      #a committed frame of a flip2 bake's cache (engine/cacheWriter.hu), through tools/flip2cache.py
    def __init__(self, directory, frame):
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools"))
        import flip2cache
        self.data = flip2cache.read_frame(directory, frame, attributes=("P",))[0]["P"]
    def np(self):
        return self.data.astype(np.float64)


def loadObj(path):  #its vertices, and its faces split into fans of triangles, as the engine reads it (scene.cpp)
    vertices, triangles = [], []
    with open(path) as lines:
        for line in lines:
            words = line.split()
            if not words:
                continue
            if words[0] == "v":
                vertices.append([float(word) for word in words[1:4]])
            elif words[0] == "f":
                corners = [int(word.split("/")[0]) for word in words[1:]]
                corners = [corner - 1 if corner > 0 else len(vertices) + corner for corner in corners]
                triangles += [[corners[0], corners[fan], corners[fan + 1]] for fan in range(1, len(corners) - 1)]
    return np.array(vertices, dtype=np.float64), np.array(triangles, dtype=np.int32)

def loadNpy(path):
    return np.load(path).reshape(-1, 3)

def decompose(matrix):  #a row-major 4x4's translation, per-axis scale and rotation, a mirroring taken as a negative z scale, as the engine does (obstacles.cu)
    m = np.array(matrix, dtype=np.float64).reshape(4, 4)
    scale = np.linalg.norm(m[:3, :3], axis=0)
    rotation = m[:3, :3] / np.where(scale > 0, scale, 1)
    if np.linalg.det(rotation) < 0:
        scale[2] = -scale[2]
        rotation[:, 2] = -rotation[:, 2]
    return m[:3, 3], scale, rotation

def quaternion(r):  #w x y z, from a rotation matrix, by Shepperd's method as the engine does
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0:
        s = 2*np.sqrt(trace + 1)
        return np.array([s/4, (r[2, 1] - r[1, 2])/s, (r[0, 2] - r[2, 0])/s, (r[1, 0] - r[0, 1])/s])
    if r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = 2*np.sqrt(1 + r[0, 0] - r[1, 1] - r[2, 2])
        return np.array([(r[2, 1] - r[1, 2])/s, s/4, (r[0, 1] + r[1, 0])/s, (r[0, 2] + r[2, 0])/s])
    if r[1, 1] > r[2, 2]:
        s = 2*np.sqrt(1 + r[1, 1] - r[0, 0] - r[2, 2])
        return np.array([(r[0, 2] - r[2, 0])/s, (r[0, 1] + r[1, 0])/s, s/4, (r[1, 2] + r[2, 1])/s])
    s = 2*np.sqrt(1 + r[2, 2] - r[0, 0] - r[1, 1])
    return np.array([(r[1, 0] - r[0, 1])/s, (r[0, 2] + r[2, 0])/s, (r[1, 2] + r[2, 1])/s, s/4])

def rotationMatrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
                     [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
                     [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]])

def slerp(q0, q1, blend):   #the short way round
    cosine = np.dot(q0, q1)
    if cosine < 0:
        q1, cosine = -q1, -cosine
    if cosine > 0.9995:
        return (1 - blend)*q0 + blend*q1
    angle = np.arccos(cosine)
    return (np.sin((1 - blend)*angle)*q0 + np.sin(blend*angle)*q1) / np.sin(angle)

def bracket(times, time):   #the keys either side of time, and how far between them it is: held before the first and after the last
    if len(times) == 1 or time <= times[0]:
        return 0, 0, 0.0
    if time >= times[-1]:
        return len(times) - 1, len(times) - 1, 0.0
    key = int(np.searchsorted(times, time, side="right")) - 1
    return key, key + 1, (time - times[key]) / (times[key + 1] - times[key])

class Obstacle:     #one of a scene's obstacles (engine/scene.hpp), as a triangle mesh in the world at any time
    def __init__(self, description, directory):
        place = lambda path: os.path.join(directory, path)
        self.samples = None
        if "mesh" in description:
            self.vertices, self.triangles = loadObj(place(description["mesh"]))
        elif "vertices" in description:
            self.vertices, self.triangles = loadNpy(place(description["vertices"])), loadNpy(place(description["triangles"])).astype(np.int32)
        elif description.get("shape") == "sphere":
            sphere = open3d.geometry.TriangleMesh.create_sphere(radius=description["radius"], resolution=24)
            self.vertices = np.asarray(sphere.vertices) + np.array(description.get("centre", description.get("center")))
            self.triangles = np.asarray(sphere.triangles)
        elif description.get("shape") == "box":
            low, high = np.array(description["min"]), np.array(description["max"])
            self.vertices = np.array([[(high if corner >> axis & 1 else low)[axis] for axis in range(3)] for corner in range(8)])
            self.triangles = np.array([[0, 2, 1], [1, 2, 3], [4, 5, 6], [5, 7, 6], [0, 1, 4], [1, 5, 4], [2, 6, 3], [3, 6, 7], [0, 4, 2], [2, 4, 6], [1, 3, 5], [3, 7, 5]])
        else:
            self.vertices, self.triangles = None, None
        if "deforming" in description:     #its vertices, in the world, at each sample
            self.times = np.array([sample["time"] for sample in description["deforming"]])
            self.samples = []
            for sample in description["deforming"]:
                if "mesh" in sample:
                    vertices, triangles = loadObj(place(sample["mesh"]))
                    if self.triangles is None:
                        self.triangles = triangles
                else:
                    vertices = loadNpy(place(sample["vertices"]))
                self.samples.append(vertices.astype(np.float64))
            return
        keys = description.get("keyframes", [{"time": 0, "transform": description.get("transform", np.eye(4).flatten().tolist())}])
        self.times = np.array([key["time"] for key in keys])
        placed = [decompose(key["transform"]) for key in keys]
        self.vertices = self.vertices*placed[0][1]     #the first key's scale is baked in; then it moves rigidly
        self.translations = [translation for translation, _, _ in placed]
        self.rotations = [quaternion(rotation) for _, _, rotation in placed]

    def moves(self):
        return len(self.times) > 1

    def at(self, time):
        first, second, blend = bracket(self.times, time)
        if self.samples is not None:
            return (1 - blend)*self.samples[first] + blend*self.samples[second]
        rotation = rotationMatrix(slerp(self.rotations[first], self.rotations[second], blend))
        translation = (1 - blend)*self.translations[first] + blend*self.translations[second]
        return self.vertices @ rotation.T + translation


def usage():
    print("usage: python3 visualizeSavedPoints.py <dir> <numFrames> [--scene scene.json]")
    print("<dir> holds a bake's cache (cache.json and frames/; compressed ones need pip install blosc) or its N.bin frames")
    print("plays the frames in realtime (at the scene's fps, or 24, skipping frames if drawing can't keep up) on a loop; space pauses, closing the window quits")
    print("with the scene the frames came from, it also draws its obstacles where they were at each frame, and its domain")

def main():
    args = sys.argv[1:]
    scene = None
    if "--scene" in args:
        at = args.index("--scene")
        if at + 1 >= len(args):
            usage()
            return
        with open(args[at + 1]) as file:
            scene = json.load(file)
        sceneDirectory = os.path.dirname(os.path.abspath(args[at + 1]))
        del args[at:at + 2]
    if len(args) != 2:
        usage()
        return
    pointData = []
    numFrames = int(args[1])
    directory = os.path.join(os.getcwd(), args[0])
    cached = os.path.exists(os.path.join(directory, "cache.json"))
    for i in range(numFrames):
        path = os.path.join(directory, str(i))
        if cached:
            pointData.append(CacheReader(directory, i))
        else:
            pointData.append(BinaryReader(path + ".bin") if os.path.exists(path + ".bin") else CSVReader(path))
    fps = scene.get("fps", 24) if scene else 24
    obstacles = [Obstacle(description, sceneDirectory) for description in scene.get("obstacles", [])] if scene else []

    #outline the domain, or everything the run covers, so the camera frames the whole run from the first frame
    if scene and "domain" in scene:
        lows, highs = np.array(scene["domain"]["min"], dtype=np.float64), np.array(scene["domain"]["max"], dtype=np.float64)
    else:
        lows = np.min([frame.np().min(axis=0) for frame in pointData], axis=0)
        highs = np.max([frame.np().max(axis=0) for frame in pointData], axis=0)
    bounds = open3d.geometry.LineSet.create_from_axis_aligned_bounding_box(open3d.geometry.AxisAlignedBoundingBox(lows, highs))

    paused = [False]
    def togglePause(visualizer):
        paused[0] = not paused[0]
        return False

    visualizer = open3d.visualization.VisualizerWithKeyCallback()
    visualizer.register_key_callback(ord(" "), togglePause)
    visualizer.create_window(window_name="flip2: " + args[0])
    visualizer.get_render_option().point_size = 1.5
    visualizer.get_render_option().point_color_option = open3d.visualization.PointColorOption.YCoordinate    #colour by height
    visualizer.get_render_option().mesh_show_back_face = True
    pointCloud = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(pointData[0].np()))
    visualizer.add_geometry(bounds)
    visualizer.add_geometry(pointCloud)
    meshes = []
    for obstacle in obstacles:
        mesh = open3d.geometry.TriangleMesh(open3d.utility.Vector3dVector(obstacle.at(0.0)), open3d.utility.Vector3iVector(obstacle.triangles))
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.75, 0.75, 0.78])
        visualizer.add_geometry(mesh)
        meshes.append(mesh)
    shown = -1
    playhead = 0.0      #simulated seconds played so far
    last = time.time()
    draws = collections.deque()     #when recent frames were drawn, for the status line
    while visualizer.poll_events():     #until the window is closed
        now = time.time()
        if not paused[0]:
            playhead += now - last
        last = now
        frame = int(playhead*fps) % len(pointData)  #the frame due now: frames that can't be drawn in time are skipped, so playback stays realtime
        if frame != shown:
            pointCloud.points = open3d.utility.Vector3dVector(pointData[frame].np())
            visualizer.update_geometry(pointCloud)
            for obstacle, mesh in zip(obstacles, meshes):
                if obstacle.moves():
                    mesh.vertices = open3d.utility.Vector3dVector(obstacle.at(frame / fps))
                    mesh.compute_vertex_normals()
                    visualizer.update_geometry(mesh)
            shown = frame
            draws.append(now)
            while draws[0] < now - 1:
                draws.popleft()
            print("\rframe %3d/%d, drawing %2d fps of %g" % (frame, len(pointData) - 1, len(draws), fps), end="", flush=True)
        visualizer.update_renderer()
        time.sleep(0.002)
    visualizer.destroy_window()
    print()

if __name__ == "__main__":
    main()
