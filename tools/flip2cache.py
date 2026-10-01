#Copyright 2023 Aberrant Behavior LLC

"""Reads a flip2 bake's cache (engine/cacheWriter.hu): cache.json, each frame's commit record and its shards.

    python3 tools/flip2cache.py info DIR            what made the cache and what's in it
    python3 tools/flip2cache.py frame DIR N         frame N's particles: count, bounds, mean speed

Raw shards need nothing beyond the standard library; with numpy installed, attributes come back as (n, 3) arrays. Compressed shards need the blosc module
(pip install blosc). A frame without a commit record is unfinished, and isn't read."""

import json
import os
import struct
import sys
from array import array

HEADER = struct.Struct("<8sIIQiIII3f3f")      #64 bytes
ATTRIBUTE = struct.Struct("<16sIIIIQQQQ")     #64 bytes
CODEC_RAW, CODEC_BLOSC = 0, 1

try:
    import numpy
except ImportError:
    numpy = None


def _known(record, path):
    if record.get("version", 1) > 1:
        raise ValueError("%s: version %s, newer than this reader knows (1)" % (path, record["version"]))
    return record


def load_cache(directory):
    path = os.path.join(directory, "cache.json")
    with open(path) as file:
        return _known(json.load(file), path)


def frame_directory(directory, frame):
    return os.path.join(directory, "frames", "%04d" % frame)


def committed_frames(directory):
    """the frames with a commit record, in order"""
    frames = []
    root = os.path.join(directory, "frames")
    for name in os.listdir(root) if os.path.isdir(root) else []:
        if name.isdigit() and os.path.exists(os.path.join(root, name, "commit.json")):
            frames.append(int(name))
    return sorted(frames)


def read_commit(directory, frame):
    path = os.path.join(frame_directory(directory, frame), "commit.json")
    with open(path) as file:
        return _known(json.load(file), path)


def _decompress(data):
    try:
        import blosc
    except ImportError:
        raise RuntimeError("this cache's shards are compressed with blosc: pip install blosc to read them, or bake with \"compression\": \"none\"")
    return blosc.decompress(data)


def read_shard(path):
    """a shard's header, and per attribute its components' planes, each bytes of float32 (or float64: a checkpoint's positions, see "types")"""
    with open(path, "rb") as file:
        data = file.read()
    magic, version, attributes, particles, frame, rank, world_size, flags, *bounds = HEADER.unpack_from(data, 0)
    if magic != b"FLIP2SHD" or version != 1:
        raise ValueError("%s: not a version 1 flip2 shard" % path)
    shard = {"particles": particles, "frame": frame, "rank": rank, "worldSize": world_size, "low": bounds[:3], "high": bounds[3:], "attributes": {}, "types": {}}
    for index in range(attributes):
        name, kind, components, codec, _, offset, stored, raw, _ = ATTRIBUTE.unpack_from(data, HEADER.size + index*ATTRIBUTE.size)
        name = name.rstrip(b"\0").decode()
        if kind not in (1, 2):      #a type a later version added: skipped (docs/cache-format.md)
            continue
        shard["types"][name] = "f" if kind == 1 else "d"
        sizes = struct.unpack_from("<%dQ" % components, data, offset)
        at = offset + 8*components
        planes = []
        for size in sizes:
            block = data[at:at + size]
            at += size
            planes.append(block if codec == CODEC_RAW else _decompress(block))
        shard["attributes"][name] = planes
    return shard


def _column(planes, kind="f"):
    """components' planes as an (n, k) numpy array, or a list of arrays without numpy"""
    if numpy is not None:
        return numpy.stack([numpy.frombuffer(plane, dtype=numpy.float32 if kind == "f" else numpy.float64) for plane in planes], axis=1)
    columns = []
    for plane in planes:
        values = array(kind)
        values.frombytes(plane)
        columns.append(values)
    return columns


def read_frame(directory, frame, attributes=("P", "v")):
    """a committed frame's attributes, every rank's shard in rank order: the order one partition would hold them in"""
    commit = read_commit(directory, frame)
    planes = {name: None for name in attributes}
    for entry in sorted(commit["shards"], key=lambda shard: shard["rank"]):
        shard = read_shard(os.path.join(frame_directory(directory, frame), entry["file"]))
        for name in attributes:
            parts = shard["attributes"][name]
            planes[name] = parts if planes[name] is None else [whole + part for whole, part in zip(planes[name], parts)]
    return {name: _column(value) for name, value in planes.items()}, commit


def main():
    if len(sys.argv) < 3 or sys.argv[1] not in ("info", "frame"):
        print(__doc__)
        return 2
    directory = sys.argv[2]
    cache = load_cache(directory)
    if sys.argv[1] == "info":
        frames = committed_frames(directory)
        print("%s: %s, scene %s (xxh64 %s)" % (directory, cache["build"], cache["scene"]["path"], cache["scene"]["xxh64"]))
        print("  %d ranks, node planes %s, %s nodes of %g m, voxels %g m; %s (sm_%d)" % (cache["worldSize"], cache["partitionPlanes"], cache["nodes"],
              cache["nodeSize"], cache["voxelSize"], cache["gpu"], cache["sm"]))
        print("  %d of %d frames committed (last %d), shards %s" % (len(frames), cache["frames"] + 1, cache["committed"], cache["shards"]["compression"]))
        return 0
    frame = int(sys.argv[3])
    values, commit = read_frame(directory, frame)
    count = commit["particles"]
    if numpy is not None:
        positions, velocities = values["P"], values["v"]
        low, high = positions.min(axis=0) if count else [0]*3, positions.max(axis=0) if count else [0]*3
        speed = float(numpy.linalg.norm(velocities, axis=1).mean()) if count else 0.0
    else:
        positions, velocities = values["P"], values["v"]
        low = [min(column) if count else 0 for column in positions]
        high = [max(column) if count else 0 for column in positions]
        speed = sum((velocities[0][i]**2 + velocities[1][i]**2 + velocities[2][i]**2)**0.5 for i in range(count))/max(count, 1)
    print("frame %d at %.6g s: %d particles in %d shards, from %s to %s, mean speed %.6g m/s" % (frame, commit["time"], count, len(commit["shards"]),
          [round(float(v), 6) for v in low], [round(float(v), 6) for v in high], speed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
