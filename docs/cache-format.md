# flip2 cache format, version 1

A flip2 bake writes everything into one output directory, the **cache**. DCC importers, pipeline tools and `flip2 resume` read it, and it can be read while the bake is still running. This document is the contract: version 1 is frozen, and the rules for later versions are under [Compatibility](#compatibility).

The reference implementations are `engine/cacheWriter.cu` (writer, and the C++ reader `readShard`) and `tools/flip2cache.py` (Python reader).

## Layout

```
DIR/
  cache.json                        the bake: scene, domain, ranks, build, progress
  frames/0042/commit.json           frame 42's commit record, written last
  frames/0042/particles.r003.f2p    rank 3's particles at frame 42 (a shard)
  checkpoints/0040/ckpt.json        checkpoint 40's record, written last
  checkpoints/0040/state.r003.f2p   rank 3's whole state at the end of frame 40 (a shard)
  frames.discarded/0043/            frames a resume worked out again, kept for comparison
```

- Frame and checkpoint directories are named by frame number in decimal, zero-padded to at least 4 digits. Frame 0 is the initial state.
- Shards are named `<base>.r<rank>.f2p`, with the rank zero-padded to at least 3 digits. Each rank writes its own. A frame has one shard per rank, and every particle is in exactly one of them.
- Readers ignore any other file. Files ending in `.tmp` are writes in progress. `particles.rNNN.json` and `state.rNNN.json` are notes ranks leave for rank 0 while committing; they aren't part of the format and may or may not be there.

## Reading while the bake runs

**A frame exists once its `commit.json` exists.**
- The writer flushes every shard to disk, then writes the commit record to a temporary file, flushes it and renames it into place.
- So a commit record is always whole, and every shard it lists is on disk with the size and hash it gives.
- A frame directory without `commit.json` is unfinished. It may hold partial or stale shards, so don't read them.

**`cache.json` can lag.** Its `committed` field is updated at least once a second and whenever the writer catches up. To follow a bake closely, watch for commit records, or for the `committed` events `flip2 bake` prints on its standard output.

**After a crash or kill:** every frame with a commit record is intact, and the newest checkpoint with a `ckpt.json` is complete.

## cache.json

| Key | Meaning |
|---|---|
| `flip2` | `"cache"` |
| `version` | `1` |
| `scene.path`, `scene.xxh64` | The scene file, as an absolute path, and the XXH64 of its bytes. `flip2 resume` refuses to resume if the hash has changed. |
| `fps` | Frames per second. Frame *n* is at time *n*/`fps` seconds. |
| `frames` | The last frame the bake will make. |
| `committed` | The last frame committed, with every frame before it committed too; `-1` before frame 0. |
| `checkpoint` | The newest committed checkpoint's frame, or `-1`. |
| `worldSize` | Ranks (partitions) writing the bake. |
| `partitionPlanes` | `worldSize + 1` numbers: each rank's first node plane along z, then the last rank's end. Rank *r* holds the particles with node z in `[planes[r], planes[r+1])`. |
| `nodes`, `nodeSize`, `voxelSize`, `domainMin` | The simulation grid: node counts along x, y, z, node size in metres (4 voxels), voxel size, and the domain's minimum corner. |
| `gpu`, `sm`, `cudaRuntime`, `build` | The GPU and build that made it. |
| `shards` | `format` `"f2p"`, `version` `1`, `compression` (`"blosc-zstd"`, `"blosc-lz4"` or `"none"`), and the attributes of frame shards. |

## commit.json

```json
{"flip2":"commit","version":1,"frame":42,"time":1.75,"particles":1240000,
 "shards":[{"file":"particles.r000.f2p","rank":0,"particles":413000,"bytes":7340032,
            "xxh64":"6a2ff317e0304ec2","low":[0.0,0.0,0.0],"high":[0.25,0.12,0.25]}, ...]}
```

`time` is the simulated time in seconds. `shards` lists the shards in rank order. Each entry gives:
- the file's name;
- the rank that wrote it;
- its particle count;
- its size in bytes;
- the XXH64 of the whole file, as 16 lowercase hex digits;
- the bounds of its positions.

`flip2 verify DIR` checks every committed frame's shards against their entries.

## Shards (.f2p)

A shard is little-endian and is laid out in this order:
1. a 64-byte header;
2. a 64-byte entry per attribute;
3. each attribute's data.

**Header**

| Offset | Type | Field |
|---|---|---|
| 0 | char[8] | magic `FLIP2SHD` |
| 8 | uint32 | version, `1` |
| 12 | uint32 | attribute count |
| 16 | uint64 | particle count *n* |
| 24 | int32 | frame |
| 28 | uint32 | rank |
| 32 | uint32 | world size |
| 36 | uint32 | flags, `0` |
| 40 | float32[3] | positions' low x, y, z |
| 52 | float32[3] | positions' high x, y, z |

**Attribute entry**

| Offset | Type | Field |
|---|---|---|
| 0 | char[16] | name, zero-padded |
| 16 | uint32 | type: `1` float32, `2` float64, `3` uint64. Reserved for later versions: `4` int32, `5` uint8, `6` float16 |
| 20 | uint32 | components *k* |
| 24 | uint32 | codec: `0` raw, `1` blosc |
| 28 | uint32 | reserved, `0` |
| 32 | uint64 | offset of its data from the start of the file |
| 40 | uint64 | stored size of its data |
| 48 | uint64 | raw size: *k* × *n* × the type's size |
| 56 | uint64 | reserved, `0` |

**An attribute's data** starts with *k* uint64 values, each component's stored size. Then come the *k* components, one after another. Each component is a plane of all *n* particles' values for that component, in the shard's particle order.

- **Raw:** each plane is the *n* values as they are.
- **Blosc:** each plane is a single blosc (version 1 frame format) buffer that decompresses to the *n* values. The bake uses zstd with bit shuffle by default, or lz4 with byte shuffle. Blosc's own header says which, so any blosc 1 or 2 decompressor reads either.

**Frame shards** have these attributes:

| Name | Type | Components | Meaning |
|---|---|---|---|
| `P` | float32 | 3 | position in metres, x y z |
| `v` | float32 | 3 | velocity in metres per second |
| `id` | uint64 | 1 | the particle's id: it keeps it for as long as it exists, and no other particle in the bake ever has it |
| `age` | float32 | 1 | seconds since the particle came to be: the frame's time for the fluid the scene started with, less for emitted fluid |

`id` and `age` are there unless the scene's `output.attributes` leaves them out; `cache.json`'s `shards.attributes` lists what a bake's frames carry. Caches made before ids existed have neither.

**Checkpoint shards** have these attributes:

| Name | Type | Components | Meaning |
|---|---|---|---|
| `P` | float64 | 3 | position in metres, x y z |
| `v` | float32 | 3 | velocity in metres per second |
| `c` | float32 | 9 | with APIC only: velocity gradient, component c along axis a at 3c + a, per second |
| `id` | uint64 | 1 | the particle's id |
| `birth` | float32 | 1 | when the particle came to be, in simulated seconds |

**Ids.** The fluid a scene starts with is numbered from 0 in the order the scene gives it; each particle an emitter makes takes the next number. The numbering doesn't depend on how the domain is split between ranks: the same bake on 1 GPU or 4 gives every particle the same id. Ids aren't reused when sinks or open faces delete particles, so they have gaps. Today they fit in 32 bits (the engine keeps 32 per particle), and `flip2 export` writes them to Houdini as int32 `id`.

A particle's order within a shard changes from frame to frame, so match particles across frames by `id`, not by index. Concatenating a frame's shards in rank order gives the order a single partition would have held the particles in.

## ckpt.json

```json
{"flip2":"checkpoint","version":1,"frame":40,"time":1.6666666666666667,"substep":152,"apic":false,"nextId":1240000,"acceleration":23.5,"particles":1240000,
 "worldSize":3,"partitionPlanes":[0,10,20,32],"sceneXxh64":"0e648f55050d9cfa","build":"a68ccce",
 "states":[{"file":"state.r000.f2p","rank":0,"particles":413000,"bytes":12582912,"xxh64":"...","low":[...],"high":[...]}, ...]}
```

| Key | Meaning |
|---|---|
| `time` | Simulated seconds at the end of the frame. |
| `substep` | Substeps taken so far; it seeds the emitters' jitter. |
| `apic` | Whether the states carry `c`. |
| `nextId` | The id the next new particle gets. A checkpoint from before ids existed has none, and a resume from it numbers the particles afresh. |
| `acceleration` | The most the last substep accelerated the fluid at, in m/s², which bounds the next substep's length. A checkpoint from before it was kept has none, and the substep after a resume from it goes by the forces alone. |
| `partitionPlanes` | The checkpoint's split, as in `cache.json`. A resume can use a different split. |
| `states` | The state shards in rank order, listed like a commit record's `shards`. |

`flip2 resume` carries on from the newest checkpoint and gives exactly the frames the uninterrupted bake would have, whatever split it resumes with. The bake keeps its newest checkpoints, 2 by default. It deletes older ones once a newer one is committed.

## Compatibility

Version 1 is frozen. Later versions keep these rules:

- **Adding an attribute** to frame or checkpoint shards doesn't change any version. Particle ids arrived this way, as `id` of type uint64, with `age` and `birth`.
- **Readers skip attributes they don't know,** including ones whose type is reserved above. They rely only on the attributes listed here.
- **Adding a key** to a JSON record doesn't change its version. Readers ignore keys they don't know.
- **Any other change bumps a version:**
  - the shard header's `version`, for a change to its binary layout;
  - the record's `version`, for a change to a JSON record's meaning.

  Readers refuse versions newer than they know.
- **Hashes and file names** stay as described here.
