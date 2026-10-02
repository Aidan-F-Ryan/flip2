#!/bin/bash
# flip2 Solver's remote bake, on the machine that bakes (solver.py): runs flip2 with the given arguments in this job's directory, with flip2 export
# following it, so each frame it commits becomes a .bgeo file as it lands, and with --mesh, flip2 mesh following it too (OPTIONS are mesh's), so each
# frame's surface does as well. When the bake ends, exports and meshes whatever it committed last and leaves logs/finished holding its exit status.
# Started detached (nohup), so the ssh session that starts it can end. "mesh" alone meshes the frames already baked again, replacing their surfaces.
#
#   job.sh FLIP2 [--mesh OPTIONS] bake scene.json --out bake --overwrite      job.sh FLIP2 [--mesh OPTIONS] resume bake      job.sh FLIP2 --mesh OPTIONS mesh
cd "$(dirname "$0")" || exit 1
program=${1/#\~/$HOME}
shift
meshing=
if [ "$1" = --mesh ]; then
    meshing=($2)    # numbers and option names only (solver.py), split into words
    shift 2
fi
mkdir -p logs
rm -f logs/finished
if [ "$1" = mesh ]; then
    "$program" mesh bake --overwrite "${meshing[@]}" > logs/mesh.events.jsonl 2> logs/mesh.log &
    mesher=$!
    echo "$mesher" > logs/bake.pid
    wait "$mesher"
    echo "$?" > logs/finished
    exit 0
fi
"$program" "$@" > logs/bake.events.jsonl 2> logs/bake.log &
bake=$!
echo "$bake" > logs/bake.pid
"$program" export bake --follow > logs/export.events.jsonl 2> logs/export.log &
exporter=$!
mesher=
if [ -n "$meshing" ]; then
    "$program" mesh bake --follow "${meshing[@]}" > logs/mesh.events.jsonl 2> logs/mesh.log &
    mesher=$!
fi
wait "$bake"
status=$?
kill "$exporter" $mesher 2> /dev/null
wait "$exporter" $mesher 2> /dev/null
"$program" export bake >> logs/export.events.jsonl 2>> logs/export.log
if [ -n "$meshing" ]; then
    "$program" mesh bake "${meshing[@]}" >> logs/mesh.events.jsonl 2>> logs/mesh.log
fi
echo "$status" > logs/finished
