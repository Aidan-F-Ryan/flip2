#!/bin/bash
# flip2 Solver's remote bake, on the machine that bakes (solver.py): runs flip2 with the given arguments in this job's directory, with flip2 export
# following it (OPTIONS are export's), so each frame it commits becomes a .bgeo.sc file as it lands, and with --mesh, flip2 mesh following it too, so
# each frame's surface does as well. When the bake ends, exports and meshes whatever it committed last and leaves logs/finished holding its exit status.
# Started detached (nohup), so the ssh session that starts it can end. "mesh" alone meshes the frames already baked again, replacing their surfaces.
#
#   job.sh FLIP2 [--export OPTIONS] [--mesh OPTIONS] bake scene.json --out bake --overwrite      job.sh FLIP2 [...] resume bake
#   job.sh FLIP2 --mesh OPTIONS mesh
cd "$(dirname "$0")" || exit 1
program=${1/#\~/$HOME}
shift
meshing=
exporting=()
while :; do     # each OPTIONS is numbers, formats and option names only (solver.py), split into words
    case "$1" in
        --mesh) meshing=($2); shift 2;;
        --export) exporting=($2); shift 2;;
        *) break;;
    esac
done
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
if [ "$1" = bake ]; then    # from the start: an earlier bake's cache and what was made of it go first, or what follows this one would take them for its own
    rm -rf bake/cache.json bake/frames bake/frames.discarded bake/checkpoints bake/export
fi
"$program" "$@" > logs/bake.events.jsonl 2> logs/bake.log &
bake=$!
echo "$bake" > logs/bake.pid
"$program" export bake --follow "${exporting[@]}" > logs/export.events.jsonl 2> logs/export.log &
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
"$program" export bake "${exporting[@]}" >> logs/export.events.jsonl 2>> logs/export.log
if [ -n "$meshing" ]; then
    "$program" mesh bake "${meshing[@]}" >> logs/mesh.events.jsonl 2>> logs/mesh.log
fi
echo "$status" > logs/finished
