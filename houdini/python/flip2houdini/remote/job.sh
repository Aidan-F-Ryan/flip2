#!/bin/bash
# flip2 Solver's remote bake, on the machine that bakes (solver.py): runs flip2 with the given arguments in this job's directory, with flip2 export
# following it, so each frame it commits becomes a .bgeo file as it lands. When the bake ends, exports whatever it committed last and leaves
# logs/finished holding its exit status. Started detached (nohup), so the ssh session that starts it can end.
#
#   job.sh FLIP2 bake scene.json --out bake --overwrite        or        job.sh FLIP2 resume bake
cd "$(dirname "$0")" || exit 1
program=${1/#\~/$HOME}
shift
mkdir -p logs
rm -f logs/finished
"$program" "$@" > logs/bake.events.jsonl 2> logs/bake.log &
bake=$!
echo "$bake" > logs/bake.pid
"$program" export bake --follow > logs/export.events.jsonl 2> logs/export.log &
exporter=$!
wait "$bake"
status=$?
kill "$exporter" 2> /dev/null
wait "$exporter" 2> /dev/null
"$program" export bake >> logs/export.events.jsonl 2>> logs/export.log
echo "$status" > logs/finished
