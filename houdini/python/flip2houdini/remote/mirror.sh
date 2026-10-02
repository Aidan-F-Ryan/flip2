#!/bin/bash
# flip2 Solver's remote bake, on this side (solver.py): every couple of seconds, brings the job's logs, its cache.json and its exported and meshed frames
# back from the remote machine, until the job leaves logs/finished there; then once more, and stops. $FLIP2_RSH is the ssh command rsync goes through.
# With "replace", for a bake from the start, frames here that the remote machine hasn't got are deleted: they're an earlier bake's. Without it, for a
# bake resumed or meshed again, they're kept: the remote copy may have been deleted since they were brought back.
#
#   mirror.sh HOST REMOTE_JOB_DIR LOCAL_DIR [replace]
host=$1
remote=$2
local=$3
replacing=()
[ "$4" = replace ] && replacing=(--delete)
rsh=${FLIP2_RSH:-ssh}
mkdir -p "$local/logs" "$local/bake/export/houdini"
sync(){
    rsync -a -e "$rsh" "$host:$remote/logs/" "$local/logs/" 2>> "$local/logs/mirror.log"
    rsync -a -e "$rsh" "$host:$remote/bake/cache.json" "$local/bake/" 2> /dev/null
    rsync -a "${replacing[@]}" -e "$rsh" "$host:$remote/bake/export/houdini/" "$local/bake/export/houdini/" 2> /dev/null
}
while :; do
    sync
    [ -f "$local/logs/finished" ] && break
    sleep 2
done
sync
