#!/bin/bash

source venv/bin/activate

BODIES=(64 512 1024 2048 4096)
THREADS=(1 2 4 8 16)

echo "version,bodies,threads,runtime" > results.csv

# function to run program and append to CSV
run_nbody() {
    local prog=$1
    local version=$2
    local bodies=$3
    local threads=$4

    runtime=$(./"$prog" $bodies $threads | grep "RUNTIME" | cut -d= -f2)
    echo "$version,$bodies,$threads,$runtime" >> results.csv
}

# cpu runs
for b in "${BODIES[@]}"; do
    for t in "${THREADS[@]}"; do
        ver=$([ "$t" -eq 1 ] && echo "sequential" || echo "parallel")
        echo "Executing CPU: bodies=$b threads=$t"
        run_nbody "nbody_cpu" "$ver" "$b" "$t"
    done
done

# gpu runs
for b in "${BODIES[@]}"; do
    echo "Executing GPU: bodies=$b"
    run_nbody "nbody_gpu" "gpu" "$b" 0
done

# generate plot and animation
echo "Generating Plot: "
python3 visualize_simulation.py results.csv