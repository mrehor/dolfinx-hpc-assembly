#!/bin/bash
set -ex
shopt -s extglob
shopt -s nullglob

results="results_assembly_routines.csv"

args="--results $results -r 1 --dofs 30000"
args="$args --ufl-forms"

assembler="-t mono"
mpirun -np 1 python3 test_assembly_routines.py $args $assembler --overwrite
mpirun -np 2 python3 test_assembly_routines.py $args $assembler
mpirun -np 4 python3 test_assembly_routines.py $args $assembler

assembler="-t block"
mpirun -np 1 python3 test_assembly_routines.py $args $assembler
mpirun -np 2 python3 test_assembly_routines.py $args $assembler
mpirun -np 4 python3 test_assembly_routines.py $args $assembler

assembler="-t nest"
mpirun -np 1 python3 test_assembly_routines.py $args $assembler
mpirun -np 2 python3 test_assembly_routines.py $args $assembler
mpirun -np 4 python3 test_assembly_routines.py $args $assembler
