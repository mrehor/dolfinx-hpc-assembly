#!/bin/bash
set -ex
shopt -s extglob
shopt -s nullglob

jobname="HPC-cpp-forms"

LAUNCHER="hpc_launcher_assembly_routines.sh"
LAUNCHER_OPTS="-J ${jobname} --ntasks-per-node 16"
#LAUNCHER_OPTS="${LAUNCHER_OPTS} -C broadwell"
COMMAND="python3 test_assembly_routines.py"
COMMAND_OPTS="--results results_${jobname}.csv -r 10 --dofs 100000"
COMMAND_OPTS="${COMMAND_OPTS} --cpp-forms"

rm -rf results_${jobname}.csv

ASSEMBLER="-t mono"
sbatch -N  1 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N  2 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N  4 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N  8 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N 16 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}

ASSEMBLER="-t block"
sbatch -N  1 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N  2 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N  4 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N  8 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N 16 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}

ASSEMBLER="-t nest"
sbatch -N  1 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N  2 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N  4 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N  8 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
sbatch -N 16 ${LAUNCHER_OPTS} $LAUNCHER $COMMAND ${COMMAND_OPTS} ${ASSEMBLER}
