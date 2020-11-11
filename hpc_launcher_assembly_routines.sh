#!/bin/bash -l
#SBATCH -J AssemblyRoutines
#SBATCH -p batch
#SBATCH --qos=besteffort
#SBATCH --time=0-00:03:00

source $HOME/fenicsx-iris-cluster/env-fenics.sh

echo "== Starting run at $(date)"
echo "== Job name: ${SLURM_JOB_NAME}"
echo "== Job ID: ${SLURM_JOBID}"
echo "== Node list: ${SLURM_NODELIST}"
echo "== Submit dir: ${SLURM_SUBMIT_DIR}"
echo "== Number of tasks: ${SLURM_NTASKS}"

cd $SLURM_SUBMIT_DIR
# srun --mpi=pmi2 "$@"
srun --mpi=pmi2 --cpu_bind=cores "$@"
# spindle srun --mpi=pmi2 --cpu_bind=cores "$@"

echo "== Finished at $(date)"
