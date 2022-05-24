#!/bin/bash

# Slurm job options (job-name, compute nodes, job time)
#SBATCH --job-name=BatteryMOF_Relaxations
#SBATCH --time=24:00:00
#SBATCH --nodes=4
#SBATCH --tasks-per-node=128
#SBATCH --cpus-per-task=1

# Replace [budget code] below with your budget code (e.g. t01)
#SBATCH --account=e05-react-wal
#SBATCH --partition=standard
#SBATCH --qos=standard
#SBATCH --export=none
##SBATCH --reservation=shortqos

# Send email
#SBATCH --mail-type=ALL
#SBATCH --mail-user=m.golomb18@imperial.ac.uk
module load cpe/21.09

export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH


# Set the number of threads to 1
#   This prevents any threaded system libraries from automatically
#   using threading.
export OMP_NUM_THREADS=1


#Set stacksize to unlimited for aims
ulimit -s unlimited

# Launch the parallel job
#   Using 512 MPI processes and 128 MPI processes per node
#   srun picks up the distribution from the sbatch options

module list
ldd /work/e05/e05/mat92/Codes/Aims/Nov21/aims.210716_2_cray.scalapack.mpi.x

srun --nodes=2 --ntasks=256 --distribution=block:block --hint=nomultithread /work/e05/e05/mat92/Codes/Aims/Nov21/aims.210716_2_cray.scalapack.mpi.x > aims_craynov21.out 
