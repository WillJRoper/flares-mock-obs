#!/bin/bash -l
#SBATCH --ntasks 28 # The number of cores you need...
#SBATCH --array=292-480
#SBATCH -p cosma7 #or some other partition, e.g. cosma, cosma6, etc.
#SBATCH -A dp004
#SBATCH --cpus-per-task=1
#SBATCH -J MOCK-CAT #Give it something meaningful.
#SBATCH -o logs/output_cat_job.%A_%a.out
#SBATCH -e logs/error_cat_job.%A_%a.err
#SBATCH -t 2:00:00

# Run the job from the following directory - change this to point to your own personal space on /lustre
cd /cosma7/data/dp004/dc-rope1/FLARES/flares-mock-obs

module purge
#load the modules used to build your program.
module load pythonconda3/4.5.4 gnu_comp/7.3.0 openmpi/3.0.1 hdf5/1.10.3

source activate flares-env

i=$(($SLURM_ARRAY_TASK_ID - 1))

# Run the program
mpirun -np 28 ./make_catalog.py $i sim Total

source deactivate

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,MaxRSS,Elapsed,ExitCode
exit

