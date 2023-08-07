#!/bin/sh
#SBATCH -p batch
#SBATCH -n 2          # number of cores (here 2 cores requested)
#SBATCH -c 2
#SBATCH -N 1
#SBATCH --time=12:00:00 # time allocation, which has the format (D-HH:MM), here set to 1 hou
#SBATCH --mem=128GB # specify memory required per node (here set to 16 GB)

# Notification configuration
#SBATCH --mail-type=END           # Send a notification email when the job is done (=END)
#SBATCH --mail-type=FAIL          # Send a notification email when the job fails (=FAIL)
#SBATCH --mail-user=a1798528@adelaide.edu.au

# Output file location
#SBATCH --output="_logs/log_static_%j.out"
#SBATCH --job-name="static_test"

echo ":: Running script";
module load arch/haswell
module load Gurobi/9.0.2
module load Anaconda3/2020.07

source activate honey_env

# Strange workaround - not sure why it works. Reference:
# https://stackoverflow.com/questions/36733179/
# Without it, the wrong python version is used (default instead of conda)
# You can try to remove and see if it still works
#sbatch --export=ALL,a="r2000",b=10,c=50,d="mixed_attack" competent_test.sh
source deactivate honey_env

source activate honey_env
echo ":: Python loaded";
echo "environment loaded"
echo "run train"
python3 driver_competent.py --fn $a --budget $b  --start $c --algo $d