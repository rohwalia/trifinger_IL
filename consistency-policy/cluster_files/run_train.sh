#! /bin/bash

#SBATCH --job=zsilot
#SBATCH --time=3-00:00  # D-HH:MM
#SBATCH --signal=USR1@300  # signal SIGUSR1 at least 5min before timeout
#SBATCH --output=/home/martius/%u/logs/%j.out
#SBATCH --error=/home/martius/%u/logs/%j.err
#SBATCH --cpus-per-task=10
#SBATCH --mem=200G
#SBATCH --gres=gpu:1
#SBATCH --partition=a100-galvani

# CONFIG
export home=$HOME
export work=$WORK
export local_scratch=$SCRATCH
export project=consistency-policy

#! /bin/bash

# Exit on errors
set -o errexit

# Location
container_src=$work/container.sif
wandb_artifact_location=$work/.wandb/artifact_location
wandb_config_dir=$work/.wandb/config_dir
wandb_artifact_dir=$work/.wandb/artifact_dir
wandb_data_dir=$work/.wandb/data_dir_lift_image
wandb_cache_dir=$work/.wandb/cache_dir


# Ensure output directories exists
mkdir -p $home/logs
mkdir -p $wandb_artifact_location
mkdir -p $wandb_artifact_dir
mkdir -p $wandb_cache_dir
mkdir -p $wandb_config_dir
mkdir -p $wandb_data_dir

# Send some noteworthy information to the output log
echo "Running on node: $(hostname)"
echo "In directory:    $(pwd)"
echo "Starting on:     $(date)"
echo "SLURM_JOB_ID:    ${SLURM_JOB_ID}"

# CONFIG
echo "home: $home"
echo "work: $work"
echo "project: $project"
echo "args:" $@

# Ensure local_scratch
if [ -z $local_scratch ]; then 
    local_scratch=$(mktemp -d)
fi
if [ -z $local_scratch ]; then 
    echo "Error: failed to create local scratch directory" >&2
    exit 1
fi
echo "local_scratch: $local_scratch"

# Start setup
cd $work/$project

# Copy data to local_scratch
container=$local_scratch/container.sif
cp $container_src $container
echo "copied container from $container_src to $container"

# Run script
singularity exec --containall --writable-tmpfs --nv \
    --bind="$work/$project,$work" --pwd="$work/$project" \
    --env WORK=$work \
    --env WANDB_API_KEY=$WANDB_API_KEY \
    --env WANDB_ARTIFACT_LOCATION=$wandb_artifact_location \
    --env WANDB_ARTIFACT_DIR=$wandb_artifact_dir \
    --env WANDB_CACHE_DIR=$wandb_cache_dir \
    --env WANDB_CONFIG_DIR=$wandb_config_dir \
    --env WANDB_DATA_DIR=$wandb_data_dir \
    $container ./cluster_files/run.sh $@ training.output_dir=$wandb_data_dir

# Send some noteworthy information to the output log
echo "Finished at:     $(date)"

# exit
exit 0
