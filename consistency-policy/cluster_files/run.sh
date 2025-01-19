#! /bin/bash
. /opt/miniconda3/etc/profile.d/conda.sh
conda activate consistency-policy
pip install -e $WORK/trifingermujocoenv
pip install scikit-learn
export WANDB_MODE=online
python3 $@
