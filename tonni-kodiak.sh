# Navigate to current folder
cd $PBS_O_WORKDIR
pwd

# Activate Conda environment
module load use.own
conda -V
eval "$(conda shell.bash hook)"
conda activate sg
python -V

# Run python script
python comparison_runner.py
