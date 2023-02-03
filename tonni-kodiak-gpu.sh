# Navigate to current folder
cd $PBS_O_WORKDIR
pwd

# Activate GPU
module load cuda11.2/toolkit/11.2.2
module load cudnn8.1-cuda11.2/8.1.1.33
export CUDA_VISIBLE_DEVICES=`idlegpu`
echo "Program running on device $CUDA_VISIBLE_DEVICES"

# Activate Conda environment
module load use.own
conda -V
eval "$(conda shell.bash hook)"
conda activate sg
python -V

# Run python script
python test_gpu.py
