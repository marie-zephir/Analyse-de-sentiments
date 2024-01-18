#!/usr/bin/tcsh

#SBATCH --time=09:00:00
#SBATCH --job-name=test_as
#SBATCH --ntasks=5
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=4
#SBATCH --mem=200G
#SBATCH --output=/users/ramarat/cnn-text-classification-pytorch/output_h24.txt
#SBATCH --error=/users/ramarat/cnn-text-classification-pytorch/err_h24.log


conda activate tf

echo "Running cnn.py ... "

python3 /users/ramarat/cnn-text-classification-pytorch/main.py

echo "Done"
