#!/usr/bin/tcsh

#SBATCH --partition=gpu
#SBATCH --time=05:00:00
#SBATCH --job-name=test_as
#SBATCH --gpus-per-node=2
#SBATCH --nodelist=galactica
#SBATCH --mem=40G
#SBATCH --output=/users/ramarat/Analyse-de-sentiments/output.txt
#SBATCH --error=/users/ramarat/Analyse-de-sentiments/err.log


conda activate rimsEnv
echo "Running Clean_data.py ... "
python3 /users/ramarat/Analyse-de-sentiments/Clean_data.py
