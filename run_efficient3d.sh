#!/bin/bash
#SBATCH --ntasks-per-node=6
#SBATCH --mem=46G
#SBATCH --gres=gpu:1
#SBATCH -t 100:00:00
#SBATCH -p hm
#SBATCH -o /trinity/home/mzijta/output/out_%j.log
#SBATCH -e /trinity/home/mzijta/output/error_%j.log
##SBATCH --exclude gpu006
##SBATCH --nodelist=gpu002

module purge
module load Python/3.7.4-GCCcore-8.3.0
module load CUDA/11.3.1
module load cuDNN/8.2.1.32-CUDA-11.3.1
module load TensorFlow/2.2.0-fosscuda-2019b-Python-3.7.4


source /trinity/home/mzijta/venvs/anomalib_env2/bin/activate


export PYTHONPATH="/trinity/home/mzijta/venvs/anomalib_env2/lib/python3.7/site-packages"
#export CUDA_VISIBLE_DEVICES=''

#python prepro_lca/do_prepro.py --set "test" --crop "center"
echo "hello_world"
echo $SLURM_JOB_ID

python efficientad_3d.py